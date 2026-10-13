"""
Stage 3 of Mini-ChatGPT: align the SFT model with human preferences using DPO.

-----------------------------------------------------------------------------
WHY A THIRD STAGE?
-----------------------------------------------------------------------------
After SFT the model *can* answer, but it imitates every training answer equally,
good or mediocre. Preference tuning teaches it which of two answers people
prefer (more helpful, more correct, better formatted, safer).

The original ChatGPT recipe used RLHF: train a reward model on human
preferences, then optimise the chat model against it with PPO (reinforcement
learning). That needs 4 models in memory and is notoriously unstable.

DPO (Rafailov et al., 2023, "Direct Preference Optimization: Your Language
Model is Secretly a Reward Model") shows the same objective can be optimised
with a simple classification-style loss directly on preference pairs:

    For a prompt x with a preferred answer y_w and a rejected answer y_l:

      implicit reward  r(y) = beta * [ log pi(y|x) - log pi_ref(y|x) ]
      loss             = -log sigmoid( r(y_w) - r(y_l) )

    * pi      = the model being trained (starts as the SFT model)
    * pi_ref  = a frozen copy of the SFT model
    * beta    = how far pi may drift from pi_ref (0.1 = standard)
    The loss rises the probability of preferred answers and lowers that of
    rejected ones *relative to the reference*, which keeps the model from
    collapsing or forgetting what it learned in SFT. No reward model, no RL.

METRICS we track (on held-out pairs):
    * reward accuracy - % of pairs where r(y_w) > r(y_l)
    * reward margin   - average r(y_w) - r(y_l)

Data: UltraFeedback (binarized) - ~60k prompts, each with a GPT-4-rated
      chosen and rejected answer.
"""
import json
import os
import random
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from mini_chatgpt import dpo_core

# ---------------------------------------------------------------------------
# Configuration (override with environment variables)
# ---------------------------------------------------------------------------
SFT_PATH = os.environ.get("SFT_PATH", "/content/out/final")
OUT = os.environ.get("DPO_OUT", "/content/dpo")
N_TRAIN = int(os.environ.get("N_TRAIN", "12000"))       # preference pairs to train on
N_EVAL = int(os.environ.get("N_EVAL", "500"))
BETA = float(os.environ.get("BETA", "0.1"))
LR = float(os.environ.get("LR", "1e-6"))                  # DPO needs a MUCH smaller LR than SFT
MICRO_BATCH = int(os.environ.get("MICRO_BATCH", "4"))     # pairs per forward pass
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", "8"))       # effective batch = 32 pairs
MAX_PROMPT, MAX_LEN = 512, 1024                           # token caps for the prompt and prompt + answer
TIME_BUDGET_HOURS = float(os.environ.get("TIME_BUDGET_HOURS", "1.0"))
SEED = 42

torch.manual_seed(SEED)
random.seed(SEED)
device = "cuda"
# The SFT output folder contains the tokenizer with the ChatML template attached.
tok = AutoTokenizer.from_pretrained(SFT_PATH)
END = tok.convert_tokens_to_ids("<|im_end|>")


# ---------------------------------------------------------------------------
# Data and loss (logic in mini_chatgpt/dpo_core.py, unit-tested there; these
# wrappers bind it to this run's tokenizer, device and hyper-parameters)
# ---------------------------------------------------------------------------
def encode_pair(ex):
    """prompt / chosen / rejected -> token ids (see dpo_core.encode_pair).

    The prompt is formatted with the same ChatML template used in SFT; only the
    answer tokens are scored (the prompt is identical for both answers anyway).
    """
    return dpo_core.encode_pair(ex, tok, END, MAX_PROMPT, MAX_LEN)


def collate(pairs):
    """Pad a micro-batch (chosen rows, then rejected rows) and move it to the GPU.

    Returns:
        (input_ids, attention_mask, loss_mask) on `device`.
    """
    ids, att, lm = dpo_core.collate_pairs(pairs, tok.pad_token_id)
    return ids.to(device), att.to(device), lm.to(device)


def dpo_loss(policy, ref, batch):
    """DPO loss and implicit rewards for one collated batch (beta = BETA).

    Returns:
        (loss, r_chosen, r_rejected).
    """
    return dpo_core.dpo_loss(policy, ref, batch, BETA)


@torch.no_grad()
def evaluate(policy, ref, data):
    """Preference metrics on held-out pairs.

    Args:
        policy: the model being trained.
        ref: the frozen SFT reference.
        data: list of encoded pairs.

    Returns:
        {"reward_accuracy": fraction of pairs where r(chosen) > r(rejected),
         "reward_margin": mean of r(chosen) - r(rejected)}.
    """
    policy.eval()  # no dropout while measuring
    acc, margin, count = 0.0, 0.0, 0
    for i in range(0, len(data), MICRO_BATCH):
        _, rc, rr = dpo_loss(policy, ref, collate(data[i:i + MICRO_BATCH]))
        acc += (rc > rr).float().sum().item()
        margin += (rc - rr).sum().item()
        count += len(rc)
    policy.train()  # back to training mode for the caller
    return {"reward_accuracy": acc / count, "reward_margin": margin / count}


@torch.no_grad()
def generate(model, prompts):
    """Greedy answers to a few fixed prompts, for a qualitative before/after comparison.

    The model must be in eval mode with the KV cache on (see the note in main()).

    Args:
        model: a causal LM on `device`.
        prompts: list of user messages.

    Returns:
        {prompt: answer text}.
    """
    out = {}
    for p in prompts:
        # transformers v5: apply_chat_template(tokenize=True) returns a dict, so format text first, then tokenize.
        text = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False)
        ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
        g = model.generate(ids, max_new_tokens=250, do_sample=False, eos_token_id=END,
                           repetition_penalty=1.1)
        out[p] = tok.decode(g[0][ids.shape[1]:], skip_special_tokens=True)
    return out


def main():
    """Run DPO end to end: load data and models, train, evaluate, save model + metrics to OUT."""
    start = time.time()
    # 1) Data: sample train/eval preference pairs and tokenize them.
    raw = load_dataset("HuggingFaceH4/ultrafeedback_binarized")
    train = raw["train_prefs"].shuffle(seed=SEED).select(range(N_TRAIN))
    test = raw["test_prefs"].shuffle(seed=SEED).select(range(N_EVAL))
    train = [encode_pair(e) for e in train]
    test = [encode_pair(e) for e in test]
    # Drop pairs whose answers were truncated away entirely.
    train = [p for p in train if p["chosen"] and p["rejected"]]
    test = [p for p in test if p["chosen"] and p["rejected"]]
    print(f"train pairs {len(train)}, eval pairs {len(test)}")

    # 2) Models: the trainable policy (fp32 master weights, bf16 autocast below)
    #    and a frozen bf16 reference copy - both start as the SFT model.
    policy = AutoModelForCausalLM.from_pretrained(SFT_PATH, dtype=torch.float32).to(device)
    policy.gradient_checkpointing_enable()
    policy.config.use_cache = False
    ref = AutoModelForCausalLM.from_pretrained(SFT_PATH, dtype=torch.bfloat16).to(device).eval()
    ref.requires_grad_(False)

    # 3) Baselines before any DPO step: sample answers and preference metrics.
    prompts = ["Explain the difference between a list and a tuple in Python.",
               "Give me three tips for a job interview.",
               "What should I do if I feel stressed at work?"]
    before = generate(ref, prompts)
    # policy == ref here, so every reward is ~0: accuracy is ~chance and the margin ~0.
    eval_before = evaluate(policy, ref, test)
    print("before:", eval_before)

    # 4) Training loop: one optimizer step per GRAD_ACCUM micro-batches (one pass over the data).
    steps = len(train) // (MICRO_BATCH * GRAD_ACCUM)
    opt = torch.optim.AdamW(policy.parameters(), lr=LR, weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(opt, int(0.1 * steps), steps)  # 10% warm-up, then cosine decay
    deadline = start + TIME_BUDGET_HOURS * 3600
    log = []
    random.shuffle(train)
    for step in range(steps):
        stats = []
        for micro in range(GRAD_ACCUM):
            i = (step * GRAD_ACCUM + micro) * MICRO_BATCH
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, rc, rr = dpo_loss(policy, ref, collate(train[i:i + MICRO_BATCH]))
            # Divide so the accumulated gradient is the mean over the effective batch, not the sum.
            (loss / GRAD_ACCUM).backward()
            stats.append((loss.item(), (rc > rr).float().mean().item(), (rc - rr).mean().item()))
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        row = {"step": step + 1, "loss": sum(s[0] for s in stats) / len(stats),
               "reward_accuracy": sum(s[1] for s in stats) / len(stats),
               "reward_margin": sum(s[2] for s in stats) / len(stats)}
        log.append(row)
        if (step + 1) % 10 == 0:
            print(json.dumps(row), f"elapsed {(time.time() - start) / 60:.1f} min", flush=True)
        if time.time() > deadline:
            print(f"[time budget] stopping at step {step + 1}")
            break

    # 5) Evaluate, generate the "after" samples, save everything.
    eval_after = evaluate(policy, ref, test)
    print("after:", eval_after)
    # Generation needs eval mode, no gradient checkpointing and the KV cache ON. In training mode with
    # checkpointing the model silently drops the cache while generate() assumes it exists, so every step after
    # the first sees a single token and the output degenerates into garbage.
    policy.eval()
    policy.gradient_checkpointing_disable()
    policy.config.use_cache = True
    after = generate(policy, prompts)

    os.makedirs(OUT, exist_ok=True)
    policy.to(torch.bfloat16).save_pretrained(os.path.join(OUT, "final"))
    tok.save_pretrained(os.path.join(OUT, "final"))
    summary = {"train_pairs": len(train), "steps": len(log), "beta": BETA, "lr": LR,
               "eval_before": eval_before, "eval_after": eval_after,
               "wall_clock_hours": round((time.time() - start) / 3600, 2)}
    json.dump(summary, open(os.path.join(OUT, "dpo_summary.json"), "w"), indent=2)
    json.dump(log, open(os.path.join(OUT, "dpo_log.json"), "w"))
    json.dump({"sft": before, "dpo": after}, open(os.path.join(OUT, "sft_vs_dpo_samples.json"), "w"), indent=2)
    print(json.dumps(summary, indent=2))
    print("DONE")


if __name__ == "__main__":
    main()
