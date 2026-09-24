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
import math
import os
import random
import time

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

SFT_PATH = os.environ.get("SFT_PATH", "/content/out/final")
OUT = os.environ.get("DPO_OUT", "/content/dpo")
N_TRAIN = int(os.environ.get("N_TRAIN", "12000"))       # preference pairs to train on
N_EVAL = int(os.environ.get("N_EVAL", "500"))
BETA = float(os.environ.get("BETA", "0.1"))
LR = float(os.environ.get("LR", "1e-6"))                  # DPO needs a MUCH smaller LR than SFT
MICRO_BATCH = int(os.environ.get("MICRO_BATCH", "4"))     # pairs per forward pass
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", "8"))       # effective batch = 32 pairs
MAX_PROMPT, MAX_LEN = 512, 1024
TIME_BUDGET_HOURS = float(os.environ.get("TIME_BUDGET_HOURS", "1.0"))
SEED = 42

torch.manual_seed(SEED)
random.seed(SEED)
device = "cuda"
tok = AutoTokenizer.from_pretrained(SFT_PATH)
END = tok.convert_tokens_to_ids("<|im_end|>")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def encode_pair(ex):
    """prompt / chosen / rejected -> token ids.

    The prompt is formatted with the same ChatML template used in SFT; only the
    answer tokens are scored (the prompt is identical for both answers anyway).
    """
    prompt_msgs = ex["chosen"][:-1]
    prompt = tok.apply_chat_template(prompt_msgs, add_generation_prompt=True, tokenize=False)
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"][-MAX_PROMPT:]  # keep the END of long prompts
    out = {"prompt": p_ids}
    for key in ["chosen", "rejected"]:
        ans = tok(ex[key][-1]["content"], add_special_tokens=False)["input_ids"] + [END]
        out[key] = ans[:MAX_LEN - len(p_ids)]
    return out


def collate(pairs):
    """Stack chosen answers then rejected answers into ONE padded batch.

    Returns input_ids, attention_mask and a loss_mask that is 1 only on answer tokens.
    """
    seqs, masks = [], []
    for key in ["chosen", "rejected"]:
        for p in pairs:
            seqs.append(p["prompt"] + p[key])
            masks.append([0] * len(p["prompt"]) + [1] * len(p[key]))
    width = max(map(len, seqs))
    ids = torch.full((len(seqs), width), tok.pad_token_id)
    att = torch.zeros((len(seqs), width), dtype=torch.long)
    lm = torch.zeros((len(seqs), width))
    for i, (s, m) in enumerate(zip(seqs, masks)):
        ids[i, :len(s)] = torch.tensor(s)
        att[i, :len(s)] = 1
        lm[i, :len(m)] = torch.tensor(m, dtype=torch.float)
    return ids.to(device), att.to(device), lm.to(device)


def answer_logprob(model, ids, att, loss_mask):
    """Sum of log-probabilities the model assigns to the answer tokens (log pi(y|x))."""
    logits = model(input_ids=ids, attention_mask=att).logits[:, :-1].float()
    targets = ids[:, 1:]
    logp = torch.gather(F.log_softmax(logits, -1), 2, targets.unsqueeze(-1)).squeeze(-1)
    return (logp * loss_mask[:, 1:]).sum(-1)


def dpo_loss(policy, ref, batch):
    ids, att, lm = batch
    n = ids.shape[0] // 2
    pi = answer_logprob(policy, ids, att, lm)
    with torch.no_grad():                           # the reference model is frozen
        pi_ref = answer_logprob(ref, ids, att, lm)
    rewards = BETA * (pi - pi_ref)                  # implicit rewards
    r_chosen, r_rejected = rewards[:n], rewards[n:]
    loss = -F.logsigmoid(r_chosen - r_rejected).mean()
    return loss, r_chosen.detach(), r_rejected.detach()


@torch.no_grad()
def evaluate(policy, ref, data):
    policy.eval()
    acc, margin, count = 0.0, 0.0, 0
    for i in range(0, len(data), MICRO_BATCH):
        _, rc, rr = dpo_loss(policy, ref, collate(data[i:i + MICRO_BATCH]))
        acc += (rc > rr).float().sum().item()
        margin += (rc - rr).sum().item()
        count += len(rc)
    policy.train()
    return {"reward_accuracy": acc / count, "reward_margin": margin / count}


@torch.no_grad()
def generate(model, prompts):
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
    start = time.time()
    raw = load_dataset("HuggingFaceH4/ultrafeedback_binarized")
    train = raw["train_prefs"].shuffle(seed=SEED).select(range(N_TRAIN))
    test = raw["test_prefs"].shuffle(seed=SEED).select(range(N_EVAL))
    train = [encode_pair(e) for e in train]
    test = [encode_pair(e) for e in test]
    # Drop pairs whose answers were truncated away entirely.
    train = [p for p in train if p["chosen"] and p["rejected"]]
    test = [p for p in test if p["chosen"] and p["rejected"]]
    print(f"train pairs {len(train)}, eval pairs {len(test)}")

    policy = AutoModelForCausalLM.from_pretrained(SFT_PATH, dtype=torch.float32).to(device)
    policy.gradient_checkpointing_enable()
    policy.config.use_cache = False
    ref = AutoModelForCausalLM.from_pretrained(SFT_PATH, dtype=torch.bfloat16).to(device).eval()
    ref.requires_grad_(False)

    prompts = ["Explain the difference between a list and a tuple in Python.",
               "Give me three tips for a job interview.",
               "What should I do if I feel stressed at work?"]
    before = generate(ref, prompts)
    eval_before = evaluate(policy, ref, test)      # policy == ref here -> accuracy ~0 margin 0
    print("before:", eval_before)

    steps = len(train) // (MICRO_BATCH * GRAD_ACCUM)
    opt = torch.optim.AdamW(policy.parameters(), lr=LR, weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(opt, int(0.1 * steps), steps)
    deadline = start + TIME_BUDGET_HOURS * 3600
    log = []
    random.shuffle(train)
    for step in range(steps):
        stats = []
        for micro in range(GRAD_ACCUM):
            i = (step * GRAD_ACCUM + micro) * MICRO_BATCH
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, rc, rr = dpo_loss(policy, ref, collate(train[i:i + MICRO_BATCH]))
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
