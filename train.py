"""
Mini-ChatGPT: turn a small *base* language model into a *chat assistant*
with Supervised Fine-Tuning (SFT).

-----------------------------------------------------------------------------
THE BIG PICTURE
-----------------------------------------------------------------------------
ChatGPT-style assistants are built in (roughly) three stages:

  1. Pre-training  - a model reads trillions of tokens of web text and learns
                     to predict the next token. It knows language and facts,
                     but it only *continues* text; it does not *answer*.
  2. SFT           - we show it thousands of example conversations
                     (user asks -> assistant answers) so it learns the
                     "assistant" behaviour and the chat format.   <-- THIS FILE
  3. RLHF / DPO    - optional polishing with human preference data.

Pre-training costs millions of dollars, so we start from an open pre-trained
base model (Qwen2.5-0.5B, 494M parameters) and do stage 2 on a single GPU.

-----------------------------------------------------------------------------
KEY IDEAS IMPLEMENTED BELOW
-----------------------------------------------------------------------------
* Chat template (ChatML): every message is wrapped as
      <|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\nHi!<|im_end|>\n
  so the model can tell who is speaking and where a reply ends.
* Loss masking: we only train on the *assistant* tokens. The user's text is
  given as context (label = -100 means "ignore in the loss"). Otherwise the
  model would also learn to imitate users.
* Length bucketing: conversations of similar length are batched together so
  we waste less GPU time on padding.
* Mixed precision (bf16) + gradient checkpointing: fit bigger batches in
  40 GB of GPU memory and train faster.
* Time budget: training stops (and saves) after a fixed number of hours so
  the Colab compute-unit spend is capped.
* Resumable: checkpoints are saved regularly; a re-run continues from the
  last one.

All settings are environment variables, so the same script is used for a
20-step smoke test and for the full run (see configs/*.env).
"""
import json
import os
import random
import time

import torch
from datasets import load_dataset
from torch.utils.data import SequentialSampler
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint

# ---------------------------------------------------------------------------
# Configuration (override any of these with environment variables)
# ---------------------------------------------------------------------------
MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-0.5B")            # base model (NOT the instruct one)
DATASET = os.environ.get("DATASET", "HuggingFaceTB/smoltalk")    # ~1M synthetic + curated chats
DATASET_CONFIG = os.environ.get("DATASET_CONFIG", "all")
N_SAMPLES = int(os.environ.get("N_SAMPLES", "150000"))           # how many conversations to train on
MAX_LEN = int(os.environ.get("MAX_LEN", "2048"))                 # truncate longer conversations (tokens)
MAX_STEPS = int(os.environ.get("MAX_STEPS", "-1"))               # -1 = one full pass over the data
TIME_BUDGET_HOURS = float(os.environ.get("TIME_BUDGET_HOURS", "3.0"))
LR = float(os.environ.get("LR", "3e-5"))                         # small LR: we only nudge a trained model
BATCH = int(os.environ.get("BATCH", "4"))                        # conversations per GPU step
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", "8"))              # effective batch = BATCH * GRAD_ACCUM = 32
OUT = os.environ.get("OUT", "/content/out")
SAVE_STEPS = int(os.environ.get("SAVE_STEPS", "300"))
SEED = 42

# ChatML template - the same format Qwen's official instruct models use.
# It is saved with the tokenizer so `tokenizer.apply_chat_template` works later.
CHAT_TEMPLATE = (
    "{% for m in messages %}"
    "{{ '<|im_start|>' + m['role'] + '\n' + m['content'] + '<|im_end|>\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)

tok = AutoTokenizer.from_pretrained(MODEL)
tok.chat_template = CHAT_TEMPLATE
if tok.pad_token is None:
    tok.pad_token = tok.eos_token


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
def encode(example):
    """Turn one conversation into token ids + labels.

    We tokenize message by message so we know exactly which tokens belong to
    the assistant. Only those get a real label; everything else gets -100,
    which PyTorch's cross-entropy loss ignores.
    """
    input_ids, labels = [], []
    for m in example["messages"]:
        head = tok(f"<|im_start|>{m['role']}\n", add_special_tokens=False)["input_ids"]
        body = tok(f"{m['content']}<|im_end|>\n", add_special_tokens=False)["input_ids"]
        input_ids += head + body
        if m["role"] == "assistant":
            # Learn the reply AND the <|im_end|> token, so the model learns when to stop.
            labels += [-100] * len(head) + body
        else:
            labels += [-100] * (len(head) + len(body))
    input_ids, labels = input_ids[:MAX_LEN], labels[:MAX_LEN]
    return {"input_ids": input_ids, "labels": labels, "length": len(input_ids)}


def has_target(example):
    """Drop conversations whose assistant part was cut off entirely by truncation."""
    return any(l != -100 for l in example["labels"])


def bucket_by_length(ds, batch_size, bucket_batches=64):
    """Reorder the dataset so each batch holds conversations of similar length.

    1. shuffle everything,
    2. cut into big chunks (64 batches each) and sort each chunk by length,
    3. split chunks into batches and shuffle the *batches*.
    Result: little padding inside a batch, but still random training order.
    """
    rng = random.Random(SEED)
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    lengths = ds["length"]
    chunk = batch_size * bucket_batches
    batches = []
    for i in range(0, len(idx), chunk):
        part = sorted(idx[i:i + chunk], key=lambda j: lengths[j])
        batches += [part[k:k + batch_size] for k in range(0, len(part), batch_size)]
    rng.shuffle(batches)
    return ds.select([j for b in batches for j in b])


def collate(batch):
    """Pad a list of examples to the longest one in the batch."""
    width = max(len(b["input_ids"]) for b in batch)
    ids, labels, mask = [], [], []
    for b in batch:
        pad = width - len(b["input_ids"])
        ids.append(b["input_ids"] + [tok.pad_token_id] * pad)
        labels.append(b["labels"] + [-100] * pad)                # padding never counts in the loss
        mask.append([1] * len(b["input_ids"]) + [0] * pad)       # 0 = "don't attend to this position"
    return {
        "input_ids": torch.tensor(ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(mask),
    }


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------
class OrderedTrainer(Trainer):
    """Hugging Face Trainer that keeps our length-bucketed order.

    The default Trainer shuffles the dataset again, which would undo the
    bucketing. We already shuffled at batch level, so read it in order.
    """

    def _get_train_sampler(self, *args, **kwargs):
        return SequentialSampler(self.train_dataset)


class TimeBudget(TrainerCallback):
    """Save a checkpoint and stop once the time budget is used up.

    Colab bills GPU time in compute units, so a hard time limit is a hard
    spending limit.
    """

    def __init__(self, hours):
        self.deadline = time.time() + hours * 3600

    def on_step_end(self, args, state, control, **kwargs):
        if time.time() > self.deadline:
            print(f"[time budget] {TIME_BUDGET_HOURS}h reached at step {state.global_step}; stopping.")
            control.should_save = True
            control.should_training_stop = True
        return control


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    start = time.time()
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"model={MODEL} dataset={DATASET}/{DATASET_CONFIG} n={N_SAMPLES} max_len={MAX_LEN}")

    # 1) Load a random subset of the chat dataset and tokenize it (in parallel on all CPU cores).
    ds = load_dataset(DATASET, DATASET_CONFIG, split="train")
    ds = ds.shuffle(seed=SEED).select(range(min(N_SAMPLES, len(ds))))
    ds = ds.map(encode, remove_columns=ds.column_names, num_proc=os.cpu_count())
    ds = ds.filter(has_target, num_proc=os.cpu_count())
    n_tokens = sum(ds["length"])
    print(f"examples={len(ds)} tokens={n_tokens:,}")
    ds = bucket_by_length(ds, BATCH)

    # 2) Load the base model. Weights stay in fp32; the forward/backward pass runs in bf16 (mixed precision).
    model = AutoModelForCausalLM.from_pretrained(MODEL, attn_implementation="sdpa")
    model.config.use_cache = False  # the generation cache is useless during training and conflicts with checkpointing
    print(f"parameters={model.num_parameters():,}")

    # 3) Training configuration.
    args = TrainingArguments(
        output_dir=OUT,
        num_train_epochs=1,
        max_steps=MAX_STEPS,
        per_device_train_batch_size=BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        lr_scheduler_type="cosine",      # LR warms up, then smoothly decays to 0
        warmup_steps=0.03,               # first 3% of steps are warm-up (float = ratio in transformers v5)
        weight_decay=0.0,
        max_grad_norm=1.0,               # clip exploding gradients
        bf16=True,                       # mixed precision on Ampere GPUs (A100)
        tf32=True,                       # faster fp32 matmuls on A100
        gradient_checkpointing=True,     # recompute activations in backward -> much less memory
        logging_steps=10,
        save_steps=SAVE_STEPS,           # frequent checkpoints: Colab VMs can be reclaimed at any time
        save_total_limit=2,              # keep only the 2 newest checkpoints on disk
        dataloader_num_workers=2,
        remove_unused_columns=False,     # we pass our own columns; don't let Trainer drop them
        report_to="none",
        seed=SEED,
    )

    trainer = OrderedTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        data_collator=collate,
        callbacks=[TimeBudget(TIME_BUDGET_HOURS)],
    )

    # 4) Train (resuming automatically if a checkpoint exists, e.g. after a disconnect).
    last = get_last_checkpoint(OUT) if os.path.isdir(OUT) else None
    if last:
        print(f"resuming from {last}")
    result = trainer.train(resume_from_checkpoint=last)

    # 5) Save the final model in bf16 (half the size of fp32; ~1 GB) plus the tokenizer/chat template.
    final = os.path.join(OUT, "final")
    model.config.use_cache = True
    model.to(torch.bfloat16).save_pretrained(final)
    tok.save_pretrained(final)

    # 6) Save metrics + the loss curve for the README.
    summary = {
        "model": MODEL,
        "dataset": f"{DATASET}/{DATASET_CONFIG}",
        "examples": len(ds),
        "train_tokens": n_tokens,
        "steps": trainer.state.global_step,
        "final_train_loss": result.training_loss,
        "wall_clock_hours": round((time.time() - start) / 3600, 2),
        "gpu": torch.cuda.get_device_name(0),
    }
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(OUT, "log_history.json"), "w") as f:
        json.dump(trainer.state.log_history, f)
    print(json.dumps(summary, indent=2))
    print(f"DONE saved to {final}")


if __name__ == "__main__":
    main()
