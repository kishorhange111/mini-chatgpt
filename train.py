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

The pure data logic (loss masking, length bucketing, padding, the chat
template) lives in the mini_chatgpt/ package, where it is unit-tested on CPU.
This file is the GPU-side orchestration: load, tokenize, train, save.

Usage:
    python train.py                               # defaults below
    set -a; . configs/full.env; set +a; python train.py   # the full run
"""

import json
import os
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

from mini_chatgpt.chat_format import CHAT_TEMPLATE
from mini_chatgpt.sft_data import bucket_by_length, collate_sft, encode_conversation, has_target

# ---------------------------------------------------------------------------
# Configuration (override any of these with environment variables)
# ---------------------------------------------------------------------------
MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-0.5B")  # base model (NOT the instruct one)
DATASET = os.environ.get("DATASET", "HuggingFaceTB/smoltalk")  # ~1M synthetic + curated chats
DATASET_CONFIG = os.environ.get("DATASET_CONFIG", "all")
N_SAMPLES = int(os.environ.get("N_SAMPLES", "150000"))  # how many conversations to train on
MAX_LEN = int(os.environ.get("MAX_LEN", "2048"))  # truncate longer conversations (tokens)
MAX_STEPS = int(os.environ.get("MAX_STEPS", "-1"))  # -1 = one full pass over the data
TIME_BUDGET_HOURS = float(os.environ.get("TIME_BUDGET_HOURS", "3.0"))
LR = float(os.environ.get("LR", "3e-5"))  # small LR: we only nudge a trained model
BATCH = int(os.environ.get("BATCH", "4"))  # conversations per GPU step
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", "8"))  # effective batch = BATCH * GRAD_ACCUM = 32
OUT = os.environ.get("OUT", "/content/out")
SAVE_STEPS = int(os.environ.get("SAVE_STEPS", "300"))
SEED = 42

# The tokenizer is loaded at import time because `encode` and `collate` below
# use it, and datasets.map(num_proc=...) ships those functions to worker
# processes. The ChatML template (see mini_chatgpt/chat_format.py) is attached
# here so it is saved with the tokenizer and `apply_chat_template` works later.
tok = AutoTokenizer.from_pretrained(MODEL)
tok.chat_template = CHAT_TEMPLATE
if tok.pad_token is None:
    # Some base models ship without a pad token; reusing EOS is safe because
    # padded positions are masked out of both attention and the loss.
    tok.pad_token = tok.eos_token


# ---------------------------------------------------------------------------
# Data preparation (the logic lives in mini_chatgpt/sft_data.py, where it is
# unit-tested; these thin wrappers bind it to this run's tokenizer/settings)
# ---------------------------------------------------------------------------
def encode(example):
    """Turn one dataset row into token ids + loss-masked labels.

    Only assistant tokens get a real label; everything else gets -100,
    which PyTorch's cross-entropy loss ignores.

    Args:
        example: a SmolTalk row with a "messages" list of {"role", "content"}.

    Returns:
        {"input_ids", "labels", "length"} (see encode_conversation).
    """
    return encode_conversation(example["messages"], tok, MAX_LEN)


def collate(batch):
    """Pad a list of encoded examples to the longest one in the batch.

    Args:
        batch: list of dicts with "input_ids" and "labels".

    Returns:
        A dict of LongTensors the model's forward() accepts.
    """
    return collate_sft(batch, tok.pad_token_id)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------
class OrderedTrainer(Trainer):
    """Hugging Face Trainer that keeps our length-bucketed order.

    The default Trainer shuffles the dataset again, which would undo the
    bucketing. We already shuffled at batch level, so read it in order.
    """

    def _get_train_sampler(self, *args, **kwargs):
        """Read the dataset front to back (it is already batch-shuffled)."""
        return SequentialSampler(self.train_dataset)


class TimeBudget(TrainerCallback):
    """Save a checkpoint and stop once the time budget is used up.

    Colab bills GPU time in compute units, so a hard time limit is a hard
    spending limit.
    """

    def __init__(self, hours):
        """Args: hours - wall-clock budget, counted from when the callback is created."""
        self.deadline = time.time() + hours * 3600

    def on_step_end(self, args, state, control, **kwargs):
        """Called by Trainer after every optimizer step; flips the stop/save flags when over budget.

        Returns:
            The (possibly modified) TrainerControl object.
        """
        if time.time() > self.deadline:
            print(f"[time budget] {TIME_BUDGET_HOURS}h reached at step {state.global_step}; stopping.")
            control.should_save = True
            control.should_training_stop = True
        return control


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    """Run SFT end to end and write the final model, summary.json and log_history.json to OUT."""
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
    # Reorder so each batch holds conversations of similar length (less padding).
    ds = ds.select(bucket_by_length(ds["length"], BATCH, seed=SEED))

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
        lr_scheduler_type="cosine",  # LR warms up, then smoothly decays to 0
        warmup_steps=0.03,  # first 3% of steps are warm-up (float = ratio in transformers v5)
        weight_decay=0.0,
        max_grad_norm=1.0,  # clip exploding gradients
        bf16=True,  # mixed precision on Ampere GPUs (A100)
        tf32=True,  # faster fp32 matmuls on A100
        gradient_checkpointing=True,  # recompute activations in backward -> much less memory
        logging_steps=10,
        save_steps=SAVE_STEPS,  # frequent checkpoints: Colab VMs can be reclaimed at any time
        save_total_limit=2,  # keep only the 2 newest checkpoints on disk
        dataloader_num_workers=2,
        remove_unused_columns=False,  # we pass our own columns; don't let Trainer drop them
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
