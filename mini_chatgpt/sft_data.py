"""
Data preparation for supervised fine-tuning (SFT).

SFT is ordinary next-token prediction on example conversations, with two
twists that this module implements:

1. Loss masking (encode_conversation). We only want the model to learn to
   write *assistant* replies, not to imitate users. Every token gets a label;
   user/system tokens get IGNORE_INDEX (-100) so the loss skips them.

2. Length bucketing (bucket_by_length) + padding (collate_sft). A batch is
   padded to its longest conversation, so mixing a 50-token chat with a
   2,000-token chat wastes ~97% of the compute on padding. We batch
   conversations of similar length together, but still in random order.

All functions are pure: they take the tokenizer or pad id as an argument, so
they can be tested with a tiny fake tokenizer and no downloads.
"""
import random

import torch

from .chat_format import IGNORE_INDEX, message_body, message_head


def encode_conversation(messages, tokenizer, max_len):
    """Turn one conversation into token ids plus loss-masked labels.

    We tokenize message by message (rather than the whole rendered string) so
    we know exactly which tokens belong to the assistant. Only those get a
    real label; everything else gets IGNORE_INDEX, which the loss ignores.

    Args:
        messages: list of {"role", "content"} dicts.
        tokenizer: callable like a Hugging Face tokenizer:
            tokenizer(text, add_special_tokens=False)["input_ids"] -> list[int].
        max_len: truncate the conversation to this many tokens.

    Returns:
        {"input_ids": list[int], "labels": list[int], "length": int}, where
        labels[i] is input_ids[i] for assistant tokens and -100 elsewhere.
        (The model shifts labels by one internally, so they line up with
        input_ids here.)
    """
    input_ids, labels = [], []
    for m in messages:
        head = tokenizer(message_head(m["role"]), add_special_tokens=False)["input_ids"]
        body = tokenizer(message_body(m["content"]), add_special_tokens=False)["input_ids"]
        input_ids += head + body
        if m["role"] == "assistant":
            # Learn the reply AND the <|im_end|> token, so the model learns when to stop.
            # The "<|im_start|>assistant\n" header is masked: at inference it is
            # always given in the prompt, so there is nothing to learn there.
            labels += [IGNORE_INDEX] * len(head) + body
        else:
            labels += [IGNORE_INDEX] * (len(head) + len(body))
    # Truncate ids and labels together so they stay aligned position by position.
    input_ids, labels = input_ids[:max_len], labels[:max_len]
    return {"input_ids": input_ids, "labels": labels, "length": len(input_ids)}


def has_target(example):
    """Return True if at least one token in the example is trained on.

    Truncation can cut off the whole assistant reply of a long conversation.
    Such an example would contribute zero loss (and a NaN mean in some
    setups), so the training script filters it out.

    Args:
        example: a dict with a "labels" list, as built by encode_conversation.

    Returns:
        bool.
    """
    return any(label != IGNORE_INDEX for label in example["labels"])


def bucket_by_length(lengths, batch_size, bucket_batches=64, seed=42):
    """Compute a dataset order in which each batch holds similar-length examples.

    1. shuffle all indices,
    2. cut them into big chunks (`bucket_batches` batches each) and sort each
       chunk by length,
    3. split chunks into batches and shuffle the *batches*.

    Result: little padding inside a batch, but training order is still random
    (sorting the whole dataset would put all short chats first, which biases
    the start of training).

    Args:
        lengths: length (in tokens) of every example, indexed like the dataset.
        batch_size: examples per batch.
        bucket_batches: how many batches are sorted together in one chunk.
            Bigger = tighter length matching but less randomness.
        seed: RNG seed, so the order is reproducible (and resumable).

    Returns:
        A list of dataset indices, a permutation of range(len(lengths)).
        Consecutive groups of `batch_size` indices form the batches.
    """
    rng = random.Random(seed)
    idx = list(range(len(lengths)))
    rng.shuffle(idx)
    chunk = batch_size * bucket_batches
    batches = []
    for i in range(0, len(idx), chunk):
        part = sorted(idx[i:i + chunk], key=lambda j: lengths[j])
        batches += [part[k:k + batch_size] for k in range(0, len(part), batch_size)]
    rng.shuffle(batches)
    return [j for b in batches for j in b]


def collate_sft(batch, pad_token_id):
    """Pad a list of encoded examples to the longest one and stack into tensors.

    Args:
        batch: list of dicts with "input_ids" and "labels" lists.
        pad_token_id: token id used to fill the padding positions.

    Returns:
        {"input_ids", "labels", "attention_mask"} as LongTensors of shape
        (batch_size, longest_length).
    """
    width = max(len(b["input_ids"]) for b in batch)
    ids, labels, mask = [], [], []
    for b in batch:
        pad = width - len(b["input_ids"])
        ids.append(b["input_ids"] + [pad_token_id] * pad)
        labels.append(b["labels"] + [IGNORE_INDEX] * pad)          # padding never counts in the loss
        mask.append([1] * len(b["input_ids"]) + [0] * pad)          # 0 = "don't attend to this position"
    return {
        "input_ids": torch.tensor(ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(mask),
    }
