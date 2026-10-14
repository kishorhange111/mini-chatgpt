"""SFT data preparation: loss masking, truncation, length bucketing and padding."""

import torch
import torch.nn.functional as F

from mini_chatgpt.chat_format import IGNORE_INDEX, IM_END, render_chatml
from mini_chatgpt.sft_data import bucket_by_length, collate_sft, encode_conversation, has_target

from .conftest import PAD_ID


def trained_text(tok, enc):
    """Decode only the tokens that carry a real label (the ones the loss sees)."""
    return tok.decode([t for t, label in zip(enc["input_ids"], enc["labels"], strict=True) if label != IGNORE_INDEX])


# --- encode_conversation ----------------------------------------------------
def test_tokens_reproduce_the_chat_template(tok, conversation):
    # Encoding message by message must give exactly the text the template renders,
    # otherwise the model would be trained on a format it never sees at inference.
    enc = encode_conversation(conversation, tok, max_len=10_000)
    assert tok.decode(enc["input_ids"]) == render_chatml(conversation)


def test_only_assistant_replies_and_their_end_token_are_trained(tok, conversation):
    enc = encode_conversation(conversation, tok, max_len=10_000)
    assert trained_text(tok, enc) == f"Hello!{IM_END}\n4{IM_END}\n"


def test_user_system_and_role_headers_are_masked(tok, conversation):
    enc = encode_conversation(conversation, tok, max_len=10_000)
    learned = trained_text(tok, enc)
    for forbidden in ["Be brief.", "Hi", "2+2?", "assistant", "user", "<|im_start|>"]:
        assert forbidden not in learned


def test_labels_align_with_input_ids(tok, conversation):
    enc = encode_conversation(conversation, tok, max_len=10_000)
    assert len(enc["labels"]) == len(enc["input_ids"]) == enc["length"]
    for t, label in zip(enc["input_ids"], enc["labels"], strict=True):
        assert label in (IGNORE_INDEX, t)


def test_truncation_keeps_ids_and_labels_aligned(tok, conversation):
    enc = encode_conversation(conversation, tok, max_len=7)
    assert enc["length"] == 7
    assert len(enc["input_ids"]) == len(enc["labels"]) == 7
    assert enc["input_ids"] == encode_conversation(conversation, tok, max_len=10_000)["input_ids"][:7]


def test_has_target_drops_examples_whose_reply_was_truncated_away(tok, conversation):
    assert has_target(encode_conversation(conversation, tok, max_len=10_000))
    # 7 tokens only covers the start of the system message - nothing left to learn.
    assert not has_target(encode_conversation(conversation, tok, max_len=7))


def test_user_only_conversation_has_no_target(tok):
    assert not has_target(encode_conversation([{"role": "user", "content": "hello"}], tok, max_len=100))


# --- bucket_by_length -------------------------------------------------------
def test_bucketing_is_a_permutation():
    lengths = [(i * 37) % 101 for i in range(1000)]
    order = bucket_by_length(lengths, batch_size=8)
    assert sorted(order) == list(range(1000))


def test_bucketing_is_deterministic_per_seed():
    lengths = list(range(500))
    assert bucket_by_length(lengths, 4, seed=1) == bucket_by_length(lengths, 4, seed=1)
    assert bucket_by_length(lengths, 4, seed=1) != bucket_by_length(lengths, 4, seed=2)


def test_bucketing_reduces_padding():
    import random

    rng = random.Random(0)
    lengths = [rng.randint(1, 2000) for _ in range(4096)]
    batch = 8

    def padding(order):
        """Total padding tokens if batches are taken consecutively from `order`."""
        total = 0
        for i in range(0, len(order), batch):
            group = [lengths[j] for j in order[i : i + batch]]
            total += max(group) * len(group) - sum(group)
        return total

    random_order = list(range(len(lengths)))
    rng.shuffle(random_order)
    # Sorting within chunks of 64 batches should remove the vast majority of padding.
    assert padding(bucket_by_length(lengths, batch)) < 0.1 * padding(random_order)


def test_bucketing_still_shuffles_batch_order():
    lengths = list(range(4096))
    order = bucket_by_length(lengths, batch_size=8)
    batch_max = [max(lengths[j] for j in order[i : i + 8]) for i in range(0, len(order), 8)]
    assert batch_max != sorted(batch_max)  # not one long short-to-long curriculum


def test_bucketing_handles_partial_last_batch():
    order = bucket_by_length([5, 3, 9, 1, 7], batch_size=2, bucket_batches=1)
    assert sorted(order) == [0, 1, 2, 3, 4]


# --- collate_sft ------------------------------------------------------------
def test_collate_pads_to_longest_and_masks_padding():
    batch = [
        {"input_ids": [5, 6, 7], "labels": [-100, 6, 7]},
        {"input_ids": [8], "labels": [8]},
    ]
    out = collate_sft(batch, pad_token_id=PAD_ID)
    assert out["input_ids"].tolist() == [[5, 6, 7], [8, PAD_ID, PAD_ID]]
    assert out["labels"].tolist() == [[-100, 6, 7], [8, -100, -100]]
    assert out["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]
    assert all(t.dtype == torch.long for t in out.values())


def test_masked_positions_do_not_affect_the_loss(tok, conversation):
    # End to end: build a batch, compute cross-entropy the way a causal LM does,
    # and check that changing the logits at user/padding positions changes nothing.
    enc = encode_conversation(conversation, tok, max_len=10_000)
    short = encode_conversation(
        [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}], tok, max_len=10_000
    )
    batch = collate_sft([enc, short], pad_token_id=PAD_ID)
    labels = batch["labels"]
    vocab = 300

    def loss(logits):
        # Causal LM convention: logits at position t predict label t+1.
        return F.cross_entropy(logits[:, :-1].reshape(-1, vocab), labels[:, 1:].reshape(-1), ignore_index=IGNORE_INDEX)

    torch.manual_seed(0)
    logits = torch.randn(*labels.shape, vocab)
    base = loss(logits)

    ignored = labels[:, 1:] == IGNORE_INDEX  # predictions the loss should skip
    noisy = logits.clone()
    noisy[:, :-1][ignored] += 100 * torch.randn(int(ignored.sum()), vocab)
    assert torch.allclose(loss(noisy), base)

    counted = ~ignored
    noisy = logits.clone()
    noisy[:, :-1][counted] += 100 * torch.randn(int(counted.sum()), vocab)
    assert not torch.allclose(loss(noisy), base)
