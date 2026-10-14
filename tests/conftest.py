"""Shared test fixtures: a character-level fake tokenizer and a tiny pure-torch language model.

Nothing here downloads anything, so the whole suite runs offline on CPU.
"""

from types import SimpleNamespace

import jinja2
import pytest
import torch
from torch import nn

from mini_chatgpt.chat_format import CHAT_TEMPLATE, IM_END

PAD_ID = 0
END_ID = 1  # stands in for the single <|im_end|> token of a real tokenizer
OFFSET = 10  # character c gets id ord(c) + OFFSET, keeping 0..9 free for specials
VOCAB = 256 + OFFSET


class CharTokenizer:
    """Minimal stand-in for a Hugging Face tokenizer: one token per character.

    Implements only what the code under test calls: __call__, decode,
    apply_chat_template (rendering the real CHAT_TEMPLATE with Jinja, as
    transformers does) and pad_token_id.
    """

    pad_token_id = PAD_ID

    def __init__(self):
        self.chat_template = CHAT_TEMPLATE

    def __call__(self, text, add_special_tokens=False):
        """Encode text as one id per character."""
        return {"input_ids": [ord(c) + OFFSET for c in text]}

    def decode(self, ids):
        """Inverse of __call__; specials map back to readable markers."""
        specials = {PAD_ID: "", END_ID: IM_END}
        return "".join(specials.get(i, chr(i - OFFSET) if i >= OFFSET else "") for i in ids)

    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=False):
        """Render messages with the chat template (text only, like tokenize=False)."""
        assert not tokenize, "the code under test always formats text first, then tokenizes"
        template = jinja2.Environment().from_string(self.chat_template)
        return template.render(messages=messages, add_generation_prompt=add_generation_prompt)


class BigramLM(nn.Module):
    """A tiny causal LM whose next-token logits depend only on the current token.

    Its forward() has the same calling convention and output shape as a
    Hugging Face causal LM (an object with a `.logits` attribute), which is all
    answer_logprob needs.
    """

    def __init__(self, vocab=VOCAB, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.table = nn.Embedding(vocab, vocab)

    def forward(self, input_ids, attention_mask=None):
        """Return logits of shape (batch, seq, vocab)."""
        return SimpleNamespace(logits=self.table(input_ids))


@pytest.fixture
def tok():
    """A fresh character-level tokenizer."""
    return CharTokenizer()


@pytest.fixture
def conversation():
    """A multi-turn chat with system, user and assistant messages."""
    return [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
        {"role": "user", "content": "2+2?"},
        {"role": "assistant", "content": "4"},
    ]
