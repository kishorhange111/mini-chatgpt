"""
mini_chatgpt: the pure, unit-tested building blocks of the Mini-ChatGPT pipeline.

The entry-point scripts in scripts/ (train_sft.py, train_dpo.py, evaluate.sh,
generate_samples.py) handle the heavy, GPU-bound work: downloading models and
datasets and running training.
Everything that is *logic* rather than *plumbing* lives in this package so it
can be imported and tested on a laptop CPU without downloading anything:

  chat_format   the ChatML chat template and a pure-Python renderer for it
  sft_data      SFT data preparation: loss masking, length bucketing, padding
  dpo_core      DPO data preparation, sequence log-probabilities and the loss
  eval_summary  turns lm-evaluation-harness result files into one summary

Every function takes the tokenizer (or model) as an argument instead of using
a global, which is what makes it possible to test with tiny fakes.
"""

from .chat_format import CHAT_TEMPLATE, IGNORE_INDEX, IM_END, IM_START, render_chatml
from .dpo_core import answer_logprob, collate_pairs, dpo_loss, dpo_loss_from_logprobs, encode_pair
from .sft_data import bucket_by_length, collate_sft, encode_conversation, has_target

__all__ = [
    "CHAT_TEMPLATE",
    "IGNORE_INDEX",
    "IM_END",
    "IM_START",
    "answer_logprob",
    "bucket_by_length",
    "collate_pairs",
    "collate_sft",
    "dpo_loss",
    "dpo_loss_from_logprobs",
    "encode_conversation",
    "encode_pair",
    "has_target",
    "render_chatml",
]
