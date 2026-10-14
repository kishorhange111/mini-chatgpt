"""
Generate side-by-side answers from the SFT and DPO models for the README.

Loads each trained model, asks it the same fixed prompts with greedy decoding
(deterministic, so the two models are compared fairly), and writes
/content/dpo/sft_vs_dpo_samples.json (copied to results/ for the repo).

The model is put in eval mode with the KV cache ON. Generating in training
mode with gradient checkpointing silently drops the cache and produces
garbage (see "Engineering notes" in the README).

Usage:  python samples.py        (needs a GPU and the outputs of train.py / dpo.py)
"""

import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = [
    "Explain the difference between a list and a tuple in Python.",
    "Give me three tips for a job interview.",
    "What should I do if I feel stressed at work?",
    "Write a haiku about machine learning.",
    "My code throws 'KeyError: user_id'. What does that mean and how do I fix it?",
]
MODELS = [("sft", "/content/out/final"), ("dpo", "/content/dpo/final")]


def main():
    """Answer every prompt with every model, save the JSON and print a short preview."""
    out = {}
    for name, path in MODELS:
        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to("cuda").eval()
        model.config.use_cache = True
        # Stop at the ChatML end-of-message token: that is where an assistant reply ends.
        end = tok.convert_tokens_to_ids("<|im_end|>")
        out[name] = {}
        for p in PROMPTS:
            # Format as a one-turn chat ending in "<|im_start|>assistant\n", then tokenize the text.
            text = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False)
            ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to("cuda")
            with torch.no_grad():
                # repetition_penalty: small models tend to loop on greedy decoding without it.
                g = model.generate(
                    ids, max_new_tokens=300, do_sample=False, eos_token_id=end, repetition_penalty=1.1, use_cache=True
                )
            # Decode only the newly generated tokens, not the prompt.
            out[name][p] = tok.decode(g[0][ids.shape[1] :], skip_special_tokens=True)
    with open("/content/dpo/sft_vs_dpo_samples.json", "w") as f:
        json.dump(out, f, indent=2)
    for p in PROMPTS:
        print("###", p, "\nSFT:", out["sft"][p][:250], "\nDPO:", out["dpo"][p][:250], "\n")


if __name__ == "__main__":
    main()
