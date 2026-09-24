"""Generate side-by-side answers from the SFT and DPO models (eval mode, KV cache on) for the README."""
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = ["Explain the difference between a list and a tuple in Python.",
           "Give me three tips for a job interview.",
           "What should I do if I feel stressed at work?",
           "Write a haiku about machine learning.",
           "My code throws 'KeyError: user_id'. What does that mean and how do I fix it?"]
out = {}
for name, path in [("sft", "/content/out/final"), ("dpo", "/content/dpo/final")]:
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to("cuda").eval()
    model.config.use_cache = True
    end = tok.convert_tokens_to_ids("<|im_end|>")
    out[name] = {}
    for p in PROMPTS:
        text = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False)
        ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to("cuda")
        with torch.no_grad():
            g = model.generate(ids, max_new_tokens=300, do_sample=False, eos_token_id=end, repetition_penalty=1.1,
                               use_cache=True)
        out[name][p] = tok.decode(g[0][ids.shape[1]:], skip_special_tokens=True)
json.dump(out, open("/content/dpo/sft_vs_dpo_samples.json", "w"), indent=2)
for p in PROMPTS:
    print("###", p, "\nSFT:", out["sft"][p][:250], "\nDPO:", out["dpo"][p][:250], "\n")
