"""Ask the fine-tuned model a few questions (runs on the Colab VM)."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PATH = "/content/out/final"
tok = AutoTokenizer.from_pretrained(PATH)
model = AutoModelForCausalLM.from_pretrained(PATH, dtype=torch.bfloat16).to("cuda").eval()

prompts = [
    "Hi! Who are you?",
    "Explain what a neural network is in simple terms.",
    "Write a Python function that checks if a number is prime.",
    "Give me 3 tips for a good resume.",
]
for p in prompts:
    msgs = [{"role": "user", "content": p}]
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to("cuda")
    out = model.generate(
        ids,
        max_new_tokens=300,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        eos_token_id=tok.convert_tokens_to_ids("<|im_end|>"),
    )
    print(f"### USER: {p}\n### ASSISTANT: {tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)}\n")
