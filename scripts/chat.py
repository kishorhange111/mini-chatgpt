"""
Quick manual check of the SFT model: ask it a few questions and print the answers.

Unlike scripts/generate_samples.py this uses *sampling* (temperature 0.7, top-p 0.9), which is
closer to how a chat app would run the model; answers change from run to run.

Usage:  python scripts/chat.py      (runs on the GPU VM, after scripts/train_sft.py)
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PATH = "/content/out/final"
PROMPTS = [
    "Hi! Who are you?",
    "Explain what a neural network is in simple terms.",
    "Write a Python function that checks if a number is prime.",
    "Give me 3 tips for a good resume.",
]


def main():
    """Load the SFT model and print one sampled answer per prompt."""
    tok = AutoTokenizer.from_pretrained(PATH)
    model = AutoModelForCausalLM.from_pretrained(PATH, dtype=torch.bfloat16).to("cuda").eval()
    for p in PROMPTS:
        msgs = [{"role": "user", "content": p}]
        # Render the ChatML prompt as text first, then tokenize: in transformers v5
        # apply_chat_template(tokenize=True) returns a dict rather than a tensor.
        text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to("cuda")
        out = model.generate(
            ids,
            max_new_tokens=300,
            do_sample=True,
            temperature=0.7,  # < 1 sharpens the distribution: fewer odd word choices
            top_p=0.9,  # nucleus sampling: ignore the long tail of unlikely tokens
            eos_token_id=tok.convert_tokens_to_ids("<|im_end|>"),  # stop at end of the assistant turn
        )
        # out[0] holds prompt + answer; slice off the prompt before decoding.
        print(f"### USER: {p}\n### ASSISTANT: {tok.decode(out[0][ids.shape[1] :], skip_special_tokens=True)}\n")


if __name__ == "__main__":
    main()
