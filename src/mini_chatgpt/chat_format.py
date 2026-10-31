"""
The ChatML chat format used throughout Mini-ChatGPT.

A base language model only sees one long string of tokens. To turn it into a
chat assistant we need a convention that marks *who* is speaking and *where*
each message ends. ChatML (the format Qwen's own instruct models use) wraps
every message like this:

    <|im_start|>user
    Hello<|im_end|>
    <|im_start|>assistant
    Hi!<|im_end|>

At inference time the prompt ends with an open "<|im_start|>assistant\\n" (the
"generation prompt"), and the model writes until it produces <|im_end|>.

Two representations of the same format live here:

* CHAT_TEMPLATE - a Jinja template. It is attached to the tokenizer and saved
  with the model, so `tokenizer.apply_chat_template(...)` (used by scripts/train_dpo.py,
  scripts/generate_samples.py and lm-evaluation-harness) produces exactly this format.
* render_chatml - the same thing in plain Python. Tests use it to check that
  the template and the SFT encoder (which builds the format token by token)
  never drift apart.
"""

# Special marker strings. In the Qwen tokenizer each of these is one token.
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"

# PyTorch's cross-entropy ignores positions whose label is -100. We use it to
# keep user/system tokens and padding out of the loss.
IGNORE_INDEX = -100

# ChatML as a Jinja template - the format the tokenizer's `apply_chat_template`
# renders. Note the "\n" below is a real newline inside a Jinja string literal.
CHAT_TEMPLATE = (
    "{% for m in messages %}"
    "{{ '<|im_start|>' + m['role'] + '\n' + m['content'] + '<|im_end|>\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)


def message_head(role):
    """Return the opening of a ChatML message, e.g. "<|im_start|>user\\n".

    Args:
        role: "system", "user" or "assistant".

    Returns:
        The header string that precedes the message content.
    """
    return f"{IM_START}{role}\n"


def message_body(content):
    """Return a message's content followed by the end-of-message marker.

    The <|im_end|> marker is part of the *body* on purpose: for assistant
    messages it is trained on, which is how the model learns when to stop.

    Args:
        content: the message text.

    Returns:
        "content<|im_end|>\\n".
    """
    return f"{content}{IM_END}\n"


def render_chatml(messages, add_generation_prompt=False):
    """Render a conversation as ChatML text (pure-Python twin of CHAT_TEMPLATE).

    Args:
        messages: list of {"role": str, "content": str} dicts, in order.
        add_generation_prompt: if True, end with an open assistant header so a
            model continues with the assistant's reply.

    Returns:
        The full conversation as one string.
    """
    text = "".join(message_head(m["role"]) + message_body(m["content"]) for m in messages)
    if add_generation_prompt:
        text += message_head("assistant")
    return text
