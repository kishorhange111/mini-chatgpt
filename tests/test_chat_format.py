"""The Jinja chat template and the pure-Python renderer must produce identical ChatML."""
import pytest

from mini_chatgpt.chat_format import IM_END, IM_START, render_chatml


@pytest.mark.parametrize("add_generation_prompt", [False, True])
def test_template_matches_python_renderer(tok, conversation, add_generation_prompt):
    rendered = tok.apply_chat_template(conversation, add_generation_prompt=add_generation_prompt)
    assert rendered == render_chatml(conversation, add_generation_prompt=add_generation_prompt)


def test_chatml_layout():
    text = render_chatml([{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Yo"}])
    assert text == f"{IM_START}user\nHi{IM_END}\n{IM_START}assistant\nYo{IM_END}\n"


def test_generation_prompt_opens_an_assistant_turn():
    text = render_chatml([{"role": "user", "content": "Hi"}], add_generation_prompt=True)
    assert text.endswith(f"{IM_END}\n{IM_START}assistant\n")


def test_empty_conversation():
    assert render_chatml([]) == ""
    assert render_chatml([], add_generation_prompt=True) == f"{IM_START}assistant\n"
