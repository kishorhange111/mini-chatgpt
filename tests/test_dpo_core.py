"""DPO: pair encoding, batching, sequence log-probabilities and the loss itself."""

import copy
import math

import pytest
import torch
import torch.nn.functional as F

from mini_chatgpt.chat_format import render_chatml
from mini_chatgpt.dpo_core import answer_logprob, collate_pairs, dpo_loss, dpo_loss_from_logprobs, encode_pair

from .conftest import END_ID, PAD_ID, BigramLM


def make_example(prompt="Tell me a joke", chosen="Why did...", rejected="No."):
    """An UltraFeedback-style row: two conversations sharing one prompt."""
    user = {"role": "user", "content": prompt}
    return {
        "chosen": [user, {"role": "assistant", "content": chosen}],
        "rejected": [user, {"role": "assistant", "content": rejected}],
    }


# --- encode_pair ------------------------------------------------------------
def test_encode_pair_formats_prompt_with_generation_prompt(tok):
    ex = make_example()
    out = encode_pair(ex, tok, END_ID, max_prompt=512, max_len=1024)
    assert tok.decode(out["prompt"]) == render_chatml(ex["chosen"][:-1], add_generation_prompt=True)


def test_encode_pair_answers_end_with_end_token(tok):
    out = encode_pair(make_example(chosen="yes", rejected="no"), tok, END_ID, 512, 1024)
    assert out["chosen"][-1] == END_ID and out["rejected"][-1] == END_ID
    assert tok.decode(out["chosen"][:-1]) == "yes"
    assert tok.decode(out["rejected"][:-1]) == "no"


def test_encode_pair_keeps_the_end_of_long_prompts(tok):
    out = encode_pair(make_example(prompt="x" * 100 + "QUESTION"), tok, END_ID, max_prompt=50, max_len=1024)
    assert len(out["prompt"]) == 50
    # The tail (latest question + assistant header) survives; the head is dropped.
    text = tok.decode(out["prompt"])
    assert text.endswith("QUESTION<|im_end|>\n<|im_start|>assistant\n")
    assert not text.startswith("<|im_start|>user")


def test_encode_pair_caps_total_length(tok):
    out = encode_pair(make_example(chosen="a" * 50, rejected="b" * 50), tok, END_ID, max_prompt=20, max_len=30)
    assert len(out["prompt"]) + len(out["chosen"]) == 30
    assert len(out["prompt"]) + len(out["rejected"]) == 30


def test_encode_pair_can_truncate_answer_to_empty(tok):
    out = encode_pair(make_example(), tok, END_ID, max_prompt=20, max_len=20)
    assert out["chosen"] == [] and out["rejected"] == []  # dpo.py drops such pairs


# --- collate_pairs ----------------------------------------------------------
def test_collate_pairs_layout():
    pairs = [
        {"prompt": [11, 12], "chosen": [21], "rejected": [31, 32, 33]},
        {"prompt": [13], "chosen": [22, 23], "rejected": [34]},
    ]
    ids, att, lm = collate_pairs(pairs, pad_token_id=PAD_ID)
    assert ids.shape == att.shape == lm.shape == (4, 5)
    # rows 0..N-1 are chosen, rows N..2N-1 rejected
    assert ids.tolist() == [
        [11, 12, 21, 0, 0],
        [13, 22, 23, 0, 0],
        [11, 12, 31, 32, 33],
        [13, 34, 0, 0, 0],
    ]
    assert att.tolist() == [[1, 1, 1, 0, 0], [1, 1, 1, 0, 0], [1, 1, 1, 1, 1], [1, 1, 0, 0, 0]]
    # loss mask: answers only, never the prompt or padding
    assert lm.tolist() == [[0, 0, 1, 0, 0], [0, 1, 1, 0, 0], [0, 0, 1, 1, 1], [0, 1, 0, 0, 0]]


# --- answer_logprob ---------------------------------------------------------
def test_answer_logprob_matches_manual_computation():
    model = BigramLM()
    pairs = [{"prompt": [11, 12, 13], "chosen": [14, 15], "rejected": [16]}]
    ids, att, lm = collate_pairs(pairs, PAD_ID)
    got = answer_logprob(model, ids, att, lm)

    def manual(prompt, answer):
        seq = prompt + answer
        logp = F.log_softmax(model.table.weight, -1)  # BigramLM: logits depend only on the current token
        # sum log p(answer token | previous token) over answer positions only
        return sum(logp[seq[t - 1], seq[t]].item() for t in range(len(prompt), len(seq)))

    expected = [manual([11, 12, 13], [14, 15]), manual([11, 12, 13], [16])]
    assert got.tolist() == pytest.approx(expected, rel=1e-5)


def test_answer_logprob_ignores_padding():
    model = BigramLM()
    short = [{"prompt": [11], "chosen": [12], "rejected": [13]}]
    padded = short + [{"prompt": [11, 11, 11, 11], "chosen": [12, 12, 12], "rejected": [13, 13, 13]}]
    a = answer_logprob(model, *collate_pairs(short, PAD_ID))
    b = answer_logprob(model, *collate_pairs(padded, PAD_ID))
    # Row order is [chosen..., rejected...], so the short pair is rows 0 and N.
    assert torch.allclose(a, b[[0, 2]])


# --- the loss ---------------------------------------------------------------
def test_loss_is_log2_when_policy_equals_reference():
    pi = torch.tensor([-3.0, -5.0, -4.0, -2.0])
    loss, rc, rr = dpo_loss_from_logprobs(pi, pi.clone(), beta=0.1)
    assert loss.item() == pytest.approx(math.log(2))
    assert torch.equal(rc, torch.zeros(2)) and torch.equal(rr, torch.zeros(2))


def test_loss_matches_formula():
    pi = torch.tensor([-1.0, -6.0])  # chosen, rejected
    pi_ref = torch.tensor([-2.0, -5.0])
    beta = 0.5
    loss, rc, rr = dpo_loss_from_logprobs(pi, pi_ref, beta)
    r_w, r_l = beta * (-1 + 2), beta * (-6 + 5)
    assert rc.item() == pytest.approx(r_w) and rr.item() == pytest.approx(r_l)
    assert loss.item() == pytest.approx(-math.log(1 / (1 + math.exp(-(r_w - r_l)))))


def test_loss_falls_as_chosen_becomes_more_likely():
    pi = torch.tensor([-3.0, -3.0], requires_grad=True)
    loss, _, _ = dpo_loss_from_logprobs(pi, torch.tensor([-3.0, -3.0]), beta=0.1)
    loss.backward()
    assert pi.grad[0] < 0  # gradient descent raises log pi(chosen)
    assert pi.grad[1] > 0  # ... and lowers log pi(rejected)


def test_rewards_are_detached():
    pi = torch.tensor([-1.0, -2.0], requires_grad=True)
    _, rc, rr = dpo_loss_from_logprobs(pi, torch.zeros(2), beta=0.1)
    assert not rc.requires_grad and not rr.requires_grad


def test_dpo_loss_does_not_train_the_reference():
    policy = BigramLM(seed=0)
    ref = copy.deepcopy(policy).requires_grad_(False)
    batch = collate_pairs([{"prompt": [11, 12], "chosen": [13, 14], "rejected": [15]}], PAD_ID)
    loss, _, _ = dpo_loss(policy, ref, batch, beta=0.1)
    assert loss.item() == pytest.approx(math.log(2), rel=1e-5)  # identical models -> log 2
    loss.backward()
    assert policy.table.weight.grad is not None
    assert ref.table.weight.grad is None


def test_a_few_dpo_steps_learn_the_preference():
    # Integration check on a toy problem: after some steps, the policy prefers
    # the chosen answers more than the frozen reference does.
    policy = BigramLM(seed=0)
    ref = copy.deepcopy(policy).requires_grad_(False)
    pairs = [{"prompt": [11 + i], "chosen": [40, 41], "rejected": [50, 51]} for i in range(4)]
    batch = collate_pairs(pairs, PAD_ID)
    opt = torch.optim.SGD(policy.parameters(), lr=1.0)
    for _ in range(20):
        loss, rc, rr = dpo_loss(policy, ref, batch, beta=0.1)
        opt.zero_grad()
        loss.backward()
        opt.step()
    loss, rc, rr = dpo_loss(policy, ref, batch, beta=0.1)
    assert loss.item() < math.log(2)
    assert (rc > rr).all()
