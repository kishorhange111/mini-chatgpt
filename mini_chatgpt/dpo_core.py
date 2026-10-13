"""
Direct Preference Optimization (DPO), implemented by hand.

For a prompt x with a human-preferred answer y_w ("chosen") and a worse
answer y_l ("rejected"), DPO (Rafailov et al., 2023) minimises

    implicit reward  r(y) = beta * [ log pi(y|x) - log pi_ref(y|x) ]
    loss             = -log sigmoid( r(y_w) - r(y_l) )

where pi is the model being trained and pi_ref a frozen copy of the SFT model.
In words: raise the probability of the chosen answer and lower the rejected
one, *relative to where the SFT model started*. Measuring against pi_ref is
what stops the model drifting into nonsense; beta sets how far it may drift.

The pieces, in the order dpo.py uses them:

  encode_pair            prompt/chosen/rejected text -> token ids
  collate_pairs          pad a micro-batch: all chosen rows, then all rejected
  answer_logprob         log pi(y|x): summed log-probs of the answer tokens
  dpo_loss_from_logprobs the loss above, given the four log-probabilities
  dpo_loss               runs both models on a batch and returns the loss
"""
import torch
import torch.nn.functional as F


def encode_pair(ex, tokenizer, end_id, max_prompt, max_len):
    """Tokenize one preference example into prompt / chosen / rejected ids.

    The prompt is formatted with the tokenizer's chat template (the same
    ChatML used in SFT) and ends with the open assistant header; each answer
    is the final assistant message plus the <|im_end|> token.

    Args:
        ex: an UltraFeedback-style row: ex["chosen"] and ex["rejected"] are
            full conversations (lists of {"role", "content"}) that share the
            same prompt and differ only in the last (assistant) message.
        tokenizer: Hugging Face-like tokenizer with `apply_chat_template` and
            `__call__(text, add_special_tokens=False)`.
        end_id: token id of <|im_end|>, appended so the model is also scored
            on stopping at the right place.
        max_prompt: keep at most this many prompt tokens (the *last* ones).
        max_len: cap on prompt + answer length.

    Returns:
        {"prompt": list[int], "chosen": list[int], "rejected": list[int]}.
        An answer can come back empty if the prompt fills max_len; callers
        drop those pairs.
    """
    prompt_msgs = ex["chosen"][:-1]  # every message except the final answer
    prompt = tokenizer.apply_chat_template(prompt_msgs, add_generation_prompt=True, tokenize=False)
    # Keep the END of long prompts: the latest user turn is the part the answer responds to.
    p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"][-max_prompt:]
    out = {"prompt": p_ids}
    for key in ["chosen", "rejected"]:
        ans = tokenizer(ex[key][-1]["content"], add_special_tokens=False)["input_ids"] + [end_id]
        out[key] = ans[:max_len - len(p_ids)]
    return out


def collate_pairs(pairs, pad_token_id):
    """Stack chosen answers then rejected answers into ONE padded batch.

    One batch of 2N rows (rows 0..N-1 chosen, rows N..2N-1 rejected) means a
    single forward pass per model instead of two.

    Args:
        pairs: list of N dicts from encode_pair.
        pad_token_id: token id used for padding.

    Returns:
        (input_ids, attention_mask, loss_mask), each of shape (2N, width), on
        CPU. loss_mask is 1.0 only on answer tokens, so the prompt and padding
        do not count towards log pi(y|x).
    """
    seqs, masks = [], []
    for key in ["chosen", "rejected"]:
        for p in pairs:
            seqs.append(p["prompt"] + p[key])
            masks.append([0] * len(p["prompt"]) + [1] * len(p[key]))
    width = max(map(len, seqs))
    ids = torch.full((len(seqs), width), pad_token_id)
    att = torch.zeros((len(seqs), width), dtype=torch.long)
    lm = torch.zeros((len(seqs), width))
    for i, (s, m) in enumerate(zip(seqs, masks, strict=True)):
        ids[i, :len(s)] = torch.tensor(s)
        att[i, :len(s)] = 1
        lm[i, :len(m)] = torch.tensor(m, dtype=torch.float)
    return ids, att, lm


def answer_logprob(model, ids, att, loss_mask):
    """Sum of log-probabilities the model assigns to the answer tokens: log pi(y|x).

    Logits at position t predict token t+1, so we drop the last logit and the
    first token to line predictions up with their targets.

    Args:
        model: a causal LM; model(input_ids=..., attention_mask=...).logits
            has shape (batch, seq, vocab).
        ids: (batch, seq) token ids.
        att: (batch, seq) attention mask.
        loss_mask: (batch, seq) 1.0 on answer tokens, 0.0 elsewhere.

    Returns:
        (batch,) tensor of summed answer log-probabilities.
    """
    # .float(): log_softmax in bf16 loses too much precision over a 150k-token vocabulary.
    logits = model(input_ids=ids, attention_mask=att).logits[:, :-1].float()
    targets = ids[:, 1:]
    # Pick out log p(actual next token) at every position.
    logp = torch.gather(F.log_softmax(logits, -1), 2, targets.unsqueeze(-1)).squeeze(-1)
    # The mask is shifted the same way: position t is scored if token t+1 is an answer token.
    return (logp * loss_mask[:, 1:]).sum(-1)


def dpo_loss_from_logprobs(pi, pi_ref, beta):
    """The DPO loss given policy and reference log-probabilities.

    Args:
        pi: (2N,) log pi(y|x) from the policy; first N rows are chosen
            answers, last N rejected (the layout of collate_pairs).
        pi_ref: (2N,) the same quantity from the frozen reference model.
        beta: strength of the implicit KL constraint (0.1 is standard).

    Returns:
        (loss, r_chosen, r_rejected): the scalar mean loss, plus the detached
        implicit rewards of the chosen and rejected answers (N each), used for
        the reward-accuracy and reward-margin metrics.
    """
    n = pi.shape[0] // 2
    rewards = beta * (pi - pi_ref)                  # implicit rewards
    r_chosen, r_rejected = rewards[:n], rewards[n:]
    # logsigmoid is the numerically stable form of log(sigmoid(x)).
    loss = -F.logsigmoid(r_chosen - r_rejected).mean()
    return loss, r_chosen.detach(), r_rejected.detach()


def dpo_loss(policy, ref, batch, beta):
    """Run policy and reference on a collated batch and return the DPO loss.

    Args:
        policy: the model being trained.
        ref: the frozen reference model (no gradients are computed for it).
        batch: (ids, att, loss_mask) from collate_pairs, already on device.
        beta: see dpo_loss_from_logprobs.

    Returns:
        (loss, r_chosen, r_rejected), as in dpo_loss_from_logprobs.
    """
    ids, att, lm = batch
    pi = answer_logprob(policy, ids, att, lm)
    with torch.no_grad():                           # the reference model is frozen
        pi_ref = answer_logprob(ref, ids, att, lm)
    return dpo_loss_from_logprobs(pi, pi_ref, beta)
