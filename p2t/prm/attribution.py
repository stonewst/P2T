"""Token attribution score I_i: how much each PRM token moved its step's PRM score.

Scores are computed on the PRM's tokenization; `prm/align.py` maps them onto the policy
tokens and `prm/p2t_reward.py` turns them into a token-level reward.

The method is gradient x input on the PRM's input embeddings:

    I_i = grad_{e_i}( S_s ) . (e_i - e_pad)

`e_pad` is a "no token here" baseline, so I_i is a first-order estimate of how much S_s
would change if token i were replaced by padding. The PRM's parameters are frozen, so
gradients flow through `inputs_embeds` only.

p2t_attribution_target
----------------------
The scalar S_s that is differentiated. I_i > 0 means the token pushed the step toward
being correct:

    logit_diff  S_s = z_pos - z_neg          gradient gate 1        no saturation
    logprob     S_s = log_softmax(z)_pos     gradient gate (1-p)    vanishes as p -> 1
    prob        S_s = softmax(z)_pos         gradient gate p(1-p)   vanishes as p -> 1

logit_diff is the default because PRM step scores are often close to 1. S_s only ranks
tokens within a step; the step reward R_s still comes from the scorer's
`reward_transform`.

backward_mode
-------------
    single    one backward on sum_s S_s. The PRM is causal, so a token in step s also
              collects gradient from every later step. Approximate, but one backward.
    per_step  one backward per step, so I_i depends only on its own step. Exact, but
              costs one backward per step.

Cost
----
Attribution costs roughly 4x the inference-only PRM time (forward + backward +
recomputation). Activation recomputation (`attribution_mode`) keeps memory bounded on
long sequences.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

ATTRIBUTION_TARGETS = ("logit_diff", "logprob", "prob")
BACKWARD_MODES = ("single", "per_step")

__all__ = [
    "ATTRIBUTION_TARGETS",
    "BACKWARD_MODES",
    "assert_no_dropout",
    "attribution_mode",
    "prm_token_influence",
    "step_segments",
]


def assert_no_dropout(model: nn.Module) -> None:
    """Fail if the model has active dropout, which `attribution_mode` would switch on.

    Dropout in train() mode would make I_i non-deterministic.
    """
    live = [
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Dropout) and float(module.p) > 0.0
    ]
    config = getattr(model, "config", None)
    cfg_drop = {
        key: value
        for key, value in (vars(config) if config is not None else {}).items()
        if "dropout" in key and isinstance(value, (int, float)) and value > 0
    }
    if live or cfg_drop:
        raise ValueError(
            "token attribution needs train() mode to enable activation recomputation, but "
            f"this PRM has live dropout (modules={live}, config={cfg_drop}). Attribution "
            "would become non-deterministic; set p2t_attribution_grad_ckpt=False to "
            "stay in eval() mode, at the cost of holding every activation."
        )


@contextmanager
def attribution_mode(prm: nn.Module, grad_checkpointing: bool):
    """Prepare the PRM for a gradient pass and restore its state afterwards.

    HuggingFace only recomputes activations when `self.gradient_checkpointing and
    self.training`, so the model is switched to train() for the duration (after
    `assert_no_dropout`). With grad_checkpointing=False this is a no-op.
    """
    was_training = prm.training
    if grad_checkpointing:
        assert_no_dropout(prm)
        prm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        prm.train()
    try:
        yield
    finally:
        if grad_checkpointing:
            prm.gradient_checkpointing_disable()
        prm.train(was_training)


def step_segments(sep_positions: Sequence[int], response_start: int = 0) -> List[Tuple[int, int]]:
    """Token span [start, end) of each step, excluding the separator.

    Step s runs from just after the previous separator up to its own. The first step
    starts no earlier than `response_start`, so prompt tokens never belong to a step.
    """
    bounds, prev = [], response_start - 1
    for sep in sep_positions:
        bounds.append((max(prev + 1, response_start), int(sep)))
        prev = int(sep)
    return bounds


def _target_scalar(logits_sep: torch.Tensor, target: str, positive_class: int) -> torch.Tensor:
    """The differentiable scalar S_s read off the 2-class head at one separator position."""
    pos, neg = positive_class, 1 - positive_class
    if target == "logit_diff":
        return logits_sep[pos] - logits_sep[neg]
    if target == "logprob":
        return F.log_softmax(logits_sep, dim=-1)[pos]
    if target == "prob":
        return F.softmax(logits_sep, dim=-1)[pos]
    raise ValueError(f"unknown p2t_attribution_target: {target!r}, want one of {ATTRIBUTION_TARGETS}")


def prm_token_influence(
    prm: nn.Module,
    input_ids: torch.Tensor,             # (1, L) long, separators included
    attention_mask: torch.Tensor,        # (1, L)
    sep_positions: Sequence[int],        # index of each separator in input_ids
    *,
    pad_id: int,
    target: str = "logit_diff",
    backward_mode: str = "single",
    positive_class: int = 1,
    response_start: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """One PRM forward + backward -> per-token influence and the logits of that forward.

    The logits are returned (detached) so the caller reads R_s off the same forward. The
    attribution arithmetic runs in fp32 regardless of the model dtype.

    Returns:
        influence: (L,) float32, 0 on separators, on prompt tokens and on anything outside
            a step span.
        logits: (L, num_classes) float32, detached.
    """
    if backward_mode not in BACKWARD_MODES:
        raise ValueError(f"unknown backward_mode: {backward_mode!r}, want one of {BACKWARD_MODES}")
    if input_ids.dim() != 2 or input_ids.size(0) != 1:
        raise ValueError(f"prm_token_influence expects a single sequence (1, L), got {tuple(input_ids.shape)}")
    if pad_id is None:
        raise ValueError(
            "the PRM tokenizer has no pad_token_id, so there is no baseline embedding to "
            "measure I_i against; set one on the tokenizer or turn attribution off"
        )

    device = input_ids.device
    length = input_ids.size(1)
    influence = torch.zeros(length, dtype=torch.float32, device=device)

    embed = prm.get_input_embeddings()
    with torch.no_grad():
        base = embed(input_ids)                                                   # (1,L,d)
        e_pad = embed(torch.tensor([[int(pad_id)]], device=device))[0, 0]          # (d,)
    delta = (base[0] - e_pad).float()                                             # (L,d)

    inputs_embeds = base.detach().requires_grad_(True)
    logits = prm(
        inputs_embeds=inputs_embeds, attention_mask=attention_mask, use_cache=False
    ).logits.float()[0]                                                           # (L,C)

    bounds = step_segments(sep_positions, response_start)
    if not bounds:
        return influence, logits.detach()
    scalars = [_target_scalar(logits[sep], target, positive_class) for _, sep in bounds]

    if backward_mode == "single":
        grad = torch.autograd.grad(torch.stack(scalars).sum(), inputs_embeds)[0][0].float()
        whole = (grad * delta).sum(-1)                                            # (L,)
        for start, end in bounds:
            influence[start:end] = whole[start:end]
    else:
        # Keep the graph until the last non-empty step.
        last = max((k for k, (start, end) in enumerate(bounds) if end > start), default=-1)
        for k, (start, end) in enumerate(bounds):
            if end <= start:
                continue
            grad = torch.autograd.grad(
                scalars[k], inputs_embeds, retain_graph=(k != last)
            )[0][0].float()
            influence[start:end] = (grad * delta).sum(-1)[start:end]

    # Treat non-finite values as "no influence".
    influence = torch.nan_to_num(influence, nan=0.0, posinf=0.0, neginf=0.0)
    return influence.detach(), logits.detach()
