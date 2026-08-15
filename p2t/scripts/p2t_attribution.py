"""
P2T (Process-to-Token) reward attribution core.

Given a Process Reward Model (PRM) that emits a per-step scalar at each
separator position, redistribute each step's reward onto the tokens *inside*
that step via Vanilla-Gradient attribution:

    I_i = grad_{e_i}( S_s ) . (e_i - e_null)          # token influence
    w   = softmax( I_i / temperature )  over tokens of step s
    R_i = R_s * w_i                                    # token-level reward

R_i redistributes the step reward R_s across its tokens by influence weight
(sum_i R_i = R_s). A flat per-token baseline (the paper's `+ omega*R_s` term) is
NOT added here: it is an independent, separately-weighted advantage component
(process_token_reward, broadcast R_s). Eq.5's `R_s + omega*R_s*w_i` is recovered
downstream as process_coef*R_s + p2t_coef*(R_s*w_i).

Two orthogonal knobs (both chosen empirically, see compare_ab_attribution.py):

  attribution_target : which scalar S_s we differentiate to get the token
                       influence (fixes gradient saturation). All three point
                       along the positive (class-1) direction so I_i>0 always
                       means "this token pushes the step toward being correct".
      - "prob"       : S_s = softmax(z)_1              (grad gate p(1-p), saturates)
      - "logprob"    : S_s = log_softmax(z)_1          (grad gate (1-p),  ~also dead when p->1)
      - "logit_diff" : S_s = z_1 - z_0                 (grad gate 1,      no saturation, DEFAULT)

  backward_mode      : how many backward passes.
      - "single"     : one backward on sum_s S_s -> O(1); early-step tokens get
                       contaminated by later steps (empirically negligible on the
                       final token reward). DEFAULT: S backward passes on a long
                       (up to 2k+8k) sequence make per_step ~S x slower, so single
                       is the practical default.
      - "per_step"   : one backward per step -> I_i uses ONLY its own step reward
                       (most accurate; costs S backward passes; for ablation).

NOTE: the *reward value* R_s used in R_i is ALWAYS the probability p=softmax(z)_1
(the meaningful [0,1] process score). attribution_target only changes the scalar
we back-propagate to decide the *within-step ranking* of tokens.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F


ATTRIBUTION_TARGETS = ("prob", "logprob", "logit_diff")
BACKWARD_MODES = ("per_step", "single")


@dataclass
class P2TConfig:
    attribution_target: str = "logit_diff"  # prob | logprob | logit_diff
    backward_mode: str = "single"          # single (default, O(1)) | per_step (ablation)
    # token reward = R_s * softmax(I_i): redistribute the step reward across its
    # tokens by influence weight (sum over the step == R_s). The flat baseline is
    # a separate advantage component (process_token_reward), so it is not added here.
    temperature: float = 1.0               # softmax temperature over I_i within a step
    positive_class: int = 1                 # class index treated as "correct"

    def validate(self):
        assert self.attribution_target in ATTRIBUTION_TARGETS, self.attribution_target
        assert self.backward_mode in BACKWARD_MODES, self.backward_mode
        assert self.temperature > 0, self.temperature


@dataclass
class P2TResult:
    token_reward: torch.Tensor        # (L,) token-level reward over PRM input_ids
    influence: torch.Tensor           # (L,) I_i per token (0 on sep/pad-only slots)
    step_rewards: List[float]         # per-step R_s (probability p), len = #sep
    step_logits: List[list]           # per-step raw PRM logits [neg, pos], len = #sep
    sep_positions: List[int]          # index of each <sep> in input_ids
    seg_bounds: List[tuple]           # (start, sep_pos) token span per step (sep excluded)


def _step_scalars(logits_sep: torch.Tensor, cfg: P2TConfig):
    """Given logits at a sep position (2,), return (reward_value_p, attribution_scalar)."""
    pos, neg = cfg.positive_class, 1 - cfg.positive_class
    logp = F.log_softmax(logits_sep, dim=-1)
    p = logp[pos].exp()                                  # reward value (prob), differentiable
    if cfg.attribution_target == "prob":
        s = p
    elif cfg.attribution_target == "logprob":
        s = logp[pos]
    else:  # logit_diff
        s = logits_sep[pos] - logits_sep[neg]
    return p, s


def compute_p2t_token_rewards(
    prm,
    input_ids: torch.Tensor,          # (1, L) long, containing the sep tokens
    sep_id: int,
    pad_id: int,
    cfg: P2TConfig,
    attention_mask: Optional[torch.Tensor] = None,
    response_start: int = 0,
) -> P2TResult:
    """Compute P2T token-level rewards for a single sequence.

    The gradient flows through inputs_embeds; PRM params should be frozen by the
    caller (or we simply never read their .grad). Runs in fp32 for the attribution
    math regardless of the model dtype.

    response_start: token index where the assistant response begins. Tokens before
    it (chat template + user prompt) must NOT take part in the redistribution,
    otherwise the first step would "own" the whole prompt.
    """
    cfg.validate()
    assert input_ids.dim() == 2 and input_ids.size(0) == 1, "expects (1, L)"
    device = input_ids.device
    L = input_ids.size(1)
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)

    embed_layer = prm.get_input_embeddings()
    with torch.no_grad():
        base_embeds = embed_layer(input_ids)                                  # (1,L,d)
        e_null = embed_layer(torch.tensor([[pad_id]], device=device))[0, 0].float()  # (d,)
    diff = base_embeds[0].float() - e_null                                    # (L,d)

    sep_positions = (input_ids[0] == sep_id).nonzero(as_tuple=True)[0].tolist()
    # token->step segmentation: step s owns (prev_sep+1 .. cur_sep-1), sep excluded.
    # The first step starts at response_start so prompt tokens are never included.
    seg_bounds = []
    prev = response_start - 1
    for sp in sep_positions:
        seg_bounds.append((max(prev + 1, response_start), sp))
        prev = sp

    token_reward = torch.zeros(L, dtype=torch.float32, device=device)
    influence = torch.zeros(L, dtype=torch.float32, device=device)

    if len(sep_positions) == 0:
        return P2TResult(token_reward, influence, [], [], [], [])

    # ---- single shared forward ----
    inputs_embeds = base_embeds.clone().detach().requires_grad_(True)
    outputs = prm(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
    logits = outputs.logits.float()                                          # (1,L,2)

    p_list, s_list = [], []
    for sp in sep_positions:
        p, s = _step_scalars(logits[0, sp], cfg)
        p_list.append(p)
        s_list.append(s)
    step_rewards = [float(p.detach()) for p in p_list]
    step_logits = [logits[0, sp].detach().cpu().tolist() for sp in sep_positions]  # [neg, pos]

    # ---- gradients -> per-token influence ----
    if cfg.backward_mode == "single":
        grad = torch.autograd.grad(sum(s_list), inputs_embeds, retain_graph=False)[0][0].float()
        I_full = (grad * diff).sum(-1)                                       # (L,)
        for (start, sp) in seg_bounds:
            idx = list(range(start, sp))
            if idx:
                influence[torch.tensor(idx, device=device)] = I_full[torch.tensor(idx, device=device)]
    else:  # per_step
        for k, (start, sp) in enumerate(seg_bounds):
            last = (k == len(seg_bounds) - 1)
            grad = torch.autograd.grad(s_list[k], inputs_embeds, retain_graph=not last)[0][0].float()
            I_s = (grad * diff).sum(-1)
            idx = list(range(start, sp))
            if idx:
                t = torch.tensor(idx, device=device)
                influence[t] = I_s[t]

    # ---- per-step redistribution ----
    for (start, sp), Rs in zip(seg_bounds, step_rewards):
        idx = list(range(start, sp))
        if len(idx) == 0:
            continue
        t = torch.tensor(idx, device=device)
        if len(idx) == 1:
            w = torch.ones(1, device=device)
        else:
            w = F.softmax(influence[t] / cfg.temperature, dim=0)
        token_reward[t] = Rs * w      # redistribute R_s across the step by influence

    return P2TResult(token_reward, influence, step_rewards, step_logits, sep_positions, seg_bounds)
