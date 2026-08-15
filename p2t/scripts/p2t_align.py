"""
Character-span alignment between PRM tokens and POLICY tokens.

Validated by p2t_dev/test_align.py: 6 real rollouts x 12 checks, all passing,
including the spike test (all influence on one known word must land on exactly
that word after alignment, even when the two tokenizers split it differently,
e.g. "Substitute" -> ["Sub", "stitute"]).

Why this is needed
------------------
P2T computes the token influence I_i on the *PRM's* tokenization (ReasonFlux-PRM,
Qwen2 vocab). RL needs those rewards on the *policy's* response tokens (e.g. Qwen3
vocab). The tokenizations differ, so we align through character positions.

Mapping strategy (per step, exact)
----------------------------------
The PRM input is a chat-templated conversation where the response steps are joined
by the separator token instead of "\\n\\n":

    response      = s0 + "\\n\\n" + s1 + "\\n\\n" + s2
    prm assistant = s0 + SEP      + s1 + SEP      + s2 + SEP

Each step's text is byte-identical in both strings, so within a step the relative
character offset is preserved:

    resp_char = step_start_in_response + (conv_char - step_start_in_conversation)

Every policy token (a char span in the response) then collects the influence of the
PRM tokens whose mapped spans overlap it, weighted by overlap length.

What is transferred: the raw influence I, NOT the final reward. The reward is
recomputed on the policy side as R_i = N * R_s * softmax(I'_i) (N = tokens in the
step) so that the influence-weighted redistribution sums to N*R_s per step -- the
same total as the broadcast process reward -- and stays exact under the policy
tokenization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import numpy as np


# ------------------------------------------------------------------ char spans
def token_char_spans(tokenizer, ids: Sequence[int]) -> Tuple[List[Tuple[int, int]], str]:
    """Char span of every token in the text obtained by concatenating per-token decodes.

    We deliberately avoid `return_offsets_mapping` so that this also works for token
    ids that came from generation (not from encoding a string). Byte-level BPE
    decoding is concatenative, which the caller should verify with `verify_spans`.
    """
    pieces = [tokenizer.decode([int(i)], skip_special_tokens=False) for i in ids]
    spans, pos = [], 0
    for p in pieces:
        spans.append((pos, pos + len(p)))
        pos += len(p)
    return spans, "".join(pieces)


def verify_spans(tokenizer, ids: Sequence[int], text: str) -> Tuple[bool, str]:
    """Check that per-token decode concatenation reproduces the full decode."""
    full = tokenizer.decode([int(i) for i in ids], skip_special_tokens=False)
    return (full == text), full


def overlap(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


# ------------------------------------------------------------------ step spans
def response_step_spans(steps: Sequence[str], delimiter: str = "\n\n") -> List[Tuple[int, int]]:
    """Char span of each step inside the original response string."""
    spans, pos = [], 0
    for k, s in enumerate(steps):
        spans.append((pos, pos + len(s)))
        pos += len(s)
        if k != len(steps) - 1:
            pos += len(delimiter)
    return spans


def conversation_step_spans(conversation: str, steps: Sequence[str], sep: str) -> List[Tuple[int, int]]:
    """Char span of each step inside the chat-templated conversation string.

    The assistant content is `sep.join(steps) + sep`; we locate it verbatim and then
    walk the separators, which stays exact even if the prompt contains `sep`.
    """
    content = sep.join(steps) + sep
    start = conversation.rfind(content)
    if start < 0:
        raise ValueError("assistant content not found verbatim in conversation string")
    spans, pos = [], start
    for s in steps:
        spans.append((pos, pos + len(s)))
        pos += len(s) + len(sep)
    return spans


# ------------------------------------------------------------------ alignment
@dataclass
class AlignResult:
    policy_influence: np.ndarray       # (Lp,) influence transferred onto policy tokens
    policy_step_idx: np.ndarray        # (Lp,) step index per policy token (-1 = unassigned)
    policy_p2t_reward: np.ndarray      # (Lp,) N*R_s*softmax(I') redistributed on policy tokens
    policy_process_reward: np.ndarray  # (Lp,) step reward broadcast to policy tokens
    n_covered: int = 0                 # policy tokens that overlapped >=1 PRM token
    n_total: int = 0
    warnings: List[str] = field(default_factory=list)


def align_prm_influence_to_policy(
    *,
    steps: Sequence[str],
    conversation: str,
    sep: str,
    prm_spans: Sequence[Tuple[int, int]],      # char spans of PRM tokens in `conversation`
    prm_influence: np.ndarray,                  # (Lprm,) influence per PRM token
    step_rewards: Sequence[float],              # R_s per step (probability)
    policy_spans: Sequence[Tuple[int, int]],    # char spans of policy tokens in `response`
    temperature: float = 1.0,
    process_reward_scale: str = "null",         # null (R_s in [0,1]) | signed (2*R_s-1 in [-1,1])
    agg: str = "mean",                          # mean (per-char density) | sum
) -> AlignResult:
    """Transfer PRM-token influence onto policy tokens, then rebuild the rewards.

    process_reward_scale maps the raw PRM step reward R_s (a probability in [0,1])
    before it is broadcast (process) and redistributed (p2t):
        "null"   : keep R_s in [0,1]
        "signed" : R_s <- 2*R_s - 1, so a bad step can carry negative reward
    Both process and p2t inherit the mapped R_s (p2t just multiplies by softmax(I)).
    """
    n_steps = len(steps)
    assert len(step_rewards) == n_steps, (len(step_rewards), n_steps)

    resp_spans = response_step_spans(steps)
    conv_spans = conversation_step_spans(conversation, steps, sep)
    warnings: List[str] = []
    for k in range(n_steps):
        if (conv_spans[k][1] - conv_spans[k][0]) != (resp_spans[k][1] - resp_spans[k][0]):
            warnings.append(f"step {k}: length mismatch between conversation and response spans")

    # map PRM token spans (conversation coords) -> response coords, keeping step id
    mapped: List[Tuple[int, int, int, float]] = []  # (r0, r1, step, influence)
    for (c0, c1), infl in zip(prm_spans, prm_influence):
        for k in range(n_steps):
            a, b = conv_spans[k]
            if overlap((c0, c1), (a, b)) <= 0:
                continue
            s0, s1 = max(c0, a), min(c1, b)          # clip to the step
            shift = resp_spans[k][0] - a             # then shift into response coords
            mapped.append((s0 + shift, s1 + shift, k, float(infl)))
            break

    Lp = len(policy_spans)
    policy_influence = np.zeros(Lp, dtype=np.float64)
    policy_step_idx = np.full(Lp, -1, dtype=np.int64)
    n_covered = 0

    for i, pspan in enumerate(policy_spans):
        best_k, best_ov = -1, 0
        for k in range(n_steps):
            ov = overlap(pspan, resp_spans[k])
            if ov > best_ov:
                best_k, best_ov = k, ov
        if best_k < 0:
            # token lies entirely inside a "\n\n" delimiter: attach to the preceding
            # step with neutral (zero) influence, so it still gets the R_s baseline
            prev_k = -1
            for k in range(n_steps):
                if resp_spans[k][1] <= pspan[0]:
                    prev_k = k
            policy_step_idx[i] = prev_k
            continue
        policy_step_idx[i] = best_k

        num, den = 0.0, 0
        for (r0, r1, k, infl) in mapped:
            if k != best_k:
                continue
            ov = overlap(pspan, (r0, r1))
            if ov > 0:
                num += infl * ov
                den += ov
        if den > 0:
            n_covered += 1
            policy_influence[i] = num / den if agg == "mean" else num

    # ---- rebuild Eq.(3) on the policy side, per step ----
    policy_p2t_reward = np.zeros(Lp, dtype=np.float64)
    policy_process_reward = np.zeros(Lp, dtype=np.float64)
    for k in range(n_steps):
        idx = np.nonzero(policy_step_idx == k)[0]
        if idx.size == 0:
            continue
        Rs = float(step_rewards[k])
        if process_reward_scale == "signed":
            Rs = 2.0 * Rs - 1.0              # [0,1] -> [-1,1]; bad steps become negative
        policy_process_reward[idx] = Rs
        z = policy_influence[idx] / temperature
        z = z - z.max()
        w = np.exp(z)
        w = w / w.sum()
        # influence-weighted redistribution of the step's total budget N*R_s (so the
        # per-token magnitude stays ~R_s and matches the broadcast process reward;
        # sum over the step == N*R_s == sum of process over the step).
        policy_p2t_reward[idx] = idx.size * Rs * w

    return AlignResult(
        policy_influence=policy_influence,
        policy_step_idx=policy_step_idx,
        policy_p2t_reward=policy_p2t_reward,
        policy_process_reward=policy_process_reward,
        n_covered=n_covered,
        n_total=Lp,
        warnings=warnings,
    )
