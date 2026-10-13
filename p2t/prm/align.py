"""Character-span alignment between PRM tokens and policy tokens.

The PRM and the policy usually have different tokenizers, so tokens are matched through
character positions:

    response = s0 + "\n\n" + s1 + "\n\n" + s2          (policy side)

Each step's text is identical on both sides. `owning_segment` assigns every policy token
to a step, and `transfer_influence` carries the per-token attribution I_i (computed on
PRM tokens, see prm/attribution.py) over to the policy tokens.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np


def token_char_spans(tokenizer, ids: Sequence[int]) -> Tuple[List[Tuple[int, int]], str]:
    """Char span of every token in the text obtained by concatenating per-token decodes.

    Avoids `return_offsets_mapping` so it also works for generated token ids. Byte-level
    BPE decoding is concatenative, so `"".join(per-token decode) == decode(all ids)`.
    """
    pieces = [tokenizer.decode([int(i)], skip_special_tokens=False) for i in ids]
    spans, pos = [], 0
    for p in pieces:
        spans.append((pos, pos + len(p)))
        pos += len(p)
    return spans, "".join(pieces)


def overlap(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def response_step_spans(steps: Sequence[str], delimiter: str = "\n\n") -> List[Tuple[int, int]]:
    """Char span of each step inside the original response string."""
    spans, pos = [], 0
    for k, s in enumerate(steps):
        spans.append((pos, pos + len(s)))
        pos += len(s)
        if k != len(steps) - 1:
            pos += len(delimiter)
    return spans


def owning_segment(policy_spans: Sequence[Tuple[int, int]],
                   seg_spans: Sequence[Tuple[int, int]]) -> np.ndarray:
    """Which segment each policy token belongs to (-1 only if it precedes them all).

    A token that lies entirely inside a delimiter (`\n\n`) belongs to the segment it
    terminates.
    """
    n = len(seg_spans)
    out = np.full(len(policy_spans), -1, dtype=np.int64)
    for i, pspan in enumerate(policy_spans):
        best_k, best_ov = -1, 0
        for k in range(n):
            ov = overlap(pspan, seg_spans[k])
            if ov > best_ov:
                best_k, best_ov = k, ov
        if best_k < 0:
            # token lies entirely inside a delimiter: attach it to the segment it ends
            for k in range(n):
                if seg_spans[k][1] <= pspan[0]:
                    best_k = k
        out[i] = best_k
    return out


def conversation_step_spans(prm_text: str, steps: Sequence[str],
                            sep: str) -> List[Tuple[int, int]]:
    """Char span of each step inside the PRM-side text.

    The PRM's assistant content is `sep.join(steps) + sep`. It is located with `rfind`
    (the assistant turn is last, so prompt text quoting the same content cannot match)
    and then split at the separators.

    Returns [] when the content is not found.
    """
    content = sep.join(steps) + sep
    start = prm_text.rfind(content)
    if start < 0:
        return []
    spans, pos = [], start
    for s in steps:
        spans.append((pos, pos + len(s)))
        pos += len(s) + len(sep)
    return spans


def transfer_influence(
    prm_spans: Sequence[Tuple[int, int]],       # char spans of PRM tokens in prm_text
    prm_influence: np.ndarray,                  # (Lprm,) I_i per PRM token
    conv_spans: Sequence[Tuple[int, int]],      # char span of each step in prm_text
    resp_spans: Sequence[Tuple[int, int]],      # char span of each step in the response
    policy_spans: Sequence[Tuple[int, int]],    # char spans of policy tokens in the response
    policy_step_idx: np.ndarray,                # (Lp,) owning step per policy token, -1 = none
) -> np.ndarray:
    """Carry PRM-token influence onto policy tokens through character positions.

    A step's text is byte-identical on both sides, so inside a step the char offset is
    preserved up to a constant shift:

        resp_char = resp_spans[k][0] + (prm_char - conv_spans[k][0])

    Each policy token takes the overlap-length-weighted mean of the influence of the PRM
    tokens it overlaps, so its influence does not depend on how finely either tokenizer
    splits the text.

    Tokens without overlap (e.g. the `\n\n` delimiters, which the PRM side replaces by
    the separator token) get 0. Runs in linear time using a moving pointer per step.

    Returns:
        (Lp,) float64 influence per policy token.
    """
    n_steps = min(len(conv_spans), len(resp_spans))
    # bucket PRM tokens by step, mapped into response coordinates
    per_step: List[List[Tuple[int, int, float]]] = [[] for _ in range(n_steps)]
    k = 0
    for (c0, c1), infl in zip(prm_spans, prm_influence):
        while k < n_steps and conv_spans[k][1] <= c0:
            k += 1
        if k >= n_steps:
            break
        a, b = conv_spans[k]
        lo, hi = max(c0, a), min(c1, b)
        if hi <= lo:
            continue                      # a separator token, or still inside the prompt
        shift = resp_spans[k][0] - a
        per_step[k].append((lo + shift, hi + shift, float(infl)))

    out = np.zeros(len(policy_spans), dtype=np.float64)
    for k, mapped in enumerate(per_step):
        if not mapped:
            continue
        idx = np.nonzero(policy_step_idx == k)[0]     # ascending, so p0 is non-decreasing
        j = 0
        for i in idx:
            p0, p1 = policy_spans[i]
            while j < len(mapped) and mapped[j][1] <= p0:
                j += 1
            num, den = 0.0, 0
            m = j
            while m < len(mapped) and mapped[m][0] < p1:
                ov = min(mapped[m][1], p1) - max(mapped[m][0], p0)
                if ov > 0:
                    num += mapped[m][2] * ov
                    den += ov
                m += 1
            if den > 0:
                out[i] = num / den
    return out
