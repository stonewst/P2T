"""Group short `\\n\\n` segments into PRM steps of a controlled token length.

Short steps tend to receive high PRM scores regardless of content, which lets the policy
inflate its process reward by splitting its reasoning ever finer. Packing merges
consecutive segments into blocks whose length lies within [min_tok, max_tok]. It only
merges, never splits.

Objective
---------
    cost(L) = max(0, min_tok - L)^2 + over * max(0, L - max_tok)^2

The squared penalty spreads violations across blocks; `over` weights the ceiling against
the floor.

Solvers
-------
"dp" finds the optimal partition; "greedy" repeatedly merges each too-short block into
its better neighbour.
"""

from __future__ import annotations

from itertools import accumulate
from typing import List, Sequence, Tuple

PACK_ALGORITHMS = ("dp", "greedy")

__all__ = ["PACK_ALGORITHMS", "pack_blocks", "block_lengths", "block_cost"]


def block_cost(length: int, min_tok: int, max_tok: int, over: float = 1.0) -> float:
    c = 0.0
    if length < min_tok:
        c += float(min_tok - length) ** 2
    if length > max_tok:
        c += over * float(length - max_tok) ** 2
    return c


def block_lengths(n_tokens: Sequence[int],
                  blocks: Sequence[Tuple[int, int]]) -> List[int]:
    pre = [0, *accumulate(n_tokens)]
    return [pre[b] - pre[a] for a, b in blocks]


def total_cost(n_tokens: Sequence[int], blocks: Sequence[Tuple[int, int]],
               min_tok: int, max_tok: int, over: float = 1.0) -> float:
    return sum(block_cost(L, min_tok, max_tok, over)
               for L in block_lengths(n_tokens, blocks))


def _pack_dp(n_tokens: Sequence[int], min_tok: int, max_tok: int,
             over: float, window: int) -> List[Tuple[int, int]]:
    """Exact partition by dynamic programming.

    The best partition of the first j segments is the best partition of the first i-1
    plus the cost of block [i, j].
    """
    K = len(n_tokens)
    pre = [0, *accumulate(n_tokens)]
    span_cap = max(window * max_tok, max(n_tokens))

    inf = float("inf")
    best = [inf] * (K + 1)
    best[0] = 0.0
    cut = [0] * (K + 1)
    for j in range(1, K + 1):
        # Descending i with strict <: ties keep the shortest trailing block.
        for i in range(j, 0, -1):
            L = pre[j] - pre[i - 1]
            # A single segment over the cap may still stand alone.
            if L > span_cap and i < j:
                break
            c = best[i - 1] + block_cost(L, min_tok, max_tok, over)
            if c < best[j]:
                best[j] = c
                cut[j] = i - 1
    blocks: List[Tuple[int, int]] = []
    j = K
    while j > 0:
        i = cut[j]
        blocks.append((i, j))
        j = i
    blocks.reverse()
    return blocks


def _pack_greedy(n_tokens: Sequence[int], min_tok: int, max_tok: int,
                 over: float, window: int) -> List[Tuple[int, int]]:
    """Repeatedly fold the first too-short block into its better neighbour.

    Scans left to right and restarts after every merge.
    """
    K = len(n_tokens)
    bounds = [(k, k + 1) for k in range(K)]
    pre = [0, *accumulate(n_tokens)]

    def length(b):
        return pre[b[1]] - pre[b[0]]

    while True:
        merged = False
        for m in range(len(bounds)):
            if length(bounds[m]) >= min_tok:
                continue
            L = length(bounds[m])
            prev_L = length(bounds[m - 1]) + L if m > 0 else None
            next_L = L + length(bounds[m + 1]) if m + 1 < len(bounds) else None
            if prev_L is None and next_L is None:
                break                                   # single block, nothing to do
            fits_prev = prev_L is not None and prev_L <= max_tok
            fits_next = next_L is not None and next_L <= max_tok

            if fits_prev and fits_next:
                side = "prev" if prev_L <= next_L else "next"
            elif fits_prev:
                side = "prev"
            elif fits_next:
                side = "next"
            else:
                # Neither side fits under the ceiling: merge only if it lowers the cost.
                stay_m = block_cost(L, min_tok, max_tok, over)
                choice = None
                for nb, merged_L, name in ((m - 1, prev_L, "prev"),
                                           (m + 1, next_L, "next")):
                    if merged_L is None:
                        continue
                    delta = (block_cost(merged_L, min_tok, max_tok, over)
                             - stay_m
                             - block_cost(length(bounds[nb]), min_tok, max_tok, over))
                    if choice is None or delta < choice[0]:
                        choice = (delta, name)
                if choice is None or choice[0] >= 0:
                    continue
                side = choice[1]
            if side == "prev":
                bounds[m - 1] = (bounds[m - 1][0], bounds[m][1])
                bounds.pop(m)
            else:
                bounds[m] = (bounds[m][0], bounds[m + 1][1])
                bounds.pop(m + 1)
            merged = True
            break
        if not merged:
            return bounds


def pack_blocks(n_tokens: Sequence[int], min_tok: int, max_tok: int,
                over: float = 1.0, algorithm: str = "dp",
                window: int = 4) -> List[Tuple[int, int]]:
    """Group consecutive segments into blocks.

    Args:
        n_tokens:  policy-token count of each `\\n\\n` segment, in order. Must include
                   the delimiter tokens, which belong to the segment they terminate.
        min_tok:   floor; a shorter block is penalised. 0 disables packing entirely.
        max_tok:   ceiling; a longer block is penalised.
        over:      weight of the ceiling penalty against the floor penalty.
        algorithm: "dp" (exact) or "greedy".
        window:    a block may span at most window*max_tok tokens; bounds the DP.

    Returns:
        [(start, end), ...] segment indices, end exclusive, covering 0..len(n_tokens).
        One block per segment when packing is off or there is nothing to merge.
    """
    K = len(n_tokens)
    if K == 0:
        return []
    if min_tok <= 0:
        return [(k, k + 1) for k in range(K)]
    if algorithm not in PACK_ALGORITHMS:
        raise ValueError(f"unknown pack algorithm {algorithm!r}, want one of {PACK_ALGORITHMS}")
    if max_tok < min_tok:
        raise ValueError(f"pack_max_token_per_process ({max_tok}) < pack_min_token_per_process ({min_tok})")
    solver = _pack_dp if algorithm == "dp" else _pack_greedy
    return solver([int(x) for x in n_tokens], int(min_tok), int(max_tok),
                  float(over), int(window))
