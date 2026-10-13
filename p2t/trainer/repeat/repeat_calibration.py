"""Overwrite the PRM step reward inside the repeated region of a response.

The PRM tends to give repeated content high step scores, so every duplicated step would
otherwise be rewarded. This module replaces R_s on those steps with a configured constant
before the advantage is computed.

Coordinates. The detector receives ``responses[i][:valid_len_i]``, so ``t1_bpe`` indexes
the response dimension directly, the same axis as ``process_step_idx``.

``t_repeat_1`` is the first token of the second occurrence of the repeating unit and
``t_repeat_2`` the first token of the third. The first occurrence is left untouched.

Two windows are available, selected by ``algorithm.use_mask_after_t2``:

``[t1, valid_len)`` -- calibrate the whole repeated tail (default). When the tail makes
    up most of the response, it dominates the token-weighted mean and the penalty is
    diluted.

``[t1, t2)`` plus dropping ``[t2, valid_len)`` from the loss -- calibrate one copy of
    the repeating unit and stop supervising the rest. The second copy stays supervised
    with the calibrated reward.

The trimmed tail must leave the normalization pool as well as the loss mask (see
``build_repeat_t2_array``).

Calibration only reshapes rewards within a response: if every step of a response is
calibrated, per-response centering maps them all to 0. Fully degenerate responses are
handled by the response-level repeat penalty instead.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

__all__ = [
    "STEP_DROPPED",
    "apply_process_reward_calibration",
    "build_repeat_t1_array",
    "build_repeat_t2_array",
    "step_is_in_repeat",
    "step_supervised_positions",
]

#: Per-step flag for a step that lies wholly past t2 and was dropped from the pool, as
#: distinct from 0 (kept as-is) and 1 (overwritten by the calibration value).
STEP_DROPPED = -1


def build_repeat_t1_array(
    records: Sequence,
    valid_lengths: Sequence[int],
) -> np.ndarray:
    """Reduce the detection records to one calibration anchor per response.

    Args:
        records:       one ``RepeatDetectRecord`` per response, in batch order.
        valid_lengths: valid BPE length per response (``response_mask.sum(-1)``).

    Returns:
        int64 array of shape (bs,): the token index from which the response is
        considered repeated, or -1 for responses with no repeat to calibrate.

    t1 is clamped into ``[0, valid_len - 1]``.
    """
    n = len(records)
    assert len(valid_lengths) == n, (
        f"len mismatch: records={n}, valid_lengths={len(valid_lengths)}"
    )

    t1_arr = np.full(n, -1, dtype=np.int64)
    for i, (rec, vlen) in enumerate(zip(records, valid_lengths)):
        vlen = int(vlen)
        if not rec.has_repeat or vlen <= 0 or rec.t1_bpe < 0:
            continue
        t1_arr[i] = min(max(int(rec.t1_bpe), 0), vlen - 1)
    return t1_arr


def build_repeat_t2_array(
    records: Sequence,
    valid_lengths: Sequence[int],
    min_keep: int = 1,
) -> np.ndarray:
    """Reduce the detection records to one trim anchor per response.

    Args:
        records:       one ``RepeatDetectRecord`` per response, in batch order.
        valid_lengths: valid BPE length per response (``response_mask.sum(-1)``).
        min_keep:      never trim a response down to fewer than this many supervised
                       tokens (a fully masked response would divide by zero in the loss).

    Returns:
        int64 array of shape (bs,): the token index from which the response stops being
        supervised, or -1 for responses that keep their full length.

    ``t2_bpe == -1`` means there is no third occurrence, so nothing is trimmed and -1 is
    returned.

    Callers must apply this anchor to both the loss mask and the step-reward pool;
    otherwise the trimmed tail still affects the per-response centering.
    """
    n = len(records)
    assert len(valid_lengths) == n, (
        f"len mismatch: records={n}, valid_lengths={len(valid_lengths)}"
    )

    t2_arr = np.full(n, -1, dtype=np.int64)
    for i, (rec, vlen) in enumerate(zip(records, valid_lengths)):
        vlen = int(vlen)
        if not rec.has_repeat or vlen <= 0 or rec.t2_bpe < 0:
            continue
        t2 = min(int(rec.t2_bpe), vlen)
        if t2 >= vlen or t2 < min_keep:
            continue                 # nothing to drop, or too little would survive
        t2_arr[i] = t2
    return t2_arr


def step_supervised_positions(pos: np.ndarray, t2: int) -> np.ndarray:
    """Restrict a step's token positions to the part that is still supervised.

    `pos` holds the response-dimension indices of the step's policy tokens; `t2` is the
    trim anchor from ``build_repeat_t2_array`` (-1 to keep everything).

    A step lying wholly at or after t2 comes back empty and should be dropped from the
    pool.
    """
    if t2 < 0 or pos.size == 0:
        return pos
    return pos[pos < t2]


def step_is_in_repeat(pos: np.ndarray, t1: int) -> bool:
    """Does this step overlap the repeated region starting at ``t1``?

    `pos` holds the response-dimension indices of the step's policy tokens. Pass the
    output of ``step_supervised_positions`` to confine the question to ``[t1, t2)``.

    Any token at or after t1 marks the step, including the step that straddles t1.
    """
    if t1 < 0 or pos.size == 0:
        return False
    return bool(pos[-1] >= t1)      # pos comes from np.nonzero, hence sorted


def apply_process_reward_calibration(
    process_reward,
    process_step_idx,
    response_mask,
    t1_arr: Optional[np.ndarray],
    t2_arr: Optional[np.ndarray],
    calib_value: float,
):
    """Rewrite the step reward inside the repeated region and optionally trim the mask.

    Operates on the broadcast representation: ``process_reward[i, t]`` (R_s of the token's
    step) and ``process_step_idx[i, t]`` (the token's step id). Two edits are made:

      * ``process_reward`` is set to ``calib_value`` on every supervised step whose span
        overlaps ``[t1, end)`` (or ``[t1, t2)`` when trimming). Supervised tokens with
        ``step_idx == -1`` (e.g. EOS) carry the last step's R_s and are calibrated with it.
      * ``[t2, end)`` is dropped from a clone of ``response_mask`` (which is a view of
        ``attention_mask`` and must not be edited in place).

    The window is selected by which arrays are passed:
      * ``t1_arr`` given, ``t2_arr`` None      -> calibrate ``[t1, end)``, no trim
        (the wide window; use_process_p2t_reward_calib only).
      * ``t1_arr`` None, ``t2_arr`` given      -> trim only, no reward rewrite
        (use_mask_after_t2 without calibration).
      * both given                             -> calibrate ``[t1, t2)`` and trim the
        tail (t2 == -1 for a response keeps the wide window and no trim, matching
        build_repeat_t2_array).

    Args:
        process_reward:   (bs, L) torch tensor, per-token R_s (broadcast within a step).
        process_step_idx: (bs, L) torch tensor, step id per token (-1 = none / EOS).
        response_mask:    (bs, L) torch tensor (a view of attention_mask -- never edited).
        t1_arr:           (bs,) int calibration anchor per response, -1 = skip; None to
                          disable calibration entirely.
        t2_arr:           (bs,) int trim anchor per response, -1 = keep-all; None to
                          disable trimming entirely.
        calib_value:      value substituted for R_s on calibrated steps.

    Returns:
        (process_reward_new, response_mask_new, response_mask_full, calibrated_mask, metrics)
          process_reward_new: (bs, L) clone with R_s overwritten on calibrated steps.
          response_mask_new:  (bs, L) trimmed clone (== a clone of the input when no trim).
          response_mask_full: (bs, L) untrimmed clone (for dumps that show the full rollout).
          calibrated_mask:    (bs, L) bool, True where R_s was overwritten (supervised only;
                              includes the step_idx == -1 tokens that trail a calibrated
                              last step).
          metrics:            dict of repeat/* scalars describing what fired.
    """
    import torch

    device = process_reward.device
    bs, L = process_reward.shape
    prew = process_reward.detach().float().cpu().numpy().copy()
    sidx = process_step_idx.detach().cpu().numpy().astype(np.int64)
    mask_full = response_mask.detach().cpu().numpy().astype(bool)
    mask = mask_full.copy()

    calibrated = np.zeros((bs, L), dtype=bool)
    n_calib_steps = 0
    n_present_steps = 0
    n_dropped_steps = 0
    n_calib_tokens = 0
    n_valid_tokens_full = int(mask_full.sum())
    resp_calibrated = np.zeros(bs, dtype=bool)
    resp_trimmed = np.zeros(bs, dtype=bool)

    for i in range(bs):
        t1 = int(t1_arr[i]) if t1_arr is not None else -1
        t2 = int(t2_arr[i]) if t2_arr is not None else -1

        # (1) trim [t2, end) from the supervised mask
        if t2 >= 0:
            mask[i, t2:] = False
            resp_trimmed[i] = True
            # steps that left the pool because of the trim
            s_full = sidx[i].copy()
            s_full[~mask_full[i]] = -1
            present_full = np.unique(s_full[s_full >= 0]).size
        # steps still present among the (trimmed) supervised tokens
        s = sidx[i].copy()
        s[~mask[i]] = -1
        present = np.unique(s[s >= 0])
        n_present_steps += present.size
        if t2 >= 0:
            n_dropped_steps += max(0, present_full - present.size)

        if t1 < 0:
            continue   # calibration disabled or nothing to calibrate for this response

        # (2) overwrite R_s on every supervised step overlapping [t1, end-of-window)
        last_calibrated = False
        for k in present:
            pos = np.nonzero(s == k)[0]           # supervised positions of step k (sorted)
            if pos.size == 0:
                continue
            if int(pos[-1]) >= t1:                # step touches the repeated region
                prew[i, pos] = calib_value
                calibrated[i, pos] = True
                n_calib_steps += 1
                n_calib_tokens += int(pos.size)
                resp_calibrated[i] = True
                if k == present[-1]:
                    last_calibrated = True

        # (3) Special tokens (step_idx = -1, e.g. EOS) carry the last step's R_s, so they are
        #     calibrated together with the last step. Not counted in n_calib_steps.
        if last_calibrated:
            spec = np.nonzero(mask[i] & (sidx[i] < 0))[0]
            if spec.size:
                prew[i, spec] = calib_value
                calibrated[i, spec] = True
                n_calib_tokens += int(spec.size)

    process_reward_new = torch.from_numpy(prew).to(device=device, dtype=process_reward.dtype)
    response_mask_new = torch.from_numpy(mask).to(device=device, dtype=response_mask.dtype)
    response_mask_full = torch.from_numpy(mask_full).to(device=device, dtype=response_mask.dtype)
    calibrated_mask = torch.from_numpy(calibrated).to(device=device)

    return process_reward_new, response_mask_new, response_mask_full, calibrated_mask, {}
