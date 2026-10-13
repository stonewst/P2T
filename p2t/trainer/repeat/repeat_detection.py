"""Malignant-repeat detection for P2T training (detection only).

Flow inside ``ray_trainer.fit``::

  1. after ``_balance_batch`` (batch order is fixed from then on),
     ``launch_repeat_detection_async`` dispatches chunked Ray tasks that run
     concurrently with reward computation, PRM scoring and the log-prob passes.

  2. in the advantage stage, ``gather_repeat_detection`` returns one
     ``RepeatDetectRecord`` per response with ``(has_repeat, t_repeat_1, t_repeat_2)``
     in BPE coordinates.

  3. ``compute_response_level_flags`` and the ``compute_*_metrics`` helpers produce
     logging metrics; ``build_repeat_det_dump_fields`` produces per-sample dump columns.

``has_repeat=True`` with ``t_repeat_2 == -1`` (no third occurrence found) is treated as
``t_repeat_2 = valid_len``.

Cost controls:

  * ``collect_stage_hits`` (off by default) runs every detector level independently to
    record which ones would fire. The char level is slow on long degenerate responses,
    so enable it only for offline diagnosis.

  * ``char_max_chars`` skips the char-level fallback for long texts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import ray
import torch

from p2t.trainer.repeat.repeat_detector import (
    compute_boxed_diagnostics,
    compute_repeat_stage_hits,
    detect_malignant_repeat_pipeline,
)

REPEAT_DET_SCHEMA_VERSION = "v1_train_aligned"

# RepeatConfig fields forwarded to detect_malignant_repeat_pipeline.
_PIPELINE_KWARG_NAMES = (
    "block_size",
    "min_repeats",
    "max_consecutive_mismatch",
    "min_period",
    "min_unique_tokens_in_unit",
    "char_fallback",
    "char_block_size",
    "char_min_period",
    "char_min_unique",
    "boxed_fallback",
    "boxed_threshold",
    "boxed_dominant_ratio",
    "boxed_m_index",
    "boxed_tail_chars",
    "spam_fallback",
    "spam_word_min_count",
    "spam_word_min_ratio",
    "spam_word_min_run",
    "spam_concat_min_repeats",
    "spam_concat_min_unit_len",
    "spam_concat_max_unit_len",
)


@dataclass
class RepeatDetectRecord:
    """Per-response detection result (shared by metrics and the parquet dump)."""

    has_repeat: bool = False
    t1_bpe: int = -1
    t2_bpe: int = -1
    source: str = "none"
    t1_raw: int = -1
    t2_raw: int = -1
    t1_raw_space: str = "none"
    t2_raw_space: str = "none"
    num_bpe: int = 0
    num_words: int = 0
    t1_mapping_fallback: bool = False
    hit_word: bool = False
    hit_char: bool = False
    hit_boxed: bool = False
    hit_spam_word: bool = False
    hit_spam_concat: bool = False
    boxed_count: int = 0
    boxed_would_detect: int = 0
    dominant_boxed_ratio: float = 0.0

    def as_tuple(self) -> Tuple[bool, int, int]:
        return (self.has_repeat, self.t1_bpe, self.t2_bpe)


def repeat_records_to_tuples(
    records: List[RepeatDetectRecord],
) -> List[Tuple[bool, int, int]]:
    return [r.as_tuple() for r in records]


def _raw_coord_spaces(source: str, t2_raw: int) -> Tuple[str, str]:
    if source in ("word", "spam_word"):
        t2_space = "word" if source == "word" and t2_raw >= 0 else "none"
        return "word", t2_space
    if source in ("char", "boxed", "spam_concat"):
        t2_space = "char" if source in ("char", "boxed") and t2_raw >= 0 else "none"
        return "char", t2_space
    return "none", "none"


def build_repeat_detection_kwargs(repeat_config) -> dict:
    """Extract the ``detect_malignant_repeat_pipeline`` kwargs from ``config.repeat``."""
    return {name: getattr(repeat_config, name) for name in _PIPELINE_KWARG_NAMES}


# --------------------------------------------------------------------
# (1a) BPE <-> word mapping
# --------------------------------------------------------------------

def _build_word_mapping_from_bpe_tokens(
    bpe_tokens: List[str],
) -> Tuple[List[str], List[int]]:
    """Fold a BPE subword sequence into words, keeping a word -> BPE-start map.

    Word boundary rules (common to Qwen / GPT-2 / Llama BPE):
      * the first subword always starts a word
      * a subword starting with U+0120 or U+2581 starts a word (leading space)
      * a subword starting with U+010A starts a word (leading newline)
      * a subword whose predecessor ends with U+010A starts a word
      * special tokens (matching '<|...|>') each form their own word
      * a normal subword right after a special token starts a word

    Newlines are word boundaries so that e.g. '---' or '###' after a line break are
    seen as separate words.

    Returns:
        words:              folded word sequence
        word_to_bpe_start:  index of words[i] in the BPE sequence
    """
    words: List[str] = []
    word_to_bpe_start: List[int] = []
    cur_word = ""
    cur_start = -1

    for i, tok in enumerate(bpe_tokens):
        # convert_ids_to_tokens returns None for unknown ids; treat them as empty.
        if tok is None:
            tok = ""
        is_special = tok.startswith("<|") and tok.endswith("|>")
        prev_tok = bpe_tokens[i - 1] if i > 0 else ""
        if prev_tok is None:
            prev_tok = ""
        is_word_start = (
            i == 0
            or is_special
            or tok.startswith(("\u0120", "\u2581", "\u010a"))
            or prev_tok.endswith("\u010a")
            or (prev_tok.startswith("<|") and prev_tok.endswith("|>"))
        )

        if is_word_start:
            if cur_start >= 0:
                words.append(cur_word)
                word_to_bpe_start.append(cur_start)
            cur_word = tok if is_special else tok.lstrip("\u0120\u2581")
            cur_start = i
        else:
            cur_word += tok

    if cur_start >= 0:
        words.append(cur_word)
        word_to_bpe_start.append(cur_start)

    return words, word_to_bpe_start


def _build_bpe_char_end_offsets(tokenizer, bpe_ids: List[int]) -> List[int]:
    """Cumulative exclusive char end offset of each BPE token in ``decode(bpe_ids)``."""
    ends: List[int] = []
    pos = 0
    for tid in bpe_ids:
        piece = tokenizer.decode([tid], skip_special_tokens=False)
        pos += len(piece)
        ends.append(pos)
    return ends


def _char_pos_to_bpe_index(char_pos: int, char_ends: List[int]) -> int:
    """Character offset -> index of the BPE token covering it."""
    if char_pos < 0 or not char_ends:
        return -1
    for i, end in enumerate(char_ends):
        if char_pos < end:
            return i
    return len(char_ends) - 1


def _bpe_repeat_anchor_fallback(valid_len_bpe: int) -> Tuple[bool, int, int, bool]:
    """Safe anchor when t1/t2 cannot be mapped: keep has_repeat, t1=last token, t2=-1.

    Calibration then covers only the last step, while monitoring still counts the
    response as has_repeat.
    """
    if valid_len_bpe <= 0:
        return False, -1, -1, False
    return True, valid_len_bpe - 1, -1, True


def _map_detection_result_to_bpe(
    detection: Tuple[bool, int, int, str],
    word_to_bpe_start: List[int],
    char_ends: List[int],
    valid_len_bpe: int,
) -> Tuple[bool, int, int, bool]:
    """Map the pipeline output (with source) to BPE-level ``(has_repeat, t1, t2, used_fallback)``."""
    has_rep, t1, t2, source = detection
    if not has_rep:
        return False, -1, -1, False

    if source in ("word", "spam_word"):
        return _map_word_indices_to_bpe(
            (has_rep, t1, t2), word_to_bpe_start, valid_len_bpe,
        )

    if source in ("char", "boxed", "spam_concat"):
        if not char_ends:
            return _bpe_repeat_anchor_fallback(valid_len_bpe)
        t1_bpe = _char_pos_to_bpe_index(t1, char_ends)
        if t1_bpe < 0 or t1_bpe >= valid_len_bpe:
            return _bpe_repeat_anchor_fallback(valid_len_bpe)
        if t2 < 0:
            t2_bpe = -1
        else:
            t2_bpe = _char_pos_to_bpe_index(t2, char_ends)
            if t2_bpe < 0:
                t2_bpe = -1
            elif t2_bpe <= t1_bpe:
                t2_bpe = -1
        return True, t1_bpe, t2_bpe, False

    return _bpe_repeat_anchor_fallback(valid_len_bpe)


def _map_word_indices_to_bpe(
    word_result: Tuple[bool, int, int],
    word_to_bpe_start: List[int],
    valid_len_bpe: int,
) -> Tuple[bool, int, int, bool]:
    """Map (has_rep, t1_word, t2_word) to BPE-level indices.

    - t1_word -> word_to_bpe_start[t1_word]
    - t2_word == -1         -> -1  (handled downstream by _normalize_repeat_results)
    - t2_word == len(words) -> valid_len_bpe  (defensive; should not occur)
    - otherwise             -> word_to_bpe_start[t2_word]
    - t1_word out of range  -> keep has_repeat, fall back to the last-token anchor
    """
    has_rep, t1_word, t2_word = word_result
    if not has_rep:
        return False, -1, -1, False

    n_words = len(word_to_bpe_start)
    if t1_word < 0 or t1_word >= n_words:
        return _bpe_repeat_anchor_fallback(valid_len_bpe)

    t1_bpe = word_to_bpe_start[t1_word]
    if t1_bpe < 0 or t1_bpe >= valid_len_bpe:
        return _bpe_repeat_anchor_fallback(valid_len_bpe)

    if t2_word < 0:
        t2_bpe = -1
    elif t2_word >= n_words:
        t2_bpe = valid_len_bpe
    else:
        t2_bpe = word_to_bpe_start[t2_word]
    return True, t1_bpe, t2_bpe, False


# --------------------------------------------------------------------
# (1b) Async detection on Ray
# --------------------------------------------------------------------

# Per-process tokenizer cache for the Ray tasks.
_TOKENIZER_CACHE: dict = {}


def _get_cached_tokenizer(tokenizer_path: str):
    if tokenizer_path not in _TOKENIZER_CACHE:
        from transformers import AutoTokenizer
        _TOKENIZER_CACHE[tokenizer_path] = AutoTokenizer.from_pretrained(
            tokenizer_path, trust_remote_code=True,
        )
    return _TOKENIZER_CACHE[tokenizer_path]


@ray.remote
def detect_repeat_chunk_remote(
    bpe_ids_chunk: List[List[int]],
    tokenizer_path: str,
    detection_kwargs: dict,
    collect_stage_hits: bool = False,
    char_max_chars: int = 0,
) -> Tuple[List[RepeatDetectRecord], int]:
    """Run repeat detection (four-level fallback) over a chunk of BPE id sequences.

    Per response:
      1. BPE ids -> BPE subword strings (convert_ids_to_tokens)
      2. subwords -> word sequence + word->BPE reverse map
      3. tokenizer.decode -> text (for the char / boxed / spam fallbacks)
      4. detect_malignant_repeat_pipeline -> (has_rep, t1, t2, source)
      5. map back to BPE indices according to source

    Args:
        collect_stage_hits: also run every level independently to record which ones
            would fire (slow; diagnosis only). When False, hit_* records which level
            actually fired, i.e. hit_x == (source == x).
        char_max_chars: skip the char-level fallback for texts longer than this
            (0 = no cap). Only affects responses the word level already missed.

    Returns:
        (records, n_t1_mapping_fallback) -- the latter counts responses whose t1
        could not be mapped and fell back to the last-token anchor.
    """
    tokenizer = _get_cached_tokenizer(tokenizer_path)

    results: List[RepeatDetectRecord] = []
    n_t1_mapping_fallback = 0
    for bpe_ids in bpe_ids_chunk:
        rec = RepeatDetectRecord()
        valid_len_bpe = len(bpe_ids)
        rec.num_bpe = valid_len_bpe
        if valid_len_bpe == 0:
            results.append(rec)
            continue
        bpe_tokens = tokenizer.convert_ids_to_tokens(bpe_ids)
        bpe_tokens = [t if t is not None else "" for t in bpe_tokens]
        words, word_to_bpe_start = _build_word_mapping_from_bpe_tokens(bpe_tokens)
        rec.num_words = len(words)
        text = tokenizer.decode(bpe_ids, skip_special_tokens=False)
        char_ends = _build_bpe_char_end_offsets(tokenizer, bpe_ids)

        kwargs = detection_kwargs
        if char_max_chars > 0 and len(text) > char_max_chars:
            kwargs = {**detection_kwargs, "char_fallback": False}

        if collect_stage_hits:
            stage_hits = compute_repeat_stage_hits(words, text, **kwargs)
            rec.hit_word = stage_hits["word"]
            rec.hit_char = stage_hits["char"]
            rec.hit_boxed = stage_hits["boxed"]
            rec.hit_spam_word = stage_hits["spam_word"]
            rec.hit_spam_concat = stage_hits["spam_concat"]

        boxed_diag = compute_boxed_diagnostics(
            text,
            boxed_threshold=kwargs.get("boxed_threshold", 5),
            boxed_dominant_ratio=kwargs.get("boxed_dominant_ratio", 0.75),
            boxed_m_index=kwargs.get("boxed_m_index", 4),
            boxed_tail_chars=kwargs.get("boxed_tail_chars", 0),
        )
        rec.boxed_count = boxed_diag["boxed_count"]
        rec.boxed_would_detect = boxed_diag["repeat_det_boxed_would_detect"]
        rec.dominant_boxed_ratio = boxed_diag["repeat_det_dominant_boxed_ratio"]

        detection = detect_malignant_repeat_pipeline(words, text, **kwargs)
        has_rep_raw, t1_raw, t2_raw, source = detection
        rec.source = source
        rec.t1_raw = t1_raw
        rec.t2_raw = t2_raw
        t1_space, t2_space = _raw_coord_spaces(source, t2_raw)
        rec.t1_raw_space = t1_space
        rec.t2_raw_space = t2_space

        if not collect_stage_hits:
            rec.hit_word = source == "word"
            rec.hit_char = source == "char"
            rec.hit_boxed = source == "boxed"
            rec.hit_spam_word = source == "spam_word"
            rec.hit_spam_concat = source == "spam_concat"

        has_rep, t1_bpe, t2_bpe, used_fallback = _map_detection_result_to_bpe(
            detection, word_to_bpe_start, char_ends, valid_len_bpe,
        )
        rec.has_repeat = has_rep
        rec.t1_bpe = t1_bpe
        rec.t2_bpe = t2_bpe
        rec.t1_mapping_fallback = used_fallback
        if used_fallback:
            n_t1_mapping_fallback += 1
        results.append(rec)
    return results, n_t1_mapping_fallback


def launch_repeat_detection_async(
    responses: torch.Tensor,
    response_mask: torch.Tensor,
    tokenizer_path: str,
    num_workers: int,
    detection_kwargs: dict,
    collect_stage_hits: bool = False,
    char_max_chars: int = 0,
) -> Tuple[list, List[int]]:
    """Slice out each response's valid tokens and dispatch them to Ray in chunks.

    Detection runs at word level inside the remote worker (BPE -> words + decoded
    text, four-level fallback), then maps t_repeat_1 / t_repeat_2 back to BPE indices.

    Must be called AFTER ``_balance_batch``: that reorders the batch, and the
    returned records are positional.

    Args:
        responses:        (bs, response_len) rollout BPE token ids
        response_mask:    (bs, response_len) valid response token mask
        tokenizer_path:   policy tokenizer path (used for BPE -> word in the worker)
        num_workers:      number of parallel chunks
        detection_kwargs: forwarded to detect_malignant_repeat_pipeline
                          (see build_repeat_detection_kwargs)

    Returns:
        futures:       List[ray.ObjectRef]
        valid_lengths: valid BPE length per response (used to bound t1/t2 later)
    """
    bs = responses.shape[0]
    valid_lengths = response_mask.sum(dim=-1).long().cpu().tolist()

    responses_cpu = responses.cpu().tolist()
    bpe_ids_list = [responses_cpu[i][: valid_lengths[i]] for i in range(bs)]

    if num_workers <= 0:
        num_workers = 1
    chunk_size = max(1, (bs + num_workers - 1) // num_workers)
    futures = [
        detect_repeat_chunk_remote.remote(
            bpe_ids_list[s:s + chunk_size], tokenizer_path, detection_kwargs,
            collect_stage_hits, char_max_chars,
        )
        for s in range(0, bs, chunk_size)
    ]
    return futures, valid_lengths


def gather_repeat_detection(futures: list) -> Tuple[List[RepeatDetectRecord], int]:
    """Block on all chunks and flatten to one ``RepeatDetectRecord`` per response.

    Returns:
        (repeat_records, n_t1_mapping_fallback)
    """
    chunks = ray.get(futures)
    flat: List[RepeatDetectRecord] = []
    n_t1_mapping_fallback = 0
    for c in chunks:
        chunk_results, n_fb = c
        flat.extend(chunk_results)
        n_t1_mapping_fallback += n_fb
    return flat, n_t1_mapping_fallback


# --------------------------------------------------------------------
# (2) Normalising the raw detection output
# --------------------------------------------------------------------

def _clamp_t1_to_valid_range(t1: int, valid_len: int) -> int:
    """Clamp ``t_repeat_1`` into ``[0, valid_len - 1]`` (when valid_len > 0)."""
    if valid_len <= 0:
        return -1
    return min(max(int(t1), 0), valid_len - 1)


def _normalize_repeat_results(
    repeat_results: List[Tuple[bool, int, int]],
    valid_lengths: List[int],
) -> List[Tuple[bool, int, int]]:
    """Resolve the ``t_repeat_2 == -1`` boundary case (degenerates to valid_len)."""
    normalized: List[Tuple[bool, int, int]] = []
    for (has_rep, t1, t2), vlen in zip(repeat_results, valid_lengths):
        if not has_rep:
            normalized.append((False, -1, -1))
            continue
        # the detector guarantees has_rep=True -> t1 in [0, n)
        # t2 = -1 degenerates to valid_len (all of [t1, vlen) counts as the first period)
        if t2 < 0:
            t2_eff = vlen
        else:
            t2_eff = min(t2, vlen)
        t1_eff = _clamp_t1_to_valid_range(t1, vlen)
        if t1_eff < 0:
            normalized.append((False, -1, -1))
            continue
        normalized.append((True, t1_eff, t2_eff))
    return normalized


def compute_response_level_flags(
    repeat_results: List[Tuple[bool, int, int]],
    valid_lengths: List[int],
    max_response_length: int,
) -> dict:
    """Turn detection results + valid_lengths into per-sample bool tensors.

    Args:
        repeat_results: per response (has_rep, t1_bpe, t2_bpe); t2 = -1 means the
                        detector found only one unit and no second boundary.
        valid_lengths:  valid BPE length per response (response_mask.sum(-1)).
        max_response_length: the configured cap, used to flag truncation.

    Returns:
        dict of five (bs,) bool tensors plus the int scalar bs:
            has_repeat            -- first return value of the detector
            t2_is_minus_one       -- has_rep AND t2 == -1
            truncated             -- valid_len >= max_response_length
            truncated_no_repeat   -- truncated AND (NOT has_rep OR t2 == -1)
            has_repeat_effective  -- has_rep AND NOT truncated_no_repeat
    """
    bs = len(repeat_results)
    assert len(valid_lengths) == bs, (
        f"len mismatch: repeat_results={bs}, valid_lengths={len(valid_lengths)}"
    )

    has_repeat = torch.zeros(bs, dtype=torch.bool)
    t2_is_minus_one = torch.zeros(bs, dtype=torch.bool)
    truncated = torch.zeros(bs, dtype=torch.bool)
    for i, ((has_rep, _t1, t2), vlen) in enumerate(zip(repeat_results, valid_lengths)):
        has_repeat[i] = bool(has_rep)
        t2_is_minus_one[i] = bool(has_rep) and (t2 < 0)
        truncated[i] = (int(vlen) >= int(max_response_length))

    truncated_no_repeat = truncated & ((~has_repeat) | t2_is_minus_one)
    has_repeat_effective = has_repeat & (~truncated_no_repeat)

    return {
        "has_repeat": has_repeat,
        "t2_is_minus_one": t2_is_minus_one,
        "truncated": truncated,
        "truncated_no_repeat": truncated_no_repeat,
        "has_repeat_effective": has_repeat_effective,
        "bs": bs,
    }


# --------------------------------------------------------------------
# (3) Monitoring metrics
# --------------------------------------------------------------------

_M = "repeat"   # wandb metric prefix


def compute_response_level_flags_metrics(flags: dict) -> dict:
    """Response-level ratios derived from the detection flags (denominator = bs).

      repeat/response_has_repeat_ratio : share of responses flagged
    """
    bs = max(flags["bs"], 1)
    has_repeat = flags["has_repeat"]
    return {
        f"{_M}/response_has_repeat_ratio": float(has_repeat.sum().item()) / bs,
    }



def compute_truncation_repeat_cross_metrics(
    flags: dict,
    acc: Optional[np.ndarray] = None,
) -> dict:
    """Truncated x has_repeat cross-tabulation.

      repeat/truncated_and_repeat_ratio : (truncated & has_repeat) / bs
      repeat/truncated_not_repeat_ratio : (truncated & ~has_repeat) / bs
      repeat/repeat_ratio_in_truncated  : among truncated, share flagged
      repeat/truncated_ratio_in_repeat  : among flagged, share truncated

    """
    bs = max(flags["bs"], 1)
    has_repeat = flags["has_repeat"]
    truncated = flags["truncated"]

    truncated_and_repeat = truncated & has_repeat
    truncated_not_repeat = truncated & (~has_repeat)
    n_truncated = int(truncated.sum().item())
    n_has_repeat = int(has_repeat.sum().item())

    metrics: dict = {
        f"{_M}/truncated_and_repeat_ratio": float(truncated_and_repeat.sum().item()) / bs,
        f"{_M}/truncated_not_repeat_ratio": float(truncated_not_repeat.sum().item()) / bs,
    }
    if n_truncated > 0:
        metrics[f"{_M}/repeat_ratio_in_truncated"] = (
            float(truncated_and_repeat.sum().item()) / n_truncated
        )
    if n_has_repeat > 0:
        metrics[f"{_M}/truncated_ratio_in_repeat"] = (
            float(truncated_and_repeat.sum().item()) / n_has_repeat
        )
    return metrics


def compute_repeat_monitoring_metrics(
    repeat_results: List[Tuple[bool, int, int]],
    valid_lengths: List[int],
    acc: Optional[np.ndarray] = None,
    sources: Optional[List[str]] = None,
) -> dict:
    """Always-on monitoring metrics for the detection result.

    Overall:
      has_repeat_ratio            : share of responses flagged

    Acc x repeat (only when acc is given):
      correct_ratio_in_has_repeat : among flagged, share correct
      wrong_ratio_in_has_repeat   : among flagged, share wrong
      has_repeat_ratio_in_correct : among correct, share flagged
      has_repeat_ratio_in_wrong   : among wrong, share flagged

    t1 / t2 positions:
      t1_position_pct_mean        : mean(t1/vlen) among flagged
      t2_position_pct_mean        : mean(t2/vlen) among flagged with t2 != -1

    Token counts:
      period1_token_ratio         : sum([t1,t2) tokens) / total valid tokens
      after_t2_token_ratio        : sum([t2,vlen) tokens) / total valid tokens
      period1_length_mean         : mean(t2-t1) among flagged with t2 != -1

    Which level fired (only when sources is given):
      source_frac/<level>         : share of flagged responses caught by each level

    Args:
        repeat_results: per response (has_rep, t1_bpe, t2_bpe).
        valid_lengths:  valid BPE length per response.
        acc:            (bs,) float array; 1.0 = correct. None skips the cross-tab.
        sources:        per response, which level fired ("word"/"char"/... /"none").
    """
    bs = len(repeat_results)
    norm_results = _normalize_repeat_results(repeat_results, valid_lengths)
    metrics: dict = {}

    has_rep_arr_raw = np.array([r[0] for r in repeat_results], dtype=bool)
    metrics[f"{_M}/has_repeat_ratio"] = float(has_rep_arr_raw.mean()) if bs > 0 else 0.0

    if acc is not None and len(acc) == bs:
        has_rep_arr = has_rep_arr_raw
        correct_arr = np.asarray(acc, dtype=float) >= 1.0
        wrong_arr = ~correct_arr

        n_rep = int(has_rep_arr.sum())
        n_correct = int(correct_arr.sum())
        n_wrong = int(wrong_arr.sum())

        if n_rep > 0:
            metrics[f"{_M}/correct_ratio_in_has_repeat"] = float(
                (has_rep_arr & correct_arr).sum() / n_rep
            )
            metrics[f"{_M}/wrong_ratio_in_has_repeat"] = float(
                (has_rep_arr & wrong_arr).sum() / n_rep
            )
        if n_correct > 0:
            metrics[f"{_M}/has_repeat_ratio_in_correct"] = float(
                (has_rep_arr & correct_arr).sum() / n_correct
            )
        if n_wrong > 0:
            metrics[f"{_M}/has_repeat_ratio_in_wrong"] = float(
                (has_rep_arr & wrong_arr).sum() / n_wrong
            )

    # Positions use the normalized results; the raw t2 distinguishes t2 == -1 from
    # t2 == vlen.
    total_valid = max(sum(valid_lengths), 1)
    t1_positions_pct: List[float] = []
    t2_positions_pct: List[float] = []
    period1_lengths: List[int] = []
    period1_tokens = 0
    after_t2_tokens = 0

    for i, ((raw_has_rep, _raw_t1, raw_t2), (has_rep, t1, t2)) in enumerate(
        zip(repeat_results, norm_results)
    ):
        if not has_rep:
            continue
        vlen = valid_lengths[i]
        original_t2_is_minus_one = raw_has_rep and (raw_t2 < 0)

        if vlen > 0:
            t1_positions_pct.append(t1 / vlen)

        if not original_t2_is_minus_one and vlen > 0:
            t2_positions_pct.append(t2 / vlen)
            period1_lengths.append(max(0, t2 - t1))

        period1_tokens += max(0, t2 - t1)
        after_t2_tokens += max(0, vlen - t2)

    if t1_positions_pct:
        metrics[f"{_M}/t1_position_pct_mean"] = float(np.mean(t1_positions_pct))
    if t2_positions_pct:
        metrics[f"{_M}/t2_position_pct_mean"] = float(np.mean(t2_positions_pct))

    metrics[f"{_M}/period1_token_ratio"] = period1_tokens / total_valid
    metrics[f"{_M}/after_t2_token_ratio"] = after_t2_tokens / total_valid
    if period1_lengths:
        metrics[f"{_M}/period1_length_mean"] = float(np.mean(period1_lengths))

    if sources is not None and len(sources) == bs:
        n_rep = max(int(has_rep_arr_raw.sum()), 1)
        for level in ("word", "char", "boxed", "spam_word", "spam_concat"):
            n_level = sum(
                1 for s, hr in zip(sources, has_rep_arr_raw) if hr and s == level
            )
            metrics[f"{_M}/source_frac/{level}"] = n_level / n_rep

    return metrics


# --------------------------------------------------------------------
# (4) Per-sample dump fields
# --------------------------------------------------------------------

def build_repeat_det_dump_fields(
    records: List[RepeatDetectRecord],
    flags: dict,
    valid_lengths: List[int],
    max_response_length: int,
) -> Dict[str, List]:
    """Build per-sample repeat columns for the rollout / parquet dump."""
    bs = len(records)
    assert len(valid_lengths) == bs

    tuples = repeat_records_to_tuples(records)
    norm_results = _normalize_repeat_results(tuples, valid_lengths)

    out: Dict[str, List] = {
        "has_repeat": [],
        "has_repeat_source": [],
        "has_repeat_t1_bpe": [],
        "has_repeat_t2_bpe": [],
        "repeat_det_valid_bpe_len": [],
        "repeat_det_num_words": [],
        "repeat_det_t1_raw": [],
        "repeat_det_t1_raw_space": [],
        "repeat_det_t2_raw": [],
        "repeat_det_t2_raw_space": [],
        "repeat_det_is_truncated": [],
        "repeat_det_schema_version": [],
        "repeat_det_t2_is_minus_one": [],
        "repeat_det_has_repeat_effective": [],
        "repeat_det_truncated_no_repeat": [],
        "repeat_det_t1_bpe_clamped": [],
        "repeat_det_t1_mapping_fallback": [],
        "repeat_det_t2_bpe_norm": [],
        "repeat_det_calib_mask_from": [],
        "repeat_det_repeat_span_bpe": [],
        "repeat_det_hit_word": [],
        "repeat_det_hit_char": [],
        "repeat_det_hit_boxed": [],
        "repeat_det_hit_spam_word": [],
        "repeat_det_hit_spam_concat": [],
        "repeat_det_boxed_count": [],
        "repeat_det_boxed_would_detect": [],
        "repeat_det_dominant_boxed_ratio": [],
    }

    for i, (rec, vlen) in enumerate(zip(records, valid_lengths)):
        has_rep, t1_n, t2_n = norm_results[i]
        vlen = int(vlen)

        out["has_repeat"].append(int(rec.has_repeat))
        out["has_repeat_source"].append(rec.source)
        out["has_repeat_t1_bpe"].append(rec.t1_bpe)
        out["has_repeat_t2_bpe"].append(rec.t2_bpe)
        out["repeat_det_valid_bpe_len"].append(vlen)
        out["repeat_det_num_words"].append(rec.num_words)
        out["repeat_det_t1_raw"].append(rec.t1_raw)
        out["repeat_det_t1_raw_space"].append(rec.t1_raw_space)
        out["repeat_det_t2_raw"].append(rec.t2_raw)
        out["repeat_det_t2_raw_space"].append(rec.t2_raw_space)

        out["repeat_det_is_truncated"].append(int(flags["truncated"][i].item()))
        out["repeat_det_schema_version"].append(REPEAT_DET_SCHEMA_VERSION)

        out["repeat_det_t2_is_minus_one"].append(int(flags["t2_is_minus_one"][i].item()))
        out["repeat_det_has_repeat_effective"].append(
            int(flags["has_repeat_effective"][i].item())
        )
        out["repeat_det_truncated_no_repeat"].append(
            int(flags["truncated_no_repeat"][i].item())
        )

        t1_clamped = 0
        if rec.has_repeat and vlen > 0:
            t1_safe = _clamp_t1_to_valid_range(rec.t1_bpe, vlen)
            t1_clamped = int(t1_safe != rec.t1_bpe)
        out["repeat_det_t1_bpe_clamped"].append(t1_clamped)
        out["repeat_det_t1_mapping_fallback"].append(int(rec.t1_mapping_fallback))

        out["repeat_det_t2_bpe_norm"].append(t2_n if has_rep else -1)
        calib_from = t2_n if (has_rep and t2_n < vlen) else -1
        out["repeat_det_calib_mask_from"].append(calib_from)
        span = max(0, t2_n - t1_n) if has_rep else 0
        out["repeat_det_repeat_span_bpe"].append(span)

        out["repeat_det_hit_word"].append(int(rec.hit_word))
        out["repeat_det_hit_char"].append(int(rec.hit_char))
        out["repeat_det_hit_boxed"].append(int(rec.hit_boxed))
        out["repeat_det_hit_spam_word"].append(int(rec.hit_spam_word))
        out["repeat_det_hit_spam_concat"].append(int(rec.hit_spam_concat))

        out["repeat_det_boxed_count"].append(rec.boxed_count)
        out["repeat_det_boxed_would_detect"].append(rec.boxed_would_detect)
        out["repeat_det_dominant_boxed_ratio"].append(rec.dominant_boxed_ratio)

    return out
