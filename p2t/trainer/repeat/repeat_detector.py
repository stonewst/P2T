"""Malignant-repeat detector (stdlib only).

Main entry points::

    detect_malignant_repeat(tokens, ...) -> (has_repeat, t_repeat_1, t_repeat_2)
    detect_malignant_repeat_pipeline(words, text, ...) -> (has_repeat, t1, t2, source)

``detect_malignant_repeat`` works on any sequence of hashable elements (token ids, words
or characters) using block hashing:

    has_repeat : bool -- whether a malignant repeat was detected
    t_repeat_1 : int  -- index of the first token of the second occurrence of the
                         repeating unit; -1 when nothing is detected
    t_repeat_2 : int  -- index of the first token of the third occurrence
                         (= t_repeat_1 + period); -1 when there is none

``detect_malignant_repeat_pipeline`` applies four levels in order
(word -> char -> boxed -> spam) and reports which one fired.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any, List, Optional, Tuple


# --------------------------------------------------------------------
# Boundary heuristics (only active when elements are strings; token-id sequences skip this)
# --------------------------------------------------------------------

_STARTER_TOKENS = {
    "---", "***", "___",
    "###", "##", "#",
    "1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.",
    "-", "*", "•",
    "**Final", "**Conclusion:**", "**Answer:**",
    "**总结**", "**结论**", "**反思**",
}


def _boundary_score(tokens: List[Any], pos: int) -> float:
    """Naturalness score for a segment boundary before position pos (string elements only).

    Words built from BPE tokens can contain paragraph markers in the middle (e.g.
    '$\\boxed{...}$.\n\n---\n\n###'), so the rules on `prev` also search inside the word.
    """
    n = len(tokens)
    if pos <= 0 or pos >= n:
        return 0.0
    prev, cur = tokens[pos - 1], tokens[pos]
    if not isinstance(prev, str) or not isinstance(cur, str):
        return 0.0

    score = 0.0

    # ============ cur: segment-start features ============
    if cur in _STARTER_TOKENS:
        score += 5.0
    if cur.startswith(("###", "##", "#")):
        score += 2.0
    # horizontal rule as segment start (BPE may produce '---\n\n', not exactly '---')
    if cur.startswith(("---", "***", "___")):
        score += 4.0
    if cur.startswith("**") and not cur.endswith("**"):
        score += 1.0

    # ============ prev: segment-end features (word-trailing punctuation) ============
    if "\\boxed" in prev and (prev.endswith("$") or prev.endswith("}$")
                              or prev.endswith(".") or prev.endswith("。")):
        score += 6.0
    elif prev.endswith("}$"):
        score += 4.0
    elif prev.endswith("$"):
        score += 2.0
    if prev.endswith(":**") or prev.endswith(".**"):
        score += 3.0
    if prev.endswith(("。", ".", "!", "?", "！", "？")):
        score += 2.0

    # ============ prev: segment-end features (paragraph marker inside word) ============
    # Newlines appear as 'Ċ' (U+010A) in BPE token strings and as '\n' in plain text.
    NL = ("\n\n", "ĊĊ")
    HR = ("---", "***", "___")
    HEADER = ("###", "##")

    if "\\boxed" in prev and any(m in prev for m in NL + HR + HEADER):
        score += 5.0   # contains \boxed followed by paragraph break/newline/header -> end of final-answer block
    # markdown horizontal rule
    if any((nl + hr) in prev for nl in NL for hr in HR):
        score += 3.0
    # markdown header
    if any((nl + h) in prev for nl in NL for h in HEADER) or prev.endswith(HEADER):
        score += 2.0
    # paragraph break (blank line)
    if any(nl in prev for nl in NL):
        score += 1.0
    return score


# --------------------------------------------------------------------
# Internal helpers
# --------------------------------------------------------------------

def _reduce_period(tokens: List[Any], anchor: int, period: int) -> int:
    for q in range(2, period + 1):
        if period % q != 0:
            continue
        ok = True
        for k in range(period):
            i = anchor - k
            j = i - q
            if j < 0:
                break
            if tokens[i] != tokens[j]:
                ok = False
                break
        if ok:
            return q
    return period


def _backward_walk(tokens: List[Any], anchor: int, period: int,
                   max_consecutive_mismatch: int = 3) -> int:
    p = anchor
    consec_mis = 0
    last_match = anchor + 1
    while p >= period:
        if tokens[p] == tokens[p - period]:
            consec_mis = 0
            last_match = p
        else:
            consec_mis += 1
            if consec_mis > max_consecutive_mismatch:
                break
        p -= 1
    return last_match


def _forward_extend(tokens: List[Any], anchor: int, period: int,
                    max_consecutive_mismatch: int = 3) -> int:
    n = len(tokens)
    p = anchor
    consec_mis = 0
    last_match = anchor
    while p + 1 < n:
        p += 1
        if p - period < 0:
            break
        if tokens[p] == tokens[p - period]:
            consec_mis = 0
            last_match = p
        else:
            consec_mis += 1
            if consec_mis > max_consecutive_mismatch:
                break
    return last_match


def _refine_t_repeat(tokens: List[Any], t_repeat: int, period: int) -> int:
    min_match = min(period, 8)
    n = len(tokens)
    upper = min(t_repeat + period, n - min_match + 1)
    for start in range(t_repeat, upper):
        ok = True
        for k in range(min_match):
            i = start + k
            if i >= n or i - period < 0:
                ok = False
                break
            if tokens[i] != tokens[i - period]:
                ok = False
                break
        if ok:
            return start
    return t_repeat


def _select_best_phase(tokens: List[Any], first_unit_start: int,
                       period: int, margin: float = 2.0) -> int:
    n = len(tokens)
    best_shift = 0
    best_score = _boundary_score(tokens, first_unit_start)
    for s in range(1, period):
        pos = first_unit_start + s
        if pos + period >= n:
            break
        sc = _boundary_score(tokens, pos)
        if sc > best_score + margin - 1e-9:
            best_score = sc
            best_shift = s
    return first_unit_start + best_shift


# --------------------------------------------------------------------
# Main API
# --------------------------------------------------------------------

def detect_malignant_repeat(
    tokens: List[Any],
    block_size: int = 12,
    min_repeats: int = 3,
    max_consecutive_mismatch: int = 3,
    min_period: int = 4,
    min_unique_tokens_in_unit: int = 3,
) -> Tuple[bool, int, int]:
    """Detect malignant repeats and return the start token indices of the first two repeat rounds.

    Returns
    -------
    (has_repeat, t_repeat_1, t_repeat_2)
        has_repeat : whether a malignant repeat is present
        t_repeat_1 : index of the first token of the 1st repeat round (-1 if none)
        t_repeat_2 : index of the first token of the 2nd repeat round (-1 if none)
    """
    n = len(tokens)
    if n < block_size + (min_repeats - 1) * min_period:
        return False, -1, -1

    hash_pos: dict = defaultdict(list)
    for i in range(n - block_size + 1):
        h = hash(tuple(tokens[i:i + block_size]))
        hash_pos[h].append(i)

    best_t_repeat: Optional[int] = None
    best_period: int = 0
    best_first_unit_start: int = 0
    seen_sigs: set = set()

    for positions in hash_pos.values():
        if len(positions) < min_repeats:
            continue
        for i in range(len(positions) - min_repeats + 1):
            sub = positions[i:i + min_repeats]
            gaps = [sub[j + 1] - sub[j] for j in range(len(sub) - 1)]
            if len(set(gaps)) != 1:
                continue
            period = gaps[0]
            if period < min_period:
                continue

            block0 = tokens[sub[0]:sub[0] + block_size]
            if not all(tokens[p:p + block_size] == block0 for p in sub[1:]):
                continue

            anchor = sub[-1] + block_size - 1
            period = _reduce_period(tokens, anchor, period)

            unit = tokens[sub[0]:sub[0] + period]
            if len(unit) < period:
                continue
            if len(set(unit)) < min_unique_tokens_in_unit:
                continue

            sig = (sub[0], period)
            if sig in seen_sigs:
                continue
            seen_sigs.add(sig)

            periodic_end = _forward_extend(tokens, anchor, period,
                                           max_consecutive_mismatch)

            t_repeat = _backward_walk(tokens, periodic_end, period,
                                      max_consecutive_mismatch)
            if t_repeat >= n:
                continue
            if t_repeat < period:
                t_repeat = period

            t_repeat = _refine_t_repeat(tokens, t_repeat, period)
            first_unit_start = max(0, t_repeat - period)

            new_first = _select_best_phase(tokens, first_unit_start, period)
            if new_first > first_unit_start:
                shift = new_first - first_unit_start
                first_unit_start = new_first
                t_repeat += shift

            if best_t_repeat is None or t_repeat < best_t_repeat:
                best_t_repeat = t_repeat
                best_period = period
                best_first_unit_start = first_unit_start

    if best_t_repeat is None:
        return False, -1, -1

    t_repeat_1 = best_t_repeat
    t_repeat_2 = t_repeat_1 + best_period
    if t_repeat_2 >= n:
        t_repeat_2 = -1
    return True, t_repeat_1, t_repeat_2


# ====================================================================
# Fallback detectors
# ====================================================================

_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")


def _position_after_boxed_tail(text: str, pos: int, extra_tail: int = 0) -> int:
    n = len(text)
    while pos < n:
        ch = text[pos]
        if ch in "$.,;:!?\t\n\r ":
            pos += 1
            continue
        if ch == "*" and pos + 1 < n and text[pos + 1] == "*":
            pos += 2
            continue
        break
    if extra_tail > 0:
        pos = min(pos + extra_tail, n)
    return pos


def _detect_boxed_repeat(
    text: str,
    threshold: int,
    dominant_ratio: float,
    m_index: int,
    tail_chars: int,
) -> Tuple[bool, int, int]:
    matches = list(_BOXED_RE.finditer(text))
    n = len(matches)
    if n < threshold:
        return False, -1, -1

    contents = [m.group(1).strip() for m in matches]
    most_common_cnt = Counter(contents).most_common(1)[0][1]
    if most_common_cnt / n < dominant_ratio:
        return False, -1, -1

    m_idx_0 = max(0, min(m_index - 1, n - 1))
    t1_char = _position_after_boxed_tail(text, matches[m_idx_0].end(), extra_tail=tail_chars)

    m1_idx_0 = m_idx_0 + 1
    if m1_idx_0 < n:
        t2_char = _position_after_boxed_tail(
            text, matches[m1_idx_0].end(), extra_tail=tail_chars,
        )
    else:
        t2_char = t1_char

    return True, t1_char, t2_char


_SPAM_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "to", "of", "in", "and", "or",
    "for", "that", "this", "it", "we", "so", "if", "as", "at", "by", "be", "with",
    "from", "on", "not", "can", "will", "has", "have", "had", "but", "which",
})

_WORD_SPAN_RE = re.compile(r"\S+")


def _char_pos_to_word_index(text: str, char_pos: int) -> int:
    """Map a character offset in *text* to 0-based word index (``\\S+`` spans)."""
    if char_pos < 0:
        return -1
    idx = 0
    for i, m in enumerate(_WORD_SPAN_RE.finditer(text)):
        if m.start() <= char_pos:
            idx = i
        else:
            break
    return idx


def _detect_word_spam(
    words: List[str],
    min_count: int,
    min_ratio: float,
    min_run: int,
) -> Tuple[bool, int]:
    n = len(words)
    if n < min_count:
        return False, -1

    word, cnt = Counter(words).most_common(1)[0]
    if word.lower() in _SPAM_STOPWORDS:
        return False, -1
    if cnt < min_count or (cnt / n) < min_ratio:
        return False, -1

    best_start, best_run = 0, 0
    i = 0
    while i < n:
        if words[i] == word:
            j = i
            while j < n and words[j] == word:
                j += 1
            run_len = j - i
            if run_len > best_run:
                best_run = run_len
                best_start = i
            i = j
        else:
            i += 1

    if best_run < min_run:
        return False, -1
    return True, best_start


def _detect_concat_spam(
    text: str,
    min_unit_len: int,
    max_unit_len: int,
    min_repeats: int,
) -> Tuple[bool, int]:
    if min_repeats < 2:
        min_repeats = 2
    if len(text) < min_unit_len * min_repeats:
        return False, -1

    pat = re.compile(
        rf"(.{{{min_unit_len},{max_unit_len}}})\1{{{min_repeats - 1},}}"
    )
    m = pat.search(text)
    if not m:
        return False, -1

    unit = m.group(1)
    if len(set(unit)) <= 1 and len(unit) < 4:
        return False, -1
    return True, m.start()


def _detect_token_spam(
    text: str,
    words: List[str],
    word_min_count: int,
    word_min_ratio: float,
    word_min_run: int,
    concat_min_repeats: int,
    concat_min_unit_len: int,
    concat_max_unit_len: int,
) -> Tuple[bool, int]:
    has_w, t1_w = _detect_word_spam(
        words, word_min_count, word_min_ratio, word_min_run,
    )
    if has_w:
        return True, t1_w

    has_c, t1_c = _detect_concat_spam(
        text, concat_min_unit_len, concat_max_unit_len, concat_min_repeats,
    )
    if has_c:
        return True, _char_pos_to_word_index(text, t1_c)

    return False, -1


def detect_malignant_repeat_pipeline(
    words: List[str],
    text: str,
    *,
    block_size: int = 12,
    min_repeats: int = 3,
    max_consecutive_mismatch: int = 3,
    min_period: int = 2,
    min_unique_tokens_in_unit: int = 2,
    char_fallback: bool = True,
    char_block_size: int = 50,
    char_min_period: int = 15,
    char_min_unique: int = 3,
    boxed_fallback: bool = True,
    boxed_threshold: int = 5,
    boxed_dominant_ratio: float = 0.75,
    boxed_m_index: int = 4,
    boxed_tail_chars: int = 0,
    spam_fallback: bool = True,
    spam_word_min_count: int = 15,
    spam_word_min_ratio: float = 0.35,
    spam_word_min_run: int = 10,
    spam_concat_min_repeats: int = 15,
    spam_concat_min_unit_len: int = 3,
    spam_concat_max_unit_len: int = 40,
) -> Tuple[bool, int, int, str]:
    """Four-level fallback detection; returns ``(has_repeat, t1, t2, source)``.

    *text* is used for the char / boxed / spam fallbacks (typically
    ``tokenizer.decode(bpe_ids)``).

    The meaning of ``t1`` / ``t2`` depends on *source* (``repeat_detection`` maps them to BPE):
      - ``word`` / ``spam_word``:    index into the BPE-collapsed *words* list
      - ``char`` / ``boxed`` / ``spam_concat``: character offset in *text*
      - ``none``:                    both are -1 when nothing is detected
    """
    word_kwargs = dict(
        block_size=block_size,
        min_repeats=min_repeats,
        max_consecutive_mismatch=max_consecutive_mismatch,
        min_period=min_period,
        min_unique_tokens_in_unit=min_unique_tokens_in_unit,
    )

    if words:
        has_rep, t1, t2 = detect_malignant_repeat(words, **word_kwargs)
        if has_rep:
            return True, t1, t2, "word"

    if char_fallback and text:
        chars = list(text)
        has_rep, t1, t2 = detect_malignant_repeat(
            chars,
            block_size=char_block_size,
            min_repeats=min_repeats,
            max_consecutive_mismatch=max_consecutive_mismatch,
            min_period=char_min_period,
            min_unique_tokens_in_unit=char_min_unique,
        )
        if has_rep:
            return True, t1, t2, "char"

    if boxed_fallback and text:
        has_rep, t1_char, t2_char = _detect_boxed_repeat(
            text,
            threshold=boxed_threshold,
            dominant_ratio=boxed_dominant_ratio,
            m_index=boxed_m_index,
            tail_chars=boxed_tail_chars,
        )
        if has_rep:
            if t2_char <= t1_char:
                t2_char = -1
            return True, t1_char, t2_char, "boxed"

    if spam_fallback and text:
        word_list = words if words else text.split()
        has_w, t1_w = _detect_word_spam(
            word_list,
            spam_word_min_count,
            spam_word_min_ratio,
            spam_word_min_run,
        )
        if has_w:
            return True, t1_w, -1, "spam_word"

        has_c, t1_c = _detect_concat_spam(
            text,
            spam_concat_min_unit_len,
            spam_concat_max_unit_len,
            spam_concat_min_repeats,
        )
        if has_c:
            return True, t1_c, -1, "spam_concat"

    return False, -1, -1, "none"


def compute_repeat_stage_hits(
    words: List[str],
    text: str,
    *,
    block_size: int = 12,
    min_repeats: int = 3,
    max_consecutive_mismatch: int = 3,
    min_period: int = 2,
    min_unique_tokens_in_unit: int = 2,
    char_fallback: bool = True,
    char_block_size: int = 50,
    char_min_period: int = 15,
    char_min_unique: int = 3,
    boxed_fallback: bool = True,
    boxed_threshold: int = 5,
    boxed_dominant_ratio: float = 0.75,
    boxed_m_index: int = 4,
    boxed_tail_chars: int = 0,
    spam_fallback: bool = True,
    spam_word_min_count: int = 15,
    spam_word_min_ratio: float = 0.35,
    spam_word_min_run: int = 10,
    spam_concat_min_repeats: int = 15,
    spam_concat_min_unit_len: int = 3,
    spam_concat_max_unit_len: int = 40,
) -> dict:
    """Whether each fallback level would independently trigger (ignoring pipeline priority)."""
    word_kwargs = dict(
        block_size=block_size,
        min_repeats=min_repeats,
        max_consecutive_mismatch=max_consecutive_mismatch,
        min_period=min_period,
        min_unique_tokens_in_unit=min_unique_tokens_in_unit,
    )
    hit_word = False
    if words:
        hit_word, _, _ = detect_malignant_repeat(words, **word_kwargs)

    hit_char = False
    if char_fallback and text:
        hit_char, _, _ = detect_malignant_repeat(
            list(text),
            block_size=char_block_size,
            min_repeats=min_repeats,
            max_consecutive_mismatch=max_consecutive_mismatch,
            min_period=char_min_period,
            min_unique_tokens_in_unit=char_min_unique,
        )

    hit_boxed = False
    if boxed_fallback and text:
        hit_boxed, _, _ = _detect_boxed_repeat(
            text,
            threshold=boxed_threshold,
            dominant_ratio=boxed_dominant_ratio,
            m_index=boxed_m_index,
            tail_chars=boxed_tail_chars,
        )

    hit_spam_word = False
    hit_spam_concat = False
    if spam_fallback and text:
        word_list = words if words else text.split()
        hit_spam_word, _ = _detect_word_spam(
            word_list,
            spam_word_min_count,
            spam_word_min_ratio,
            spam_word_min_run,
        )
        hit_spam_concat, _ = _detect_concat_spam(
            text,
            spam_concat_min_unit_len,
            spam_concat_max_unit_len,
            spam_concat_min_repeats,
        )

    return {
        "word": hit_word,
        "char": hit_char,
        "boxed": hit_boxed,
        "spam_word": hit_spam_word,
        "spam_concat": hit_spam_concat,
    }


def compute_boxed_diagnostics(
    text: str,
    *,
    boxed_threshold: int = 5,
    boxed_dominant_ratio: float = 0.75,
    boxed_m_index: int = 4,
    boxed_tail_chars: int = 0,
) -> dict:
    """\\boxed{} count and whether the boxed fallback would trigger (for rollout logging / analysis)."""
    matches = list(_BOXED_RE.finditer(text))
    n = len(matches)
    if n == 0:
        return {
            "boxed_count": 0,
            "repeat_det_boxed_would_detect": 0,
            "repeat_det_dominant_boxed_ratio": 0.0,
        }
    contents = [m.group(1).strip() for m in matches]
    dominant_ratio = Counter(contents).most_common(1)[0][1] / n
    would_detect, _, _ = _detect_boxed_repeat(
        text,
        threshold=boxed_threshold,
        dominant_ratio=boxed_dominant_ratio,
        m_index=boxed_m_index,
        tail_chars=boxed_tail_chars,
    )
    return {
        "boxed_count": n,
        "repeat_det_boxed_would_detect": int(would_detect),
        "repeat_det_dominant_boxed_ratio": float(dominant_ratio),
    }
