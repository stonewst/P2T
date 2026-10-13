"""Which policy token ids carry the step delimiter ``\\n\\n``.

A boolean lookup table over the vocabulary, built once per tokenizer.
``p2t_adv_mask_mode`` uses it to zero the P2T advantage on separator tokens.
"""

import numpy as np

__all__ = ["separator_token_lut"]


def separator_token_lut(tokenizer, sep: str = "\n\n") -> np.ndarray:
    """Boolean lookup table over the vocabulary: does token id ``i`` carry ``sep``?

    Returns a bool array indexable by token id, sized to cover every id the tokenizer
    can produce (``len(tokenizer)`` includes added/special tokens a rollout can emit).

    A token matches if its text contains ``sep`` (not only if it ends with it).
    """
    n = max(len(tokenizer), int(getattr(tokenizer, "vocab_size", 0)))
    lut = np.zeros(n, dtype=bool)
    for i in range(n):
        piece = tokenizer.convert_ids_to_tokens(i)
        if piece is None:
            continue
        if sep in tokenizer.convert_tokens_to_string([piece]):
            lut[i] = True
    return lut
