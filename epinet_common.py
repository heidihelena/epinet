"""Small shared helpers used across the EpiNet modules.

Kept dependency-light (pandas only) so any module can import it without pulling
in scikit-learn or matplotlib.
"""

from __future__ import annotations

import pandas as pd

# Values that count as "no label" once stringified and lowercased.
_BLANK_TOKENS = {"", "nan", "none"}


def is_blank_value(value: object) -> bool:
    """True if a single label value is missing/blank (unlabeled scaffold)."""
    if pd.isna(value):
        return True
    return str(value).strip().lower() in _BLANK_TOKENS


def blank_label_mask(values: pd.Series) -> pd.Series:
    """Boolean Series, True where the label is blank/NaN.

    Robust to the modern pandas ``str``/``string`` dtypes (where ``astype(str)``
    leaves NaN as a float), which is the bug this helper exists to prevent from
    being re-introduced in each module.
    """
    mask = values.isna()
    if not pd.api.types.is_numeric_dtype(values):
        text = values.astype("string").fillna("").str.strip().str.lower()
        mask = mask | text.isin(list(_BLANK_TOKENS))
    return mask


def labeled_mask(values: pd.Series) -> pd.Series:
    """Boolean Series, True where the label is present (complement of blank)."""
    return ~blank_label_mask(values)
