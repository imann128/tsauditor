"""
tsauditor.adapters.timesfm
--------------------------
Format an audited, repaired series into the numpy array Google TimesFM expects.

TimesFM (and similar zero-shot forecasters) tokenize a clean, contiguous, finite
numeric context window. Real-world series rarely arrive that way — gaps, outliers
and stuck runs make tokenization fail. This adapter runs the tsauditor audit/fix
engine first, then hands back a plain ``float32`` array, so the messy-data step
and the model step are cleanly separated.

Design notes
-----------
- It does **not** import ``timesfm``. The adapter only produces numpy; the caller
  owns the model. tsauditor gains no new dependency.
- It does **not** pass ``target=target_col`` to ``fix``. ``fix`` deliberately
  protects the target column from repair (you never clean a label), but for a
  univariate forecast this series is exactly what must be cleaned. So it is
  repaired as an ordinary column.
- It **verifies** the result is finite before returning. Repair does not always
  eliminate every NaN (e.g. a lone, unflagged missing value, or an all-NaN
  column), so the adapter raises rather than let a NaN reach the model and fail
  tokenization silently. This is what makes the "crash-free" claim true.
- Imputed values are *estimates*, not ground truth: the model treats them as real
  history. Inspect ``report.last_fixes`` (pass ``return_report=True``) to see
  what was filled before trusting a forecast.

Context length: TimesFM 2.5 accepts a wide range of context lengths (up to 16k)
and does not require a frequency indicator. ``context_len`` and ``min_context``
below are *your* knobs, not TimesFM constants — set them for your use case and
the model version you target.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd


def to_timesfm(
    df: pd.DataFrame,
    target_col: str,
    *,
    domain: Optional[str] = None,
    context_len: int = 1024,
    min_context: int = 32,
    return_report: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, "object"]]:
    """
    Audit, repair, and format a single series into a TimesFM-ready array.

    Parameters
    ----------
    df : pd.DataFrame
        Raw input with the series to forecast.
    target_col : str
        Column to extract and forecast.
    domain : Optional[str]
        Domain hint passed to the audit ("finance", "sensor", or None).
    context_len : int
        Maximum number of trailing points to keep. Longer series are truncated to
        the most recent ``context_len``. Default 1024 (conservative; TimesFM 2.5
        supports far longer contexts — raise it if you want more history).
    min_context : int
        Your minimum acceptable length; the adapter raises below it. Default 32.
        This is a caller-set guard, not a TimesFM requirement — verify what your
        target model version actually needs.
    return_report : bool
        If True, return ``(array, report)`` so the audit trail is not discarded.
        Default False.

    Returns
    -------
    np.ndarray | tuple[np.ndarray, GuardReport]
        A 1-D ``float32`` array (most recent ``context_len`` points), optionally
        with the ``GuardReport`` describing what was audited and repaired.

    Raises
    ------
    KeyError
        If ``target_col`` is not in ``df``.
    TypeError
        If ``target_col`` is not numeric (a categorical or string column, for
        example). Raised here rather than surfacing as a raw ``ValueError``
        from the numpy conversion further down.
    ValueError
        If the repaired series still contains non-finite values, or has fewer
        than ``min_context`` points.
    """
    from tsauditor import fix

    if target_col not in df.columns:
        raise KeyError(f"target_col '{target_col}' not found in DataFrame columns.")

    if not pd.api.types.is_numeric_dtype(df[target_col]):
        # Without this, a non-numeric target_col (a categorical, a string
        # column, anything remediate.py's own numeric-dtype guards silently
        # pass through unrepaired) fails much later and much less clearly:
        # `clean[target_col].to_numpy(dtype=np.float32)` below raises a raw
        # `ValueError: could not convert string to float: '...'` with no
        # mention of target_col or what the caller needs to fix. The finite
        # check further down exists for exactly this reason -- to raise
        # clearly rather than let a bad value reach the model silently --
        # and a non-numeric column deserves the same treatment, not a
        # generic numpy conversion error.
        raise TypeError(
            f"target_col '{target_col}' has dtype {df[target_col].dtype}, not "
            f"numeric. TimesFM forecasts a numeric series; encode or select a "
            f"numeric column before calling to_timesfm()."
        )

    # target_col is the series to forecast, so it is cleaned as an ordinary
    # column here (not protected the way fix() protects a label).
    clean, report = fix(df, domain=domain)

    values = clean[target_col].to_numpy(dtype=np.float32)

    finite = np.isfinite(values)
    if not finite.all():
        raise ValueError(
            f"'{target_col}' still has {int((~finite).sum())} non-finite value(s) "
            f"after repair, so it cannot be fed to TimesFM. Inspect the missing "
            f"values (e.g. lone/unflagged NaNs) before forecasting; "
            f"report.last_fixes shows what was repaired."
        )

    if len(values) < min_context:
        raise ValueError(
            f"Series has {len(values)} points, fewer than the required minimum "
            f"({min_context}). Lower min_context or supply more history."
        )

    if len(values) > context_len:
        values = values[-context_len:]  # keep the most recent context window

    return (values, report) if return_report else values
