"""
tsauditor.utils.validation
--------------------------
Input validation and DataFrame normalization.
All public functions raise TypeError or ValueError with clear messages
so the user knows exactly what to fix before the scan proceeds.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd


def validate_dataframe(
    df: pd.DataFrame,
    target: Optional[str],
    time_col: Optional[str],
) -> pd.DataFrame:
    """
    Validate and normalize the input DataFrame.

    Steps
    -----
    1. Confirm input is a DataFrame.
    2. Resolve time index: use time_col if supplied, else expect DatetimeIndex.
    3. Sort by time index ascending.
    4. Validate target column exists (if supplied).

    Parameters
    ----------
    df : pd.DataFrame
        Raw input.
    target : Optional[str]
        Name of the target/label column.
    time_col : Optional[str]
        Name of the datetime column; None means the index is already datetime.

    Returns
    -------
    pd.DataFrame
        Normalized DataFrame with a sorted DatetimeIndex.

    Raises
    ------
    TypeError
        If df is not a DataFrame.
    ValueError
        If time_col or target column is missing, or index cannot be parsed
        as datetime.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"tsauditor.scan() expects a pandas DataFrame, got {type(df).__name__}."
        )

    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    df = df.copy()

    # ── Resolve datetime index ────────────────────────────────────────────────
    if time_col is not None:
        if time_col not in df.columns:
            raise ValueError(
                f"time_col='{time_col}' not found in DataFrame columns: {list(df.columns)}"
            )
        try:
            df[time_col] = pd.to_datetime(df[time_col])
        except Exception as exc:
            raise ValueError(
                f"Could not parse column '{time_col}' as datetime: {exc}"
            ) from exc
        df = df.set_index(time_col)

    if not isinstance(df.index, pd.DatetimeIndex):
        # Attempt coercion as a last resort
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            raise ValueError(
                "DataFrame index is not a DatetimeIndex and could not be coerced. "
                "Either pass time_col='your_date_column' or set the index to datetime "
                "before calling tsauditor.scan()."
            )

    df = df.sort_index()

    # ── Validate target ───────────────────────────────────────────────────────
    if target is not None and target not in df.columns:
        raise ValueError(
            f"target='{target}' not found in DataFrame columns: {list(df.columns)}"
        )

    return df


def infer_frequency(index: pd.DatetimeIndex) -> str:
    """
    Infer a human-readable frequency label from a DatetimeIndex.

    Returns one of: "daily", "weekly", "monthly", "sub-daily", "irregular".
    This is intentionally coarse — precise frequency inference is handled
    by profiler.frequency.
    """
    if len(index) < 2:
        return "unknown"

    median_delta = pd.Series(index).diff().dropna().median()

    if pd.isna(median_delta):
        return "unknown"

    hours = median_delta.total_seconds() / 3600

    if hours < 20:
        return "sub-daily"
    if 20 <= hours <= 28:
        return "daily"
    if 140 <= hours <= 196:
        return "weekly"
    if 600 <= hours <= 960:
        return "monthly"
    return "irregular"
