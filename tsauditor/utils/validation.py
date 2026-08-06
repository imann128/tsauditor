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


def _is_polars(obj) -> bool:
    """True if ``obj`` is a polars DataFrame, without importing polars."""
    return type(obj).__module__.split(".", 1)[0] == "polars"


def _polars_to_pandas(df, time_col: Optional[str]) -> pd.DataFrame:
    """
    Convert a polars DataFrame to pandas at the scan() boundary.

    polars has no index, so a polars input must name its datetime column via
    ``time_col`` — there is otherwise no way to know which column is time.
    See https://github.com/imann128/tsauditor/issues/28.
    """
    if time_col is None:
        raise ValueError(
            "polars input requires time_col= (polars has no index). "
            "Pass tsauditor.scan(df, time_col='your_datetime_column'). "
            "See https://github.com/imann128/tsauditor/issues/28"
        )
    try:
        return df.to_pandas()
    except Exception as exc:  # pragma: no cover - depends on user's pyarrow
        raise ImportError(
            "Converting a polars DataFrame requires pyarrow. "
            "Install it with:  pip install 'tsauditor[polars]'"
        ) from exc


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
    # polars support (issue #28): convert at the boundary; internals stay pandas.
    if _is_polars(df):
        df = _polars_to_pandas(df, time_col)

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
        # Do NOT silently coerce a numeric index. pd.to_datetime would happily
        # reinterpret a plain RangeIndex like [0, 1, 2, ...] as nanosecond epoch
        # timestamps (all near 1970-01-01), producing quietly wrong frequency and
        # gap results. A numeric index almost never means "datetime", so refuse it.
        if pd.api.types.is_numeric_dtype(df.index):
            raise ValueError(
                "DataFrame index is numeric, not datetime, and will not be coerced "
                "(it would be misread as epoch timestamps). Pass "
                "time_col='your_date_column' or set a DatetimeIndex before calling "
                "tsauditor.scan()."
            )
        # For string/object date-like labels, attempt coercion as a last resort.
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            raise ValueError(
                "DataFrame index is not a DatetimeIndex and could not be coerced. "
                "Either pass time_col='your_date_column' or set the index to datetime "
                "before calling tsauditor.scan()."
            )

    # kind="mergesort" specifically: it is the one sort pandas/numpy document
    # as stable. The default, "quicksort", is not, so two rows sharing an
    # identical duplicate timestamp can come out in an order that depends on
    # the input's original row order in an unspecified way rather than
    # preserving it. That matters here because a duplicate timestamp is
    # already a PRF004 CRITICAL finding, and both audit_frequency's own
    # dedup (`df[~df.index.duplicated(keep="first")]`) and anything else
    # downstream that assumes "first" means "first as the caller supplied
    # it" would otherwise silently pick an arbitrary survivor -- the same
    # DataFrame's rows in a different (but equally valid) input order could
    # produce a different repaired result.
    df = df.sort_index(kind="mergesort")

    # ── Validate target ───────────────────────────────────────────────────────
    if target is not None and target not in df.columns:
        raise ValueError(
            f"target='{target}' not found in DataFrame columns: {list(df.columns)}"
        )

    return df


def ensure_sorted_datetime_index(df: pd.DataFrame, context: str) -> pd.DataFrame:
    """
    Validate that ``df`` has a DatetimeIndex and return it sorted ascending.

    Every detector whose logic depends on row order (rolling windows,
    ``.shift()``, consecutive-run detection, positional lag alignment) must
    call this at its *own* entry point, not just rely on scan()'s
    ``validate_dataframe`` having already sorted upstream. Every
    ``audit_*``/``detect_*`` function in this package is also public API —
    called directly in this codebase's own test suite (see
    ``tests/test_adapters.py``, and the leakage/anomaly unit tests) — so
    "the caller already sorted it" is only true on the ``scan()`` path, not
    when a user imports a detector and calls it themselves.

    Before this existed, a DataFrame with a genuinely valid DatetimeIndex
    that was merely out of chronological order made these detectors produce
    wrong-but-silent results instead of an error: ``audit_correlation_leakage``
    and ``audit_temporal_leakage`` missed a perfect, constructed lag+1 leak
    entirely (returned ``[]``, no exception) when the same rows were shuffled
    out of order; ``audit_contextual_anomalies`` likewise missed an 8-point
    stuck run. Both are silent false negatives in exactly the class of bug
    this library exists to catch.

    Uses ``kind="mergesort"`` (stable) for the same reason
    ``validate_dataframe`` does: two rows sharing a duplicate timestamp must
    keep their original relative order rather than an unspecified one.

    Parameters
    ----------
    df : pd.DataFrame
        Input to validate.
    context : str
        Short description of the caller (e.g. "audit_correlation_leakage"),
        used only to make the error message point at the right function.

    Returns
    -------
    pd.DataFrame
        ``df`` sorted ascending by its DatetimeIndex.

    Raises
    ------
    ValueError
        If ``df.index`` is not a ``pd.DatetimeIndex``.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        # Message kept consistent with every other detector's existing
        # ValueError wording ("DataFrame index must be a pd.DatetimeIndex")
        # rather than inventing new phrasing -- callers and tests across the
        # codebase already match against that exact string.
        raise ValueError(f"DataFrame index must be a pd.DatetimeIndex ({context}).")
    return df.sort_index(kind="mergesort")


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
