"""
tsauditor.leakage._common
--------------------------
Shared helpers used across the leakage detectors.

Kept here rather than duplicated per-module (correlation.py, temporal.py, and
equivalence.py all needed the same target-encoding logic, and had each grown
their own byte-identical copy) so a future change to the encoding rule cannot
drift out of sync between detectors that are supposed to agree on it.
"""

import pandas as pd


def encode_target(series: pd.Series, name: str) -> pd.Series:
    """
    Return a numeric float target; encode a binary categorical as 0.0/1.0.

    Numeric targets pass through unchanged (as float). A non-numeric target
    is only accepted if it has exactly two distinct non-null values — encoded
    deterministically by sorting the categories as strings, so the same input
    always maps to the same 0/1 assignment regardless of row order. Anything
    else (more than two categories, or non-numeric with fewer than two) is a
    caller error: there is no sound way to correlate a target with three or
    more unordered categories against a single numeric feature.

    Parameters
    ----------
    series : pd.Series
        The raw target column.
    name : str
        The target's column name, used only to make the error message
        actionable.

    Returns
    -------
    pd.Series
        A float series: the original values if already numeric, or 0.0/1.0
        for a binary categorical.

    Raises
    ------
    ValueError
        If the target is non-numeric and not binary.
    """
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    categories = sorted(series.dropna().unique(), key=str)
    if len(categories) == 2:
        return series.map({categories[0]: 0.0, categories[1]: 1.0})
    raise ValueError(
        f"target '{name}' is non-numeric and not binary; cannot correlate."
    )
