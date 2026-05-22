"""
tsauditor.leakage.temporal
---------------------------
Rolling window lookahead detection.

A rolling feature computed with window W at time T should only use
data from [T-W+1, T]. If it includes T+1 or beyond, it is leaky.
This is detectable statistically: a rolling feature will show unusually
high correlation with the target at lag 0 compared to lag 1, even after
controlling for autocorrelation structure.

Issue codes raised
------------------
LEK003  Rolling window lookahead suspected.
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd

from tsauditor.report.summary import Issue


def audit_temporal_leakage(
    df: pd.DataFrame,
    target: str,
    domain: Optional[str] = None,
) -> List[Issue]:
    """
    Detect suspected lookahead in rolling or lagged features.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a sorted DatetimeIndex.
    target : str
        Name of the target column.
    domain : Optional[str]
        "finance", "sensor", or None.

    Returns
    -------
    List[Issue]
        Zero or more LEK003 Issues.
    """
    # TODO: implement in Week 3
    raise NotImplementedError("audit_temporal_leakage is not yet implemented.")
