"""
tsauditor.leakage.correlation
------------------------------
Cross-correlation leakage detection across a range of lags.

A legitimate feature should correlate with the target at lag <= 0
(past or present information predicts future target). If the peak
correlation occurs at a positive lag, the feature contains future
information relative to the target — this is leakage.

Detection method
----------------
For each numeric feature column, compute the cross-correlation with
the target across lags in [-max_lag, +max_lag]. If the peak absolute
correlation occurs at a positive lag AND exceeds a minimum correlation
threshold, raise LEK002.

Issue codes raised
------------------
LEK002  Positive-lag peak detected.
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd

from tsauditor.report.summary import Issue


def audit_correlation_leakage(
    df: pd.DataFrame,
    target: str,
    max_lag: int = 10,
    min_correlation: float = 0.1,
    domain: Optional[str] = None,
) -> List[Issue]:
    """
    Detect leakage via cross-correlation peak at positive lags.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a sorted DatetimeIndex.
    target : str
        Name of the target column.
    max_lag : int
        Maximum lag (in periods) to test in each direction. Default 10.
    min_correlation : float
        Minimum absolute correlation for a lag to be considered meaningful.
        Prevents flagging near-zero noise correlations. Default 0.1.
    domain : Optional[str]
        "finance", "sensor", or None.

    Returns
    -------
    List[Issue]
        One LEK002 Issue per flagged feature column.
    """
    # TODO: implement in Week 3
    raise NotImplementedError("audit_correlation_leakage is not yet implemented.")
