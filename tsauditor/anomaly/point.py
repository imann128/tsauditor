"""
tsauditor.anomaly.point
-----------------------
Point anomaly detection using z-score and IQR methods.

Issue codes raised
------------------
ANO002  Point outlier detected (z-score or IQR method).
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd

from tsauditor.report.summary import Issue


def audit_point_anomalies(
    df: pd.DataFrame,
    zscore_threshold: float = 4.0,
    domain: Optional[str] = None,
) -> List[Issue]:
    """
    Detect point outliers in each numeric column.

    Uses z-score by default. For financial domain, applies a wider
    threshold to avoid flagging legitimate fat-tail events.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a sorted DatetimeIndex.
    zscore_threshold : float
        Number of standard deviations to flag as outlier. Default 4.0.
    domain : Optional[str]
        "finance", "sensor", or None.

    Returns
    -------
    List[Issue]
    """
    # TODO: implement in Week 2
    raise NotImplementedError("audit_point_anomalies is not yet implemented.")
