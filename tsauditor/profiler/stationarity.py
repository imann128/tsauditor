"""
tsauditor.profiler.stationarity
--------------------------------
ADF stationarity test per numeric column using scipy.

Issue codes raised
------------------
PRF003  Non-stationary column detected (ADF p-value > threshold).
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd

from tsauditor.report.summary import Issue


def audit_stationarity(
    df: pd.DataFrame,
    alpha: float = 0.05,
    domain: Optional[str] = None,
) -> List[Issue]:
    """
    Run ADF stationarity test on each numeric column.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a sorted DatetimeIndex.
    alpha : float
        Significance level for the ADF test. Default 0.05.
    domain : Optional[str]
        "finance", "sensor", or None.

    Returns
    -------
    List[Issue]
        One Issue per non-stationary column.
    """
    # TODO: implement in Week 1
    raise NotImplementedError("audit_stationarity is not yet implemented.")
