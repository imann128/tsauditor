"""
tsauditor.anomaly.contextual
----------------------------
Contextual anomaly detection: stuck values and implausible jumps
evaluated relative to neighboring observations.

Issue codes raised
------------------
ANO001  Stuck values: same value repeated beyond plausibility threshold.
ANO003  Contextual anomaly: value implausible given neighboring context.
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd

from tsauditor.report.summary import Issue


def audit_contextual_anomalies(
    df: pd.DataFrame,
    stuck_window: int = 5,
    domain: Optional[str] = None,
) -> List[Issue]:
    """
    Detect contextual anomalies in each numeric column.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a sorted DatetimeIndex.
    stuck_window : int
        Number of consecutive identical values to flag as stuck.
        Finance default: 5. Sensor default: 3.
    domain : Optional[str]
        "finance", "sensor", or None.

    Returns
    -------
    List[Issue]
    """
    # TODO: implement in Week 2
    raise NotImplementedError("audit_contextual_anomalies is not yet implemented.")
