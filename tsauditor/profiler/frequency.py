"""
tsauditor.profiler.frequency
----------------------------
Checks timestamp regularity, gap clustering, and duplicate timestamps.

Issue codes raised
------------------
PRF001  Irregular timestamp frequency detected.
PRF004  Duplicate timestamps detected.
PRF005  Gap cluster detected (missing periods concentrated in one window).
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd

from tsauditor.report.summary import Issue


def audit_frequency(
    df: pd.DataFrame,
    domain: Optional[str] = None,
) -> List[Issue]:
    """
    Audit the timestamp structure of the DataFrame index.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a sorted DatetimeIndex.
    domain : Optional[str]
        "finance", "sensor", or None.

    Returns
    -------
    List[Issue]
        Zero or more Issue objects.
    """
    # TODO: implement in Week 1
    raise NotImplementedError("audit_frequency is not yet implemented.")
