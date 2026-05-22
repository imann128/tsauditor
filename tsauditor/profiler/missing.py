"""
tsauditor.profiler.missing
--------------------------
Temporal missing value pattern analysis.

Distinguishes between randomly distributed missing values (acceptable)
and clustered missing values (indicative of a data feed outage or
systematic collection failure).

Issue codes raised
------------------
PRF002  Clustered missing values detected.
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd

from tsauditor.report.summary import Issue


def audit_missing(
    df: pd.DataFrame,
    domain: Optional[str] = None,
) -> List[Issue]:
    """
    Audit missing value patterns across all columns.

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
    raise NotImplementedError("audit_missing is not yet implemented.")
