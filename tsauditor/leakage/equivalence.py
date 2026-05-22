"""
tsauditor.leakage.equivalence
------------------------------
Detects features that are mathematically equivalent or near-equivalent
to the target variable at lag 0.

This is the canonical OGDC ChangeP case: a feature derived from the
same-day closing price being used to predict a target also derived from
the same-day closing price. The two columns may have different names and
come from different sources, yet be functionally identical.

Detection method
----------------
Pearson correlation at lag 0 between each feature and the target.
If |corr| >= equivalence_threshold, the feature is flagged as LEK001.

The threshold is set conservatively at 0.95 by default. In financial
data, legitimate features rarely exceed 0.7 correlation with a return-
based target. A 0.95+ correlation almost always indicates equivalence.

Issue codes raised
------------------
LEK001  Target equivalence detected.
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd

from tsauditor.report.summary import Issue


def audit_equivalence(
    df: pd.DataFrame,
    target: str,
    equivalence_threshold: float = 0.95,
    domain: Optional[str] = None,
) -> List[Issue]:
    """
    Detect features that are near-identical to the target at lag 0.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a sorted DatetimeIndex.
    target : str
        Name of the target column.
    equivalence_threshold : float
        Absolute Pearson correlation threshold. Default 0.95.
    domain : Optional[str]
        "finance", "sensor", or None.

    Returns
    -------
    List[Issue]
        One LEK001 Issue per flagged feature column.
    """
    # TODO: implement in Week 3
    raise NotImplementedError("audit_equivalence is not yet implemented.")
