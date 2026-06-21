"""
tsauditor.leakage.correlation
------------------------------
Cross-correlation leakage detection across a range of lags.

A legitimate feature should carry its information from the past or present:
its association with the target should peak at lag <= 0. If a feature's
peak cross-correlation with the target occurs at a *positive* lag, the
feature aligns most strongly with *future* target values — a signature of
lookahead leakage.

Detection method
----------------
For each numeric feature, compute the rank (Spearman) cross-correlation
with the target across lags in [-max_lag, +max_lag], where

    r(tau) = corr( feature_t , target_{t+tau} )

so tau > 0 means the feature is being compared against the target's future.
If the peak |r| occurs at a positive lag AND exceeds ``min_correlation``,
raise LEK002.

Spearman is used (not Pearson) for consistency with the equivalence module
and because it is robust and captures monotonic association; binary targets
are encoded 0/1 (the rank correlation is attenuated but the *lag* of the
peak — the actual signal here — is preserved).

Important limitation
--------------------
In pure cross-correlation a genuine strong predictor and a lookahead leak
both peak at a positive lag. The separator is magnitude: real one-step
predictive power is weak, whereas leakage is strong. LEK002 is therefore a
WARNING-level *suspicion* flag for review, not a proof of leakage.

Issue codes raised
------------------
LEK002  Positive-lag peak detected.  (WARNING)
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from tsauditor.report.summary import Issue, WARNING


def _encode_target(series: pd.Series, name: str) -> pd.Series:
    """Return a numeric float target; encode a binary categorical as 0/1."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    categories = sorted(series.dropna().unique(), key=str)
    if len(categories) == 2:
        return series.map({categories[0]: 0.0, categories[1]: 1.0})
    raise ValueError(
        f"target '{name}' is non-numeric and not binary; cannot correlate."
    )


def audit_correlation_leakage(
    df: pd.DataFrame,
    target: str,
    max_lag: int = 10,
    min_correlation: float = 0.1,
    min_obs: int = 30,
    domain: Optional[str] = None,
) -> List[Issue]:
    """
    Detect leakage via a cross-correlation peak at positive lags.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with rows in time order (sorted DatetimeIndex).
    target : str
        Name of the target column. Must exist in ``df``.
    max_lag : int
        Maximum lag (in periods) to test in each direction. Default 10.
    min_correlation : float
        Minimum absolute correlation for a peak to be considered meaningful.
        Prevents flagging near-zero noise correlations. Default 0.1.
    min_obs : int
        Minimum overlapping observations at the peak lag for it to count.
        Default 30.
    domain : Optional[str]
        Accepted for API consistency.

    Returns
    -------
    List[Issue]
        One LEK002 Issue per flagged feature column.
    """
    issues: List[Issue] = []

    if target not in df.columns:
        raise ValueError(f"target '{target}' not found in DataFrame columns.")

    y = _encode_target(df[target], target)
    if y.dropna().nunique() < 2:
        return issues

    lags = range(-max_lag, max_lag + 1)

    for col in df.select_dtypes(include=["number"]).columns:
        if col == target:
            continue

        x = df[col].astype(float).replace([np.inf, -np.inf], np.nan)
        if x.nunique() < 2:
            continue

        best_lag = 0
        best_signed = 0.0
        best_abs = 0.0

        for tau in lags:
            # r(tau) = corr(feature_t, target_{t+tau}); shift(-tau) brings
            # target_{t+tau} onto row t. .corr aligns on index, drops NaNs.
            shifted = y.shift(-tau)
            pair = pd.concat([x, shifted], axis=1).dropna()
            if len(pair) < min_obs:
                continue
            if pair.iloc[:, 1].nunique() < 2:
                continue
            r = pair.iloc[:, 0].corr(pair.iloc[:, 1], method="spearman")
            if pd.isna(r):
                continue
            if abs(r) > best_abs:
                best_abs, best_lag, best_signed = abs(r), tau, float(r)

        if best_lag > 0 and best_abs >= min_correlation:
            issues.append(
                Issue(
                    module="leakage",
                    code="LEK002",
                    severity=WARNING,
                    description=(
                        f"Feature '{col}' has its peak cross-correlation with target "
                        f"'{target}' at lag +{best_lag} (Spearman={best_signed:.3f}); it "
                        f"aligns most strongly with future target values, suggesting "
                        f"lookahead leakage. Review how this feature is constructed."
                    ),
                    column=col,
                    evidence={
                        "peak_lag": int(best_lag),
                        "peak_correlation": round(best_signed, 4),
                        "min_correlation": min_correlation,
                        "max_lag": max_lag,
                        "metric": "spearman",
                    },
                )
            )

    return issues
