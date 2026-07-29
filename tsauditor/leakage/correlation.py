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

from tsauditor.leakage._common import encode_target as _encode_target
from tsauditor.report.summary import Issue, WARNING


def _align(a: np.ndarray, b: np.ndarray, tau: int):
    """Slice ``a`` and ``b`` so element i pairs a_t with b_{t+tau}."""
    n = len(a)
    if tau >= 0:
        return a[: n - tau], b[tau:]
    s = -tau
    return a[s:], b[: n - s]


def audit_correlation_leakage(
    df: pd.DataFrame,
    target: str,
    max_lag: int = 10,
    min_correlation: float = 0.5,
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
        Minimum absolute correlation for a peak at a positive lag to be
        reported. Default 0.5.

        This gate carries more weight than it appears to. The rule below fires
        whenever the argmax over lags lands at a positive lag, and for two
        persistent series (a price level, a random walk, a slow AR process)
        spurious correlation is large by construction while *which* lag wins is
        close to a coin flip. A low gate therefore reports leakage between
        columns that are statistically independent.

        Measured over 100 trials per cell on 400-point series, where FP columns
        are two independently generated series and TP columns are a genuine
        t+1 lookahead::

            min_correlation   FP walk   FP AR(.98)   TP iid   TP walk
            0.1 (until 0.3.1)    37%          51%     100%      100%
            0.5 (current)        13%           8%     100%      100%

        Raising the gate removed no true positive in 200 trials.

        Note for anyone tempted to replace this with a margin over the lag-0
        correlation, which is what LEK003 does: it does not work here. On a
        persistent target a genuine lookahead correlates with the target at
        lag 0 almost as strongly as at lag 1, so a flat margin suppresses real
        leaks too. Measured, a 0.10 margin cut false positives to 3% but
        dropped true detection on a random-walk target from 100% to 0%.
        LEK003 escapes this by dividing by the target's *measured*
        autocorrelation rather than subtracting a constant.
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

    # Rank-transform the target once (Spearman == Pearson of ranks). Ranking the
    # full series a single time and correlating the shifted ranks across lags
    # avoids re-ranking on every lag — the previous hot path.
    ry = y.rank().to_numpy(dtype=float)

    for col in df.select_dtypes(include=["number"]).columns:
        if col == target:
            continue

        x = df[col].astype(float).replace([np.inf, -np.inf], np.nan)
        if x.nunique() < 2:
            continue
        rx = x.rank().to_numpy(dtype=float)

        best_lag = 0
        best_signed = 0.0
        best_abs = 0.0

        for tau in range(-max_lag, max_lag + 1):
            # r(tau) = corr(feature_t, target_{t+tau}) on rank-transformed data.
            a, b = _align(rx, ry, tau)
            mask = ~(np.isnan(a) | np.isnan(b))
            if int(mask.sum()) < min_obs:
                continue
            aa, bb = a[mask], b[mask]
            if aa.std() == 0 or bb.std() == 0:  # constant subset -> undefined
                continue
            r = float(np.corrcoef(aa, bb)[0, 1])
            if np.isnan(r):
                continue
            if abs(r) > best_abs:
                best_abs, best_lag, best_signed = abs(r), tau, r

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
