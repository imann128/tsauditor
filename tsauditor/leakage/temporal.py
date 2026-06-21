"""
tsauditor.leakage.temporal
---------------------------
Rolling/lagged window lookahead detection.

A rolling feature computed with window W at time T should only use data
from [T-W+1, T]. A forward-looking or centered window also pulls in values
at T+1 and beyond, so the feature ends up carrying genuine future
information about the target.

The hard part — and why a naive test fails
-------------------------------------------
Time-series targets (e.g. price levels) are strongly autocorrelated. A
perfectly *legitimate* trailing feature will therefore still correlate with
the target's future, purely through persistence: if feature_t tracks
target_t, and target_t predicts target_{t+k} on its own, then feature_t
correlates with target_{t+k} too. A detector that just looks at
"correlation with the future" would flag every honest feature.

So we control for that persistence explicitly. The future correlation a
feature can reach *legitimately* is bounded by its present association with
the target times the target's own autocorrelation:

    expected(k) = corr(feature_t, target_t) * corr(target_t, target_{t+k})

We compare this to what is actually observed:

    observed(k) = corr(feature_t, target_{t+k})

If observed(k) exceeds expected(k) by more than ``excess_threshold`` at any
lag k in 1..max_lag, the feature knows the future better than persistence
alone allows — the signature of a forward-looking window. All correlations
are Spearman, for consistency with the rest of the leakage module.

Issue codes raised
------------------
LEK003  Rolling window lookahead suspected.  (WARNING)
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


def _spearman(a: pd.Series, b: pd.Series, min_obs: int) -> Optional[float]:
    """Pairwise-complete Spearman correlation, or None if underdetermined."""
    pair = pd.concat([a, b], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(pair) < min_obs:
        return None
    if pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
        return None
    r = pair.iloc[:, 0].corr(pair.iloc[:, 1], method="spearman")
    return None if pd.isna(r) else float(r)


def audit_temporal_leakage(
    df: pd.DataFrame,
    target: str,
    max_lag: int = 5,
    excess_threshold: float = 0.1,
    min_correlation: float = 0.1,
    min_obs: int = 30,
    domain: Optional[str] = None,
) -> List[Issue]:
    """
    Detect suspected lookahead in rolling or lagged features.

    A feature is flagged (LEK003) if, at some lag k in 1..max_lag, its
    observed correlation with the future target exceeds the level reachable
    through the target's own persistence by more than ``excess_threshold``.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with rows in time order (sorted DatetimeIndex).
    target : str
        Name of the target column. Must exist in ``df``.
    max_lag : int
        Number of forward lags to examine. Default 5.
    excess_threshold : float
        How much the observed future correlation must exceed the
        persistence-explained baseline to be flagged. Default 0.1.
    min_correlation : float
        The observed future correlation must itself be at least this large,
        so trivial noise excesses are ignored. Default 0.1.
    min_obs : int
        Minimum overlapping observations for a correlation to count. Default 30.
    domain : Optional[str]
        Accepted for API consistency.

    Returns
    -------
    List[Issue]
        Zero or more LEK003 Issues.
    """
    issues: List[Issue] = []

    if target not in df.columns:
        raise ValueError(f"target '{target}' not found in DataFrame columns.")

    y = _encode_target(df[target], target)
    if y.dropna().nunique() < 2:
        return issues

    # The target's own autocorrelation does not depend on any feature, so the
    # shifted-target series and their persistence correlations are computed once
    # here rather than recomputed inside the per-feature loop.
    futures = {k: y.shift(-k) for k in range(1, max_lag + 1)}
    persistence = {k: _spearman(y, futures[k], min_obs) for k in range(1, max_lag + 1)}

    for col in df.select_dtypes(include=["number"]).columns:
        if col == target:
            continue

        x = df[col].astype(float).replace([np.inf, -np.inf], np.nan)
        if x.nunique() < 2:
            continue

        r0 = _spearman(x, y, min_obs)  # present feature-target link
        if r0 is None:
            continue

        best_excess = 0.0
        best_lag = 0
        best_observed = 0.0

        for k in range(1, max_lag + 1):
            per = persistence[k]  # target autocorrelation
            if per is None:
                continue
            observed = _spearman(x, futures[k], min_obs)  # feature vs future target
            if observed is None:
                continue

            expected = abs(r0) * abs(per)  # legitimately reachable
            excess = abs(observed) - expected
            if excess > best_excess:
                best_excess = excess
                best_lag = k
                best_observed = observed

        if (
            best_lag > 0
            and best_excess >= excess_threshold
            and abs(best_observed) >= min_correlation
        ):
            issues.append(
                Issue(
                    module="leakage",
                    code="LEK003",
                    severity=WARNING,
                    description=(
                        f"Feature '{col}' correlates with target '{target}' at lag "
                        f"+{best_lag} (Spearman={best_observed:.3f}) more strongly than "
                        f"the target's own persistence explains (excess="
                        f"{best_excess:.3f}). This is the signature of a forward-looking "
                        f"or centered window — verify the feature uses only past data."
                    ),
                    column=col,
                    evidence={
                        "lag": int(best_lag),
                        "observed_future_corr": round(best_observed, 4),
                        "excess_over_persistence": round(best_excess, 4),
                        "excess_threshold": excess_threshold,
                        "metric": "spearman",
                    },
                )
            )

    return issues
