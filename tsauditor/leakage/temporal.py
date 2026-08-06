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

All three quantities above are computed on one common sample per feature
and lag: rows where the feature, the target, and the shifted target are
*simultaneously* non-null. Computing each on its own independent
pairwise-complete sample instead lets them silently describe different
populations whenever a feature has its own missingness (e.g. a column only
recorded starting partway through the series), which can shift the
persistence estimate by more than the default ``excess_threshold`` on
realistic data. See ``_aligned_correlations`` and CHANGELOG [0.5.0]
for the concrete case that motivated this.

Issue codes raised
------------------
LEK003  Rolling window lookahead suspected.  (WARNING)
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from tsauditor.leakage._common import encode_target as _encode_target
from tsauditor.report.summary import Issue, WARNING
from tsauditor.utils.validation import ensure_sorted_datetime_index


def _spearman(a: pd.Series, b: pd.Series, min_obs: int) -> Optional[float]:
    """Pairwise-complete Spearman correlation, or None if underdetermined."""
    pair = pd.concat([a, b], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(pair) < min_obs:
        return None
    if pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
        return None
    r = pair.iloc[:, 0].corr(pair.iloc[:, 1], method="spearman")
    return None if pd.isna(r) else float(r)


def _aligned_correlations(
    x: pd.Series, y: pd.Series, future_y: pd.Series, min_obs: int
):
    """
    r0 (x vs y), persistence (y vs future_y), and observed (x vs future_y),
    all three computed on one common mask: rows where x, y, and future_y are
    *simultaneously* non-null and finite.

    Why this matters: computing each correlation on its own independent
    pairwise-complete sample (the previous approach) lets them describe
    different populations whenever the feature has its own missingness --
    e.g. a column only recorded starting partway through the series. The
    persistence baseline is supposed to answer "how far could this specific
    feature's own population legitimately reach into the future via
    persistence alone?", not "how persistent is the target in general,
    including periods this feature was never even present for." On a
    synthetic regime-switching target (persistent early, choppy late) with a
    trailing, honest feature recorded only in the choppy half, persistence
    measured on the full series came out 0.75; measured on just the rows the
    feature actually occupies, 0.22 -- a difference far larger than the
    default ``excess_threshold`` of 0.1, easily large enough to flip a
    verdict. See CHANGELOG [0.5.0] for the concrete case.

    Returns
    -------
    (r0, persistence, observed) : tuple[float | None, float | None, float | None]
        Any entry is None if the common sample has fewer than ``min_obs``
        rows or if any of the three series is constant on that sample.
    """
    common = (
        pd.concat([x, y, future_y], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    )
    if len(common) < min_obs:
        return None, None, None

    cx, cy, cf = common.iloc[:, 0], common.iloc[:, 1], common.iloc[:, 2]
    if cx.nunique() < 2 or cy.nunique() < 2 or cf.nunique() < 2:
        return None, None, None

    r0 = cx.corr(cy, method="spearman")
    persistence = cy.corr(cf, method="spearman")
    observed = cx.corr(cf, method="spearman")

    def _clean(v):
        return None if pd.isna(v) else float(v)

    return _clean(r0), _clean(persistence), _clean(observed)


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

    Notes
    -----
    The present-day correlation (``r0``), the target's own persistence, and
    the observed future correlation are all computed on one common,
    simultaneously-non-null sample per feature and lag, not three
    independent pairwise-complete samples. This matters whenever a feature
    has its own missingness (e.g. a column only recorded starting partway
    through the series): the target's persistence can differ materially
    between the period the feature occupies and the series as a whole, and
    measuring it on the wrong population can mask a real leak or, less
    often, flag an honest feature. See ``_aligned_correlations`` for the
    concrete case that motivated this (a whole-series persistence estimate
    of 0.75 versus 0.22 restricted to a feature's own rows, on the same
    data).
    """
    issues: List[Issue] = []

    if target not in df.columns:
        raise ValueError(f"target '{target}' not found in DataFrame columns.")

    # y.shift(-k) below is positional, not label-aware, so an out-of-order
    # DatetimeIndex silently shifts by row position rather than by time
    # distance. See ensure_sorted_datetime_index's docstring.
    df = ensure_sorted_datetime_index(df, "audit_temporal_leakage")

    y = _encode_target(df[target], target)
    if y.dropna().nunique() < 2:
        return issues

    # The shifted-target series do not depend on any feature, so they're built
    # once here. `persistence_prefilter` is a cheap, deliberately *unaligned*
    # early-exit signal only -- computed on the loosest possible (y,
    # future_y) pairwise-complete sample, ignoring any feature's own
    # missingness. It is never used in the expected(k) math itself, only to
    # skip a lag outright when even that loosest sample already has fewer
    # than min_obs rows: any feature-aligned sample below is a *subset* of
    # this one (it additionally requires the feature to be non-null), so it
    # can only be smaller, never larger -- this pre-filter therefore never
    # discards a lag that the aligned computation could otherwise use.
    futures = {k: y.shift(-k) for k in range(1, max_lag + 1)}
    persistence_prefilter = {
        k: _spearman(y, futures[k], min_obs) for k in range(1, max_lag + 1)
    }

    for col in df.select_dtypes(include=["number"]).columns:
        if col == target:
            continue

        x = df[col].astype(float).replace([np.inf, -np.inf], np.nan)
        if x.nunique() < 2:
            continue

        # Same cheap-pre-filter reasoning as above: the aligned per-lag sample
        # (which also requires future_y non-null) is always a subset of this
        # (x, y) pairwise-complete sample, so if even this one is too small,
        # every lag's aligned sample is too.
        if _spearman(x, y, min_obs) is None:
            continue

        best_excess = 0.0
        best_lag = 0
        best_observed = 0.0

        for k in range(1, max_lag + 1):
            if persistence_prefilter[k] is None:
                continue

            # r0, persistence, and observed here are all computed on the
            # *same* common mask (rows where x, y, and future_y are all
            # simultaneously present), unlike the three independent
            # pairwise-complete samples above. This is what expected(k) =
            # |r0| * |persistence| actually needs to mean something: a bound
            # on what this feature's own population could legitimately reach
            # via persistence, not a bound estimated from a population the
            # feature was never even observed in.
            r0, per, observed = _aligned_correlations(x, y, futures[k], min_obs)
            if r0 is None or per is None or observed is None:
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
