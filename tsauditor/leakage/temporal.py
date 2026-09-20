"""
tsauditor.leakage.temporal
---------------------------
Rolling/lagged window lookahead detection.

A rolling feature computed with window W at time T should only use data
from [T-W+1, T]. A forward-looking or centered window also pulls in values
at T+1 and beyond, so the feature ends up carrying genuine future
information about the target.

The hard part, and why a naive test fails
-------------------------------------------
Time-series targets (e.g. price levels) are strongly autocorrelated. A
perfectly legitimate trailing feature will therefore still correlate with
the target's future, purely through persistence: if feature_t tracks
target_t, and target_t predicts target_{t+k} on its own, then feature_t
correlates with target_{t+k} too. A detector that just looks at
"correlation with the future" would flag every honest feature.

So we control for that persistence explicitly. The future correlation a
feature can reach legitimately is bounded by its present association with
the target times the target's own autocorrelation:

    expected(k) = corr(feature_t, target_t) * corr(target_t, target_{t+k})

We compare this to what is actually observed:

    observed(k) = corr(feature_t, target_{t+k})

If observed(k) exceeds expected(k) by more than ``excess_threshold`` at any
lag k in 1..max_lag, the feature knows the future better than persistence
alone allows: the signature of a forward-looking window. All correlations
are Spearman, for consistency with the rest of the leakage module.

Why the comparison happens in Fisher-z space, not raw correlation space
-------------------------------------------------------------------------
Correlation is bounded in [-1, 1] and compresses near the endpoints: the
gap between r=0.90 and r=0.95 represents far more "additional dependence"
than the same 0.05 gap between r=0.10 and r=0.15. For a near-unit-root
target (e.g. an undifferenced price level, this library's own stated
finance use case), persistence(k) and a trailing feature's observed(k) both
sit up near 0.999+, and a leaky feature's observed(k) sits only slightly
higher still. Subtracting two numbers both jammed against the same ceiling
destroys almost all resolving power: swept over an AR(1) target's own
autocorrelation phi, a deliberately leaky centered-window feature was
caught 100% of the time through phi=0.9, then collapsed to 13% at phi=0.95
and 0% from phi=0.97 through a literal random walk (phi=1.0), verified by
direct simulation, matching what the raw-difference formula predicts it
should do.

``excess(k)`` is therefore computed as the difference of Fisher
z-transforms (``arctanh``) rather than of the raw correlations:

    excess(k) = arctanh(|observed(k)|) - arctanh(|expected(k)|)

``arctanh`` is the standard variance-stabilizing transform for a
correlation coefficient: it stretches the space near +/-1 back out, so a
fixed ``excess_threshold`` means roughly the same "real" amount of excess
dependence regardless of how persistent the target is, instead of becoming
either unreachable (raw difference, high persistence) or noise-amplified.
A ratio/relative-excess reformulation was tried first and rejected: at
persistence near 1 its denominator ``1 - expected`` approaches zero, which
amplifies ordinary Spearman sampling noise into false positives (measured
12-30% false-positive rate on legitimately trailing features at phi >=
0.95, against zero for the z-transform at the same recall). At
small-to-moderate correlations arctanh is nearly the identity (arctanh(r)
~= r for |r| << 1), so this leaves detection at ordinary (non-persistent)
targets unchanged in practice: re-running this module's own pre-existing
test scenarios gave z-space excess values within 0.01 of the original
raw-difference values. ``excess_threshold``'s default (0.1) is unchanged
and was verified, not assumed, to still mean approximately the same thing
at typical persistence.

This is a real, verified improvement, not a full fix: at extreme
persistence (phi >= 0.97) a weak, heavily-noised leak (as opposed to an
obvious one like a centered rolling window) is still only caught 13-57% of
the time rather than the near-total collapse the raw-difference formula
produced, because distinguishing a barely-there leak from noise when the
target is nearly a random walk is intrinsically a harder statistics
problem, not merely a units problem. See CHANGELOG for the measured
before/after sweep.

All three quantities above are computed on one common sample per feature
and lag: rows where the feature, the target, and the shifted target are
simultaneously non-null. Computing each on its own independent
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


def _fisher_z(r: float) -> float:
    """
    Fisher's variance-stabilizing transform, ``arctanh(r)``.

    Clipped to +/-0.999999 first: an exact +/-1.0 correlation (a feature
    that is a deterministic function of the target on the aligned sample,
    routine for a tiny ``min_obs``-sized overlap, not just a contrived edge
    case) sends ``arctanh`` to +/-inf, which would make ``excess`` infinite
    and therefore trivially ">= excess_threshold" regardless of what the
    other side of the subtraction is. Clipping keeps a perfect correlation
    "very large" rather than "infinite", which behaves the same for
    thresholding purposes without the inf/inf-minus-inf edge cases that
    follow it downstream (e.g. an f-string formatting inf, or two perfectly
    correlated quantities producing an undefined inf - inf excess).
    """
    return float(np.arctanh(np.clip(r, -0.999999, 0.999999)))


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
    simultaneously non-null and finite.

    Why this matters: computing each correlation on its own independent
    pairwise-complete sample (the previous approach) lets them describe
    different populations whenever the feature has its own missingness,
    e.g. a column only recorded starting partway through the series. The
    persistence baseline is supposed to answer "how far could this specific
    feature's own population legitimately reach into the future via
    persistence alone?", not "how persistent is the target in general,
    including periods this feature was never even present for." On a
    synthetic regime-switching target (persistent early, choppy late) with a
    trailing, honest feature recorded only in the choppy half, persistence
    measured on the full series came out 0.75; measured on just the rows the
    feature actually occupies, 0.22, a difference far larger than the
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
        persistence-explained baseline to be flagged, measured in Fisher-z
        space (``arctanh(|observed|) - arctanh(|expected|)``), not raw
        correlation units; see the module docstring's "Why the comparison
        happens in Fisher-z space" section for why. Default 0.1, chosen
        because it reproduces this module's own pre-existing test scenarios
        to within 0.01 of the previous raw-difference values at ordinary
        (non-extreme) persistence: it means approximately the same thing it
        always did there, while no longer collapsing to an unreachable bar
        as persistence approaches 1.
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

    # The shifted-target series do not depend on any feature, so they're
    # built once here. `persistence_prefilter` is a cheap, deliberately
    # unaligned early-exit signal only, computed on the loosest possible
    # (y, future_y) pairwise-complete sample, ignoring any feature's own
    # missingness. It is never used in the expected(k) math itself, only to
    # skip a lag outright when even that loosest sample already has fewer
    # than min_obs rows: any feature-aligned sample below is a subset of
    # this one (it additionally requires the feature to be non-null), so it
    # can only be smaller, never larger. This pre-filter therefore never
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
        best_expected = 0.0

        for k in range(1, max_lag + 1):
            if persistence_prefilter[k] is None:
                continue

            # r0, persistence, and observed here are all computed on the
            # same common mask (rows where x, y, and future_y are all
            # simultaneously present), unlike the three independent
            # pairwise-complete samples above. This is what expected(k) =
            # |r0| * |persistence| actually needs to mean something: a bound
            # on what this feature's own population could legitimately reach
            # via persistence, not a bound estimated from a population the
            # feature was never even observed in.
            r0, per, observed = _aligned_correlations(x, y, futures[k], min_obs)
            if r0 is None or per is None or observed is None:
                continue

            expected = abs(r0) * abs(per)  # legitimately reachable, raw scale
            # Fisher-z difference, not a raw subtraction; see the module
            # docstring's "Why the comparison happens in Fisher-z space".
            # Raw `abs(observed) - expected` collapses to ~0 once both sides
            # are pinned near 1 (a near-unit-root target), even when the
            # feature genuinely knows more than persistence explains.
            excess = _fisher_z(abs(observed)) - _fisher_z(expected)
            if excess > best_excess:
                best_excess = excess
                best_lag = k
                best_observed = observed
                best_expected = expected

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
                        f"the target's own persistence explains (persistence-expected "
                        f"Spearman={best_expected:.3f}, Fisher-z excess={best_excess:.3f}). "
                        f"This is the signature of a forward-looking or centered window. "
                        f"Verify the feature uses only past data."
                    ),
                    column=col,
                    evidence={
                        "lag": int(best_lag),
                        "observed_future_corr": round(best_observed, 4),
                        "expected_from_persistence": round(best_expected, 4),
                        # Fisher-z scale (arctanh difference), not a raw
                        # correlation-point difference; see the module
                        # docstring. Comparable to excess_threshold, which is
                        # in the same units.
                        "excess_over_persistence": round(best_excess, 4),
                        "excess_threshold": excess_threshold,
                        "excess_scale": "fisher_z",
                        "metric": "spearman",
                    },
                )
            )

    return issues
