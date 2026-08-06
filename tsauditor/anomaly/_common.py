"""
tsauditor.anomaly._common
--------------------------
Shared thresholds, presets, and mask logic used by the anomaly detectors
(point.py's ANO002, contextual.py's ANO001/ANO003) *and* by remediate.py's
repair step.

Before this module existed, remediate.py kept its own hand-written copy of
every domain preset and every masking formula, connected to the real
detectors only by a comment ("Match anomaly/point.py ANO002"). That drifted
out of sync at least once already (the ANO001 single-row-gap bridge was added
to contextual.py without a matching update to remediate's copy, so `scan()`
would flag a run that `apply_fixes()` then silently failed to repair -- see
CHANGELOG [0.5.0]). Centralizing the presets and masks here means there
is exactly one place to change a threshold or a formula, and detection and
repair cannot disagree about what they mean by "stuck", "outlier", or "spike"
because they call the same function.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

# Contextual window for ANO003, shared by the detector and the repair step.
# Must be wide enough to estimate the local spread reliably: a 4-5 point
# window gives a noisy std and floods the result with false positives once
# the current point is excluded.
SPIKE_WINDOW = 21


# ── domain presets ─────────────────────────────────────────────────────────


def zscore_preset(domain: Optional[str]) -> float:
    """ANO002's default z-score threshold for a domain (finance 5.0, sensor
    3.5, otherwise 4.0)."""
    if domain == "finance":
        return 5.0
    if domain == "sensor":
        return 3.5
    return 4.0


def stuck_window_preset(domain: Optional[str]) -> int:
    """ANO001's default stuck-run window for a domain (sensor 3, otherwise
    5)."""
    if domain == "sensor":
        return 3
    return 5


def spike_threshold_preset(domain: Optional[str]) -> float:
    """ANO003's default local z-score threshold for a domain (finance 4.0,
    sensor 3.0, otherwise 3.5)."""
    if domain == "finance":
        return 4.0
    if domain == "sensor":
        return 3.0
    return 3.5


# ── ANO002: z-score + IQR outliers ─────────────────────────────────────────


def zscore_iqr_masks(
    series: pd.Series, z_thresh: float
) -> Tuple[pd.Series, pd.Series, pd.Series, bool]:
    """
    Combined z-score and IQR outlier detection.

    Parameters
    ----------
    series : pd.Series
        Finite, NaN-free numeric values (inf/-inf and NaN should already be
        dropped by the caller).
    z_thresh : float
        Absolute z-score above which a point is flagged.

    Returns
    -------
    z_mask, iqr_mask, z_scores, degenerate : pd.Series, pd.Series, pd.Series, bool
        ``z_mask``/``iqr_mask`` are boolean masks aligned to ``series``'s
        index. ``z_scores`` is the signed z-score for every point (used by
        callers that need the worst offender or a magnitude for evidence).
        ``degenerate`` is True when the column has zero variance or fewer
        than two observations (``std`` is 0 or NaN) -- in that case both
        masks are all-False and ``z_scores`` is all-NaN, since there is
        nothing to flag either way.
    """
    mean, std = series.mean(), series.std()
    if std == 0 or pd.isna(std):
        false_mask = pd.Series(False, index=series.index)
        nan_scores = pd.Series(np.nan, index=series.index)
        return false_mask, false_mask, nan_scores, True

    z_scores = (series - mean) / std
    z_mask = z_scores.abs() > z_thresh

    q25, q75 = series.quantile([0.25, 0.75])
    iqr = q75 - q25
    iqr_mask = (series < q25 - 1.5 * iqr) | (series > q75 + 1.5 * iqr)

    return z_mask, iqr_mask, z_scores, False


def clip_bounds(series: pd.Series, z_thresh: float) -> Tuple[float, float]:
    """
    Winsorization bounds = the region a point must be in to be flagged by
    *neither* method: the intersection of the z-band and the IQR fence.
    Clipping to [L, U] pulls in exactly the flagged outliers and leaves every
    inlier untouched.
    """
    mean, std = series.mean(), series.std()
    q25, q75 = series.quantile([0.25, 0.75])
    iqr = q75 - q25
    lower = max(mean - z_thresh * std, q25 - 1.5 * iqr)
    upper = min(mean + z_thresh * std, q75 + 1.5 * iqr)
    return lower, upper


# ── ANO001: stuck runs ──────────────────────────────────────────────────────


def stuck_run_mask(series: pd.Series, window: int) -> Tuple[pd.Series, pd.Series]:
    """
    Run-length stuck-value mask: a run longer than ``window`` is flagged.

    Groups on a bridged view (a single interior NaN interpolated) rather than
    the raw series. A lone missing reading inside an otherwise-flat run is
    still a stuck run; grouping on the raw series would split it in two via
    ``diff()`` reading a NaN as "changed" both at the gap and at the row
    right after it, and neither half might cross ``window`` even though the
    true, uninterrupted run does. Interpolating a single NaN only produces a
    zero diff when both neighbours already agree, so a genuine transition (a
    gap between two *different* values) still breaks the group correctly --
    this never masks a real change, only bridges a real stuck run.

    Parameters
    ----------
    series : pd.Series
        The raw column (NaNs allowed; not dropped).
    window : int
        A run longer than this is flagged.

    Returns
    -------
    mask, counts : pd.Series, pd.Series
        ``mask`` is True for every row that is part of a flagged run
        (including a bridged gap). ``counts`` is each row's run length,
        needed by the detector for ``max_stuck_duration`` evidence.
    """
    bridge_series = series.interpolate(method="linear", limit=1)
    diffs = bridge_series.diff().ne(0).cumsum()
    counts = bridge_series.groupby(diffs).transform("count")
    mask = (counts > window) & series.notna()
    return mask, counts


# ── ANO003: contextual spikes ────────────────────────────────────────────────


def spike_stats(
    values: pd.Series, window: int, threshold: float
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Local-context spike detection: each point is compared to the mean/std of
    its surrounding window, *excluding the point itself*. If the point stayed
    in its own window, an extreme spike would inflate the window's mean and
    std and mask itself (a 50x spike scoring only z ~= 1.8 in a centered
    5-window was the original bug this guards against).

    Parameters
    ----------
    values : pd.Series
        NaN-free numeric values (the caller's NaN-handling view).
    window : int
        Width of the rolling local-context window.
    threshold : float
        Local z-score above which a point is flagged.

    Returns
    -------
    mask, z_scores, flat_context_spike, local_mean, local_std
        ``mask`` is the final spike flag (z-score rule OR the flat-context
        special case). ``z_scores`` is the local z-score for every point
        (NaN/inf where the local context is degenerate). ``flat_context_spike``
        is True where the local neighbourhood is perfectly flat (std == 0)
        but the point itself differs -- a definite spike whose z-score is
        undefined (x / 0), flagged explicitly instead of silently dropped as
        NaN. ``local_mean``/``local_std`` are returned so callers needing a
        repair band can derive ``local_mean +/- threshold * local_std``.
    """
    sq = values.pow(2)
    mp = max(3, window // 2)
    roll = values.rolling(window=window, center=True, min_periods=mp)
    roll_sq = sq.rolling(window=window, center=True, min_periods=mp)

    n_excl = roll.count() - 1  # neighbours, excluding self
    sum_excl = roll.sum() - values
    sumsq_excl = roll_sq.sum() - sq

    local_mean = sum_excl / n_excl
    local_var = (sumsq_excl / n_excl) - local_mean.pow(2)
    local_std = np.sqrt(local_var.clip(lower=0))  # clip kills tiny fp negatives
    deviation = (values - local_mean).abs()

    with np.errstate(divide="ignore", invalid="ignore"):
        z_scores = deviation / local_std

    flat_context_spike = (local_std == 0) & (deviation > 0) & (n_excl >= 2)
    mask = ((z_scores > threshold) | flat_context_spike).fillna(False)

    return mask, z_scores, flat_context_spike, local_mean, local_std


def spike_bounds(
    values: pd.Series, window: int, threshold: float
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Contextual-spike mask plus the local clip band, for repair. Returns
    (mask, lower, upper) where [lower, upper] is the local acceptable band
    (local_mean +/- threshold * local_std); clipping a flagged point to it
    pulls it back to the edge of its own neighbourhood.
    """
    mask, _, _, local_mean, local_std = spike_stats(values, window, threshold)
    lower = local_mean - threshold * local_std
    upper = local_mean + threshold * local_std
    return mask, lower, upper
