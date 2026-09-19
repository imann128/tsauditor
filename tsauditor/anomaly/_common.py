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

from typing import List, NamedTuple, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# Contextual window for ANO003, shared by the detector and the repair step.
# Must be wide enough to estimate the local spread reliably: a 4-5 point
# window gives a noisy std and floods the result with false positives once
# the current point is excluded.
SPIKE_WINDOW = 21

# Cap on how many outliers the ESD diagnostic will look for, as a fraction of
# the column length. ESD is O(k*n); beyond ~40% contamination the "outliers"
# are a second population rather than anomalies.
_ESD_MAX_FRACTION = 0.4
_ESD_MIN_OBS = 15


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

    Does not reach an ESD-recovered point (see ``esd_masking_recovery``): by
    construction such a point is inside both the z-band and the IQR fence
    (that is the entire reason the z-score and IQR rules both missed it), so
    clipping the whole column to this intersection cannot move it. No
    winsorization bound exists for these points either (see
    ``_generalized_esd``'s docstring) -- ``remediate.py``'s
    ``_esd_masking_repair`` NaNs the specific rows ``esd_masking_recovery``
    names instead of clipping them.
    """
    mean, std = series.mean(), series.std()
    q25, q75 = series.quantile([0.25, 0.75])
    iqr = q75 - q25
    lower = max(mean - z_thresh * std, q25 - 1.5 * iqr)
    upper = min(mean + z_thresh * std, q75 + 1.5 * iqr)
    return lower, upper


# ── ANO002: ESD masking recovery ─────────────────────────────────────────────


def _generalized_esd(
    values: np.ndarray, alpha: float = 0.05
) -> Tuple[int, List[int], Optional[Tuple[float, float, float]]]:
    """
    Rosner's Generalized ESD test: estimated number of outliers and which
    positions (into ``values``) they are.

    Returns ``(count, positions, bound)``.

    ``positions`` has length ``count`` and is ordered most-extreme-first
    (Rosner's removal order); it is a *subset* of every position the
    procedure considered removing (up to ``_ESD_MAX_FRACTION * n``),
    specifically the prefix the test statistic actually judged significant.

    ``bound`` is kept for backward-compatible unpacking (callers that only
    use ``count``/``positions`` and ignore it, and the ``test_masking_...``
    monkeypatch in tests/test_point.py, are unaffected either way) but is
    deliberately always ``None`` now -- see the "why not a single band"
    note below. Do not resurrect a non-``None`` value here without also
    reading that note; an earlier version of this function computed one
    (from ``step_means[count-1]``/``step_stds[count-1]``/
    ``criticals[count-1]``, the scale at the step that judged the
    least-extreme flagged point) and it was wrong.

    Why not a single band: Rosner's test is a *sequential* procedure. At
    each step it removes the single most extreme *remaining* point and
    recomputes mean/std before testing the next; the number of outliers is
    then read off as the *last* step index ``j`` for which the test
    statistic ``R_j`` exceeded the critical value ``lambda_j`` -- and every
    point removed up to and including step ``j`` is declared an outlier,
    whether or not that specific point's own ``R_i`` individually cleared
    ``lambda_i`` at its own step. Under heavy, roughly-symmetric
    contamination (verified: n=200, 30% contamination at magnitude 4.0,
    seed 4 in the parameter sweep documented in remediate.py's
    ``_esd_masking_repair``), ``R_i`` is *not* monotonic in removal order --
    it can be lower for early (most extreme by raw magnitude) removals than
    for later, less extreme ones, because std shrinks faster than the
    removed values' magnitude falls as contamination is peeled off. That
    means no single ``(mean, std, critical)`` triple -- not the boundary
    step's, not the "fully cleaned" post-removal step's -- is guaranteed to
    exclude every flagged point: empirically, both were tried and both left
    3 of 62 flagged points inside the band on that fixture. There is no
    per-point fix either, for the same reason: some flagged points never
    individually cleared their own step's critical value, so there is no
    data-derived boundary that is simultaneously "outside" for that point
    and consistent with why ESD flagged it. See
    ``remediate.py``'s ``_esd_masking_repair`` for how repair handles this
    (NaN instead of an invented clip bound).

    Used by ``audit_point_anomalies`` (via ``esd_masking_recovery`` below,
    shared with ``remediate.py``'s repair step): ``count`` alone drives
    ``masking_suspected``; when ``masking_suspected`` is True, ``positions``
    is folded into the points ANO002 flags and into what
    ``apply_fixes(outliers=...)`` repairs.

    Why it is here: the z-score half of ANO002 goes blind under heavy
    contamination, because the outliers inflate the standard deviation that
    judges them. That makes ``agreement_count`` drop to zero exactly when
    contamination is worst, indistinguishable from a harmlessly skewed column.
    ESD removes the most extreme point and *recomputes* the mean and standard
    deviation before testing the next, so masking cannot occur by construction.

    Measured against 1,000 clean points with outliers planted at 10 sigma: exact
    at every level (1, 5, 20, 50, 150, 300), and 0 on clean Gaussian data where
    the IQR rule reports 10 false positives.

    Reference: Rosner, B. (1983), "Percentage Points for a Generalized ESD
    Many-Outlier Procedure", Technometrics 25(2), 165-172.
    """
    n = len(values)
    if n < _ESD_MIN_OBS:
        return 0, [], None

    max_outliers = max(1, int(_ESD_MAX_FRACTION * n))

    # criticals[i] (0-indexed; i+1 is the removal step) is a deterministic
    # function of n, i, and alpha alone -- it never depends on the data or
    # on `work`. The original loop called scipy.stats.t.ppf as a *scalar*
    # call once per removal step (up to max_outliers = 0.4*n times per
    # column), and each call carries real per-call dispatch/validation
    # overhead on top of the actual inversion -- on a 3279-row, 26-column
    # benchmark this ppf loop alone accounted for roughly two-thirds of
    # audit_point_anomalies' total runtime (profiled: ~6.9s of a 12.1s
    # scan). Computing the whole criticals array in one vectorized call
    # before the removal loop starts is ~100x faster for this piece and
    # produces bit-identical values (verified against the original
    # elementwise loop across n = 50/500/3279/10000).
    i_arr = np.arange(1, max_outliers + 1)
    p_arr = 1.0 - alpha / (2.0 * (n - i_arr + 1))
    dof_arr = n - i_arr - 1
    t_arr = stats.t.ppf(p_arr, dof_arr)
    criticals = (
        (n - i_arr) * t_arr / np.sqrt((n - i_arr - 1 + t_arr**2) * (n - i_arr + 1))
    )

    # This part is inherently sequential -- Rosner's test removes the most
    # extreme point and *recomputes* mean/std before judging the next one,
    # which is the whole point (see the docstring above) -- so it stays a
    # plain Python loop over already-vectorized per-step numpy ops.
    #
    # idx_map tracks, in parallel with `work`, which original position each
    # entry of `work` came from -- so that when `work`'s worst point is
    # removed, we can record *which row of `values`* that was, not just that
    # a point was removed. step_means/step_stds record the scale used to
    # judge each step, so the step that actually decided `count` can be
    # recovered afterward without a second pass over the removal loop.
    work = values.astype(float, copy=True)
    idx_map = np.arange(n)
    test_statistics = []
    removed_positions: List[int] = []

    for idx in range(max_outliers):
        if len(work) < 3:
            break
        mean, std = work.mean(), work.std(ddof=1)
        if std == 0 or not np.isfinite(std):
            break

        deviation = np.abs(work - mean)
        worst = int(deviation.argmax())
        test_statistics.append(deviation[worst] / std)
        removed_positions.append(int(idx_map[worst]))
        work = np.delete(work, worst)
        idx_map = np.delete(idx_map, worst)

    estimated = 0
    for i in range(len(test_statistics)):
        if test_statistics[i] > criticals[i]:
            estimated = i + 1

    if estimated == 0:
        return 0, [], None

    # No `bound` (see docstring): a single (or per-point) scale claiming to
    # be "what ESD used" cannot be derived correctly here. Always None.
    return estimated, removed_positions[:estimated], None


class EsdRecovery(NamedTuple):
    """Result of ``esd_masking_recovery``. See that function's docstring."""

    masking_suspected: bool
    n_esd: Optional[int]
    esd_positions: List[int]


def esd_masking_recovery(
    series: pd.Series, z_mask: pd.Series, iqr_mask: pd.Series
) -> EsdRecovery:
    """
    Decide whether ANO002's z-score rule was blinded by masking and, if so,
    what ESD found instead: shared by ``audit_point_anomalies`` (detection)
    and ``remediate.py`` (repair), so the two cannot disagree about which
    points a masked column's ANO002 finding actually covers. Before this was
    centralized, ``masking_suspected`` and the points it implies were computed
    independently in each place, which is exactly the kind of drift this
    module's docstring exists to prevent (see CHANGELOG [0.5.0]).

    "Ambiguous" (the z-score rule finds nothing, IQR finds something) is
    consulted first, cheaply; ESD (O(k*n)) only runs when it might change the
    answer. ``masking_suspected`` is ``n_esd > n_iqr * 0.5`` -- a heuristic
    multiplier, not one derived from the ESD/Rosner literature or validated
    against a labeled contamination benchmark; it distinguishes real masking
    from an ordinarily skewed column, where ESD and IQR roughly agree.

    Returns
    -------
    EsdRecovery
        ``masking_suspected`` : bool
        ``n_esd`` : Optional[int] -- ESD's own outlier count, or ``None`` if
            not ambiguous (nothing to disambiguate).
        ``esd_positions`` : List[int] -- positions ESD flagged, or ``[]``
            unless ``masking_suspected`` is True. Deliberately gated on
            ``masking_suspected``, not just ``ambiguous``: a bare "z found
            nothing, IQR found something" is also the ordinary signature of a
            harmlessly skewed column, and ESD's own count relative to IQR's is
            what tells the two apart. Unconditionally trusting ESD's raw
            positions regardless of that check would fold points into every
            skewed-but-clean column too.

    No ``esd_bound``/clip-band field: an earlier version of this returned
    one for ``remediate.py``'s "clip" repair mode to clip ESD-recovered
    points to. It was statistically unsound -- see ``_generalized_esd``'s
    docstring for why no single data-derived band is guaranteed consistent
    with which points Rosner's sequential test actually flags. Repair
    instead NaNs these specific points regardless of ``outliers="clip"``;
    see ``remediate.py``'s ``_esd_masking_repair``.
    """
    n_zscore = int(z_mask.sum())
    n_iqr = int(iqr_mask.sum())
    ambiguous = n_zscore == 0 and n_iqr > 0

    if not ambiguous:
        return EsdRecovery(False, None, [])

    n_esd, esd_positions, _ = _generalized_esd(series.to_numpy(dtype=float))
    masking_suspected = n_esd > n_iqr * 0.5

    if not masking_suspected:
        return EsdRecovery(False, n_esd, [])

    return EsdRecovery(True, n_esd, esd_positions)


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
