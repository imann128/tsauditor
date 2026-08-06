import numpy as np
import pandas as pd
from tsauditor.report.summary import Issue, WARNING, CRITICAL
from tsauditor.profiler._common import consecutive_run_lengths
from tsauditor.utils.validation import ensure_sorted_datetime_index

# Minimum observations the leakage detectors require before they will score a
# column (audit_equivalence, audit_correlation_leakage, audit_temporal_leakage
# and audit_combination_leakage all default to this). PRF007 reports when
# discarding non-finite values drops a column below it, because at that point
# the column is silently skipped by those checks rather than merely degraded.
_LEAKAGE_MIN_OBS = 30


def audit_non_finite(df: pd.DataFrame) -> list:
    """
    Audit numeric columns for infinite values (PRF007).

    Why this is a separate check from missingness
    ---------------------------------------------
    ``np.inf`` is not a missing value and is not an outlier. ``isna()`` is False
    for it, so PRF002 and PRF006 never see it, and every anomaly and leakage
    detector in this library quietly replaces it with NaN on its own working
    copy so its arithmetic does not break. The result before this check existed
    was that an inf was reported by nothing and repaired by nothing: a user
    could run ``scan()``, see no relevant issue, run ``fix()``, and still hand
    infinities to their model.

    Why there is no rate threshold
    ------------------------------
    PRF006 needs a threshold because some missingness is normal and the question
    is how much is too much. That question does not arise here. An infinity is
    never a measurement; it is the residue of a division by zero, an overflow, or
    a log of zero somewhere upstream. One is a defect, so the threshold is one.
    This is deliberate rather than an oversight, and it is why PRF007 takes no
    threshold parameter.

    Why CRITICAL
    ------------
    Matching PRF004 (duplicate timestamps), and for the same reason: it
    invalidates other checks rather than merely describing the data. A single inf
    makes a column's mean inf and its standard deviation NaN, so any statistic
    computed on the raw column is meaningless, and scikit-learn raises at
    ``fit`` time rather than degrading gracefully.

    Evidence
    --------
    ``n_finite_remaining`` is the count the other detectors actually work with,
    and ``below_leakage_min_obs`` reports whether that count falls under the 30
    observations the leakage checks require. That second key matters more than
    the raw count: below it, the column is not merely noisier, it is skipped
    entirely by LEK001, LEK002, LEK003 and LEK005 with no message.

    Parameters
    ----------
    df : pd.DataFrame
        Time-series DataFrame with a DatetimeIndex.

    Returns
    -------
    list
        One PRF007 Issue per affected column.
    """
    issues = []

    # "first_occurrence" below reports df.index[argmax(isinf)] -- the row at
    # the first *position*, not the first chronological timestamp. On an
    # unsorted-but-valid DatetimeIndex those differ, silently mislabeling
    # which occurrence is "first". See ensure_sorted_datetime_index's
    # docstring.
    df = ensure_sorted_datetime_index(df, "audit_non_finite")

    if df.empty:
        return issues

    for col in df.select_dtypes(include=[np.number]).columns:
        values = df[col].to_numpy(dtype=float, copy=False)

        # isinf is False for NaN, so the two categories do not overlap and the
        # counts below are additive with the PRF002/PRF006 missing counts.
        pos = int((values == np.inf).sum())
        neg = int((values == -np.inf).sum())
        total = pos + neg
        if total == 0:
            continue

        finite_remaining = int(np.isfinite(values).sum())
        first_pos = int(np.argmax(np.isinf(values)))

        issues.append(
            Issue(
                module="profiler",
                code="PRF007",
                severity=CRITICAL,
                description=(
                    f"Column '{col}' contains {total} infinite value(s). These are "
                    f"not missing values and not outliers: every statistic computed "
                    f"on the raw column is invalid, and the anomaly and leakage "
                    f"checks discard them before scoring."
                ),
                column=col,
                evidence={
                    "non_finite_count": total,
                    "positive_inf_count": pos,
                    "negative_inf_count": neg,
                    "non_finite_percentage": round(100.0 * total / len(values), 4),
                    "n_finite_remaining": finite_remaining,
                    "below_leakage_min_obs": bool(finite_remaining < _LEAKAGE_MIN_OBS),
                    "leakage_min_obs": _LEAKAGE_MIN_OBS,
                    "first_occurrence": df.index[first_pos].strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                },
            )
        )

    return issues


def audit_missing(
    df: pd.DataFrame,
    cluster_threshold: int = None,
    missing_rate_threshold: float = 0.30,
    domain: str = None,
) -> list:
    """
    Audit individual columns for systematic missing value clusters and high
    missing rates (PRF002, PRF005, PRF006).

    Parameters
    ----------
    df : pd.DataFrame
        Time-series DataFrame with a DatetimeIndex.
    cluster_threshold : int, optional
        Minimum consecutive NaNs to count as a cluster. If None, derived
        automatically from domain: 5 for "finance", 3 otherwise (including
        "sensor" -- see Notes).
    missing_rate_threshold : float, default 0.30
        Proportion threshold (0.0 to 1.0) above which a column is flagged
        for high missingness.
    domain : str, optional
        Domain context ('finance', 'sensor', or None).

    Returns
    -------
    list
        List of Issue objects describing missing value anomalies.

    Notes
    -----
    ``cluster_threshold``'s domain resolution only special-cases "finance";
    "sensor" silently falls into the same default as domain=None. Unlike
    ``audit_frequency``'s ``maximum_gap_threshold`` (which is *relative* to
    the series' own median gap and so already adapts to any sampling
    cadence), ``cluster_threshold`` is a flat row count. "3 consecutive
    missing rows" means something very different for a 1-second sensor feed
    than for daily data, so this genuinely could use a sampling-rate-aware
    default rather than a domain-keyed one. No sensor-specific constant has
    been added here, deliberately: picking a number without a measured basis
    would be an unvalidated heuristic dressed up as domain expertise, the
    same failure mode flagged elsewhere in this codebase (see
    ``audit_point_anomalies``'s ``masking_suspected`` Notes). If this needs
    fixing, the more defensible fix is likely making the default scale with
    the series' inferred sampling frequency (as ``audit_frequency`` already
    does via the series' own median gap), not adding a guessed "sensor"
    branch.
    """
    issues = []

    # Cluster/consecutive-run detection (PRF002/PRF005) is inherently
    # positional -- consecutive_run_lengths walks row-to-row. See
    # ensure_sorted_datetime_index's docstring.
    df = ensure_sorted_datetime_index(df, "audit_missing")

    if df.empty:
        return issues

    # Resolve cluster_threshold from domain if not explicitly provided.
    # See the Notes above: this is a known, open asymmetry, not an oversight
    # papered over with a guessed constant.
    if cluster_threshold is None:
        cluster_threshold = 5 if domain == "finance" else 3

    # Process numeric columns exclusively
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        series = df[col]
        total_rows = len(series)
        missing_count = int(series.isna().sum())

        if missing_count == 0:
            continue

        missing_pct = float(missing_count / total_rows)

        # Check PRF006: High overall missing rate
        if missing_pct >= missing_rate_threshold:
            issues.append(
                Issue(
                    module="profiler",
                    code="PRF006",
                    severity=WARNING,
                    description=f"Column '{col}' exhibits a high missing data rate.",
                    column=col,
                    evidence={
                        "missing_count": missing_count,
                        "missing_percentage": round(missing_pct * 100, 2),
                        "threshold_percentage": round(missing_rate_threshold * 100, 2),
                    },
                )
            )

        # Check PRF002: Vectorized RLE for consecutive NaN clusters
        is_missing = series.isna().astype(int).values
        run_starts, run_ends, run_lengths = consecutive_run_lengths(is_missing)

        # Filter for runs that violate our structural cluster ceiling
        cluster_mask = run_lengths >= cluster_threshold

        if cluster_mask.any():
            total_clusters = int(cluster_mask.sum())
            longest_run = int(run_lengths.max())

            # Extract the first matching run sequence position
            first_cluster_idx = run_starts[cluster_mask][0]
            first_occurrence_ts = df.index[first_cluster_idx]

            issues.append(
                Issue(
                    module="profiler",
                    code="PRF002",
                    severity=WARNING,
                    description=f"Column '{col}' contains clustered missing value sequences indicating an outage.",
                    column=col,
                    evidence={
                        "missing_percentage": round(missing_pct * 100, 2),
                        "longest_consecutive_run": longest_run,
                        "cluster_count": total_clusters,
                        "first_occurrence": first_occurrence_ts.strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
                        "cluster_threshold": cluster_threshold,
                    },
                )
            )

    return issues
