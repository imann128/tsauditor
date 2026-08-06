import pandas as pd
from tsauditor.report.summary import Issue, CRITICAL, WARNING
from tsauditor.profiler._common import consecutive_run_lengths


def audit_frequency(df: pd.DataFrame, domain: str = None) -> list:
    """
    Audits time-series indices for duplicates, extreme gaps, and gap clustering.

    Parameters:
        df (pd.DataFrame): Time-series DataFrame with a DatetimeIndex.
        domain (str): Domain context ('finance', 'sensor', or None).

    Returns:
        list: List of Issue objects describing discovered data quality issues.
    """
    issues = []

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a pd.DatetimeIndex")

    if df.empty:
        return issues

    # 1. Check duplicate timestamps -> PRF004 CRITICAL
    if df.index.duplicated().any():
        duplicate_mask = df.index.duplicated(keep=False)
        duplicate_timestamps = df.index[duplicate_mask].unique()

        # Distinguish a genuine duplication bug from panel/long-format data.
        # In a panel every timestamp legitimately repeats once per entity, so
        # the repeat count is uniform and > 1 across (nearly) the whole index.
        # Flagging that as broken alignment is unhelpful: the user needs to be
        # told about group_col=, not told their data is corrupt.
        counts = df.index.value_counts()
        looks_like_panel = (
            len(counts) > 1 and counts.min() > 1 and counts.nunique() <= 2
        )

        description = "Duplicate timestamps detected in the index. Chronological alignment broken."
        if looks_like_panel:
            description += (
                f" Every timestamp repeats {int(counts.min())}-{int(counts.max())} "
                f"times, which is the shape of panel (long-format) data rather "
                f"than a duplication bug. If these rows are separate entities, "
                f"pass group_col='<your entity column>' so each entity is "
                f"audited as its own time series."
            )

        issues.append(
            Issue(
                module="profiler",
                code="PRF004",
                severity=CRITICAL,
                description=description,
                column=None,
                evidence={
                    "duplicate_count": int(duplicate_mask.sum()),
                    "examples": [
                        ts.strftime("%Y-%m-%d %H:%M:%S")
                        for ts in duplicate_timestamps[:5]
                    ],
                    "looks_like_panel": bool(looks_like_panel),
                    "repeats_per_timestamp": [int(counts.min()), int(counts.max())],
                },
            )
        )
        # Drop duplicates to ensure subsequent gap math is valid
        df = df[~df.index.duplicated(keep="first")]

    df_sorted = df.sort_index()

    # 2. Calculating consecutive gaps with clean index reset
    gap_days = (
        pd.Series(df_sorted.index)
        .diff()
        .dropna()
        .dt.total_seconds()
        .div(86400)
        .reset_index(drop=True)
    )

    if gap_days.empty:
        return issues

    median_gap = gap_days.median()

    # 3. Finding maximum_gap threshold based on domain.
    #
    # Deliberately two-way, not three-way like the anomaly presets: the
    # non-finance branch is already a *relative*, self-calibrating threshold
    # (3x the series' own median gap), which adapts to whatever the actual
    # sampling cadence is -- exactly what sensor data needs, since its
    # cadence varies from sub-second to hourly depending on the device, and
    # no single absolute day-count would work across that range. Finance is
    # the one domain that gets an absolute constant instead, specifically
    # because trading calendars have a known, bounded gap structure
    # (weekends/holidays, ~1-4 days) that a relative multiplier would handle
    # less predictably. A sensor-specific branch would need its own relative
    # multiplier (not 3.0x), and there is no measured basis for a different
    # number yet -- see audit_missing's cluster_threshold docstring for the
    # related, less defensible version of this gap.
    if domain == "finance":
        maximum_gap_threshold = 5.0
    else:
        maximum_gap_threshold = 3.0 * median_gap if median_gap > 0 else 1.0

    # 4. Flagging Individual large gaps -> PRF001 WARNING
    large_gap_mask = gap_days >= maximum_gap_threshold

    if large_gap_mask.any():
        # gap_days[i] is the gap ending at df_sorted row i+1, so i+1 locates
        # it. gap_days always has length len(df_sorted) - 1, so i+1 can never
        # reach len(df_sorted); no bounds guard is needed here.
        large_gap_indices = large_gap_mask[large_gap_mask].index
        gap_locations = df_sorted.index[large_gap_indices + 1]

        issues.append(
            Issue(
                module="profiler",
                code="PRF001",
                severity=WARNING,
                description=f"Large missing data gaps detected exceeding the threshold of {maximum_gap_threshold:.1f} days.",
                column=None,
                evidence={
                    "gap_count": int(large_gap_mask.sum()),
                    "maximum_gap_days": float(gap_days.max()),
                    "locations": [
                        ts.strftime("%Y-%m-%d %H:%M:%S") for ts in gap_locations[:5]
                    ],
                },
            )
        )

    # 5. Detect gap clusters through run-length -> PRF005 WARNING
    is_large_gap = large_gap_mask.astype(int).values
    run_starts, run_ends, run_lengths = consecutive_run_lengths(is_large_gap)
    cluster_runs = run_lengths >= 2

    if cluster_runs.any():
        total_clusters = int(cluster_runs.sum())
        # Same +1 mapping as above, same reason no bounds guard is needed:
        # run_starts indexes into is_large_gap, which has length
        # len(df_sorted) - 1, so cluster_starts + 1 can never reach
        # len(df_sorted).
        cluster_starts = run_starts[cluster_runs]
        cluster_locations = df_sorted.index[cluster_starts + 1]

        issues.append(
            Issue(
                module="profiler",
                code="PRF005",
                severity=WARNING,
                description="Clustered gap sequences detected. Missing data points are systematically bundled together.",
                column=None,
                evidence={
                    "cluster_count": total_clusters,
                    "max_consecutive_gaps": int(run_lengths.max()),
                    "cluster_start_locations": [
                        ts.strftime("%Y-%m-%d %H:%M:%S") for ts in cluster_locations[:5]
                    ],
                },
            )
        )

    return issues
