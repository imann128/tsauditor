import numpy as np
import pandas as pd
from tsauditor.report.summary import Issue, CRITICAL, WARNING


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

        issues.append(
            Issue(
                module="profiler",
                code="PRF004",
                severity=CRITICAL,
                description="Duplicate timestamps detected in the index. Chronological alignment broken.",
                column=None,
                evidence={
                    "duplicate_count": int(duplicate_mask.sum()),
                    "examples": [
                        ts.strftime("%Y-%m-%d %H:%M:%S")
                        for ts in duplicate_timestamps[:5]
                    ],
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

    # 3. Finding maximum_gap threshold based on domain
    if domain == "finance":
        maximum_gap_threshold = 5.0
    else:
        maximum_gap_threshold = 3.0 * median_gap if median_gap > 0 else 1.0

    # 4. Flagging Individual large gaps -> PRF001 WARNING
    large_gap_mask = gap_days >= maximum_gap_threshold

    if large_gap_mask.any():
        large_gap_indices = large_gap_mask[large_gap_mask].index
        # Boundary guard to prevent index out of bounds on the last entry
        safe_indices = [i + 1 for i in large_gap_indices if i + 1 < len(df_sorted)]
        gap_locations = df_sorted.index[safe_indices]

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

    run_starts = np.where((is_large_gap[:-1] == 0) & (is_large_gap[1:] == 1))[0] + 1
    if len(is_large_gap) > 0 and is_large_gap[0] == 1:
        run_starts = np.insert(run_starts, 0, 0)

    run_ends = np.where((is_large_gap[:-1] == 1) & (is_large_gap[1:] == 0))[0] + 1
    if len(is_large_gap) > 0 and is_large_gap[-1] == 1:
        run_ends = np.append(run_ends, len(is_large_gap))

    run_lengths = run_ends - run_starts
    cluster_runs = run_lengths >= 2

    if cluster_runs.any():
        total_clusters = int(cluster_runs.sum())
        cluster_starts = run_starts[cluster_runs]
        # Boundary guard applied to cluster start indexing mapping
        safe_cluster_indices = [i + 1 for i in cluster_starts if i + 1 < len(df_sorted)]
        cluster_locations = df_sorted.index[safe_cluster_indices]

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
