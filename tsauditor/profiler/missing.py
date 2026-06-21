import numpy as np
import pandas as pd
from tsauditor.report.summary import Issue, WARNING


def audit_missing(
    df: pd.DataFrame,
    cluster_threshold: int = None,
    missing_rate_threshold: float = 0.30,
    domain: str = None,
) -> list:
    """
    Audits individual columns for systematic missing value clusters and high missing rates.

    Parameters:
        df (pd.DataFrame): Time-series DataFrame with a DatetimeIndex.
        cluster_threshold (int): Minimum consecutive NaNs to count as a cluster.
                                 If None, derived automatically from domain.
        missing_rate_threshold (float): Proportion threshold (0.0 to 1.0) above which
                                        a column is flagged for high missingness.
        domain (str): Domain context ('finance', 'sensor', or None).

    Returns:
        list: List of Issue objects describing missing value anomalies.
    """
    issues = []

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a pd.DatetimeIndex")

    if df.empty:
        return issues

    # Resolve cluster_threshold from domain if not explicitly provided
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

        run_starts = np.where((is_missing[:-1] == 0) & (is_missing[1:] == 1))[0] + 1
        if len(is_missing) > 0 and is_missing[0] == 1:
            run_starts = np.insert(run_starts, 0, 0)

        run_ends = np.where((is_missing[:-1] == 1) & (is_missing[1:] == 0))[0] + 1
        if len(is_missing) > 0 and is_missing[-1] == 1:
            run_ends = np.append(run_ends, len(is_missing))

        run_lengths = run_ends - run_starts

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
