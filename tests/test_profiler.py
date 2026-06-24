import pytest
import pandas as pd
from tsauditor.profiler.frequency import audit_frequency
from tsauditor.report.summary import CRITICAL, WARNING


def test_clean_financial_df_no_issues(clean_financial_df):
    # Case 1 — Clean daily financial df -> no issues returned.
    issues = audit_frequency(clean_financial_df, domain="finance")
    assert len(issues) == 0
    assert isinstance(issues, list)


def test_duplicate_timestamps_critical(clean_financial_df):
    # Case 2 — Df with duplicate timestamps -> PRF004 flagged as CRITICAL.
    df = clean_financial_df.copy()
    # Duplicate the first timestamp by appending its row back to the dataframe
    first_row = df.iloc[[0]]
    df_with_dups = pd.concat([first_row, df])

    issues = audit_frequency(df_with_dups, domain="finance")

    # Assert PRF004 is flagged inside the returned list
    dup_issues = [i for i in issues if i.code == "PRF004"]
    assert len(dup_issues) == 1

    issue = dup_issues[0]
    assert issue.severity == CRITICAL
    assert issue.module == "profiler"
    assert "duplicate_count" in issue.evidence
    assert "examples" in issue.evidence


def test_single_large_gap_finance(clean_financial_df):
    """Case 3 — Df with a 10-day gap -> PRF001 flagged, finance domain."""
    df = clean_financial_df.copy().sort_index()

    # Introducing a 10-day structural calendar gap by dropping rows and finding a point mid-dataframe to cut out rows
    mid_idx = len(df) // 2
    target_date = df.index[mid_idx]

    # Drop rows that fall within a 9-day window following the target date
    drop_window = (df.index > target_date) & (
        df.index <= target_date + pd.Timedelta(days=9)
    )
    df_with_gap = df[~drop_window]

    issues = audit_frequency(df_with_gap, domain="finance")

    gap_issues = [i for i in issues if i.code == "PRF001"]
    assert len(gap_issues) == 1

    issue = gap_issues[0]
    assert issue.severity == WARNING
    assert issue.evidence["maximum_gap_days"] >= 10.0
    assert len(issue.evidence["locations"]) > 0


def test_clustered_gaps_run_length(clean_financial_df):
    # Case 4 — Df with more than 3 consecutive large gaps -> PRF005 flagged.
    df = clean_financial_df.copy().sort_index()

    # Create 3 distinct consecutive large gaps (each >= 5 days) by cutting gaps
    # separated by isolated single rows.
    mid_idx = len(df) // 2
    base_ts = df.index[mid_idx]

    # making distinct synthetic timestamps with consecutive large step gaps
    t0 = base_ts
    t1 = t0 + pd.Timedelta(days=6)  # Gap 1 (6 days)
    t2 = t1 + pd.Timedelta(days=6)  # Gap 2 (6 days)
    t3 = t2 + pd.Timedelta(days=6)  # Gap 3 (6 days)

    # extracting surrounding blocks and concatenate with the explicitly gapped timeline
    part1 = df.iloc[:mid_idx]
    part2 = df.iloc[mid_idx + 1 : mid_idx + 4].copy()
    part2.index = [t1, t2, t3]

    df_clustered = pd.concat([part1, part2]).sort_index()

    issues = audit_frequency(df_clustered, domain="finance")

    cluster_issues = [i for i in issues if i.code == "PRF005"]
    assert len(cluster_issues) == 1

    issue = cluster_issues[0]
    assert issue.severity == WARNING
    assert issue.evidence["cluster_count"] >= 1
    assert issue.evidence["max_consecutive_gaps"] >= 3


def test_single_row_df():
    # Case 5 — Single row df -> no issue
    dates = pd.date_range("2026-05-22", periods=1, freq="B")
    df_single = pd.DataFrame({"value": [100.0]}, index=dates)

    issues = audit_frequency(df_single, domain="finance")
    assert len(issues) == 0
    assert isinstance(issues, list)


def test_non_datetime_index_raises_value_error():
    # Case 6 — Non-DatetimeIndex df -> raises ValueError.
    df_bad_index = pd.DataFrame({"value": [1, 2, 3]}, index=[0, 1, 2])

    with pytest.raises(ValueError, match="DataFrame index must be a pd.DatetimeIndex"):
        audit_frequency(df_bad_index, domain="finance")


def test_sensor_domain_median_threshold(sensor_df):
    """Case 7 — Sensor domain with large gap -> PRF001 using 3x median threshold."""
    df = sensor_df.copy().sort_index()
    """
    # Hourly frequency means the baseline median gap is 1/24 days (0.0416 days) so
     3x median threshold would be roughly 0.125 days.
     Injecting a 1-day (24 hours) gap, which easily breaches 3x median but is below the finance 5-day limit.
    """
    mid_idx = len(df) // 2
    drop_mask = (df.index > df.index[mid_idx]) & (
        df.index <= df.index[mid_idx] + pd.Timedelta(hours=23)
    )
    df_sensor_gap = df[~drop_mask]

    # Pass domain=None or domain="sensor" to check adaptive threshold behavior
    issues = audit_frequency(df_sensor_gap, domain="sensor")

    gap_issues = [i for i in issues if i.code == "PRF001"]
    assert len(gap_issues) == 1

    issue = gap_issues[0]
    assert issue.severity == WARNING
    assert issue.evidence["maximum_gap_days"] >= 1.0


def test_single_row_df_missing():
    """Single row DataFrame for frequency: returns empty list, no crash."""
    dates = pd.date_range("2026-01-01", periods=1, freq="D")
    df = pd.DataFrame({"val": [1.0]}, index=dates)
    issues = audit_frequency(df, domain="finance")
    assert isinstance(issues, list)
    assert len(issues) == 0
