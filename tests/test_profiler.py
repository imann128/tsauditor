import pytest
import pandas as pd
from tsauditor.profiler.frequency import audit_frequency
from tsauditor.report.summary import CRITICAL, WARNING
import tsauditor.profiler.frequency as frequency
import tsauditor.profiler.missing as missing
import tsauditor.profiler._common as profiler_common


def test_frequency_and_missing_share_the_same_rle_function():
    """
    Structural guarantee: audit_frequency (PRF001/PRF005) and audit_missing
    (PRF002) must compute run lengths through the exact same function object
    in tsauditor.profiler._common, not independent copies of the same
    three-line numpy pattern. Mirrors
    test_detector_and_repair_share_the_same_threshold_and_mask_functions in
    tests/test_fix.py for the anomaly/remediate module -- same category of
    duplication, same fix. If this ever fails, someone has reintroduced a
    hand-copied run-length-encoding block instead of importing from _common.
    """
    assert frequency.consecutive_run_lengths is profiler_common.consecutive_run_lengths
    assert missing.consecutive_run_lengths is profiler_common.consecutive_run_lengths


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


def test_exactly_two_repeat_counts_still_looks_like_panel():
    """
    The panel-shape guard is `counts.nunique() <= 2`, allowing entities to
    differ by at most one row (e.g. a ragged panel where one entity has one
    fewer observation). Exactly 2 distinct repeat counts must still count as
    panel-shaped, not be dismissed as a duplication bug.
    """
    idx = pd.to_datetime(
        ["2024-01-01", "2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"]
    )
    df = pd.DataFrame({"value": range(5)}, index=idx)
    issues = audit_frequency(df, domain="finance")
    dup = next(i for i in issues if i.code == "PRF004")
    assert dup.evidence["looks_like_panel"] is True


def test_exactly_two_consecutive_gaps_forms_a_cluster():
    """
    The cluster guard is `run_lengths >= 2`, so exactly two consecutive large
    gaps must be reported as a cluster (PRF005), not just two isolated PRF001
    gaps.
    """
    base = pd.date_range("2023-01-01", periods=10, freq="D")
    idx = base.append(
        pd.DatetimeIndex(
            [base[-1] + pd.Timedelta(days=10), base[-1] + pd.Timedelta(days=20)]
        )
    )
    df = pd.DataFrame({"value": range(len(idx))}, index=idx)
    issues = audit_frequency(df, domain="finance")
    cluster_issues = [i for i in issues if i.code == "PRF005"]
    assert len(cluster_issues) == 1
    assert cluster_issues[0].evidence["max_consecutive_gaps"] == 2


def test_finance_gap_exactly_five_days_flags():
    """The finance threshold is `gap_days >= 5.0`; a gap of exactly 5 days
    must still flag, not be waved through as just under the limit."""
    base = pd.date_range("2023-01-01", periods=10, freq="D")
    idx = base.append(pd.DatetimeIndex([base[-1] + pd.Timedelta(days=5)]))
    df = pd.DataFrame({"value": range(len(idx))}, index=idx)
    issues = audit_frequency(df, domain="finance")
    assert any(i.code == "PRF001" for i in issues)


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


def test_non_finance_gap_multiplier_is_pinned_at_3x_median():
    """
    Pins the non-finance gap threshold at exactly 3x the median gap.

    test_sensor_domain_median_threshold only checks that a gap far above the
    threshold (23h against a ~0.125-day/3h threshold) gets flagged - a
    multiplier mutated to 5.0 or even 10.0 would still pass that test. This
    test builds a gap on each side of the 3x boundary (3.5x flags, 2.5x does
    not) so a multiplier mutated in either direction is caught.
    """
    base = pd.date_range("2023-01-01", periods=20, freq="h")  # 19 gaps of 1h
    # median gap stays 1h (0.041667 days) throughout, unaffected by one outlier

    # 3.5x median -> above a correct 3x threshold, still below a 4x threshold.
    idx_above = base.append(pd.DatetimeIndex([base[-1] + pd.Timedelta(hours=3.5)]))
    df_above = pd.DataFrame({"value": range(len(idx_above))}, index=idx_above)
    issues_above = audit_frequency(df_above, domain=None)
    assert any(i.code == "PRF001" for i in issues_above)

    # 2.5x median -> below a correct 3x threshold, above a 2x threshold.
    idx_below = base.append(pd.DatetimeIndex([base[-1] + pd.Timedelta(hours=2.5)]))
    df_below = pd.DataFrame({"value": range(len(idx_below))}, index=idx_below)
    issues_below = audit_frequency(df_below, domain=None)
    assert not any(i.code == "PRF001" for i in issues_below)


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
