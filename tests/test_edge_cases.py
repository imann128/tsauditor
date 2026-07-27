"""
Edge cases: degenerate columns and awkward time indices.

Two things are being pinned here.

1. **No detector may crash on a degenerate column.** A single all-NaN,
   all-inf, constant, or non-numeric column must never abort someone's whole
   scan. Detectors skip what they cannot score; they do not raise.

2. **The time index is handled correctly in the awkward cases** — unsorted
   input, timezones, DST, leap days, sub-second sampling, string labels — and
   is *refused* in the one case where silent coercion would be dangerous.

Where behaviour is a deliberate trade-off rather than obviously right, the
test says so in a comment, so a future reader knows it was a decision.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import tsauditor as tsa
from tsauditor.anomaly.contextual import audit_contextual_anomalies
from tsauditor.anomaly.point import audit_point_anomalies
from tsauditor.leakage.asof import audit_asof_leakage
from tsauditor.leakage.correlation import audit_correlation_leakage
from tsauditor.leakage.equivalence import audit_equivalence
from tsauditor.leakage.temporal import audit_temporal_leakage
from tsauditor.profiler.frequency import audit_frequency
from tsauditor.profiler.missing import audit_missing
from tsauditor.profiler.stationarity import audit_stationarity

N = 60
IDX = pd.date_range("2024-01-01", periods=N, freq="D")

UNTARGETED_DETECTORS = [
    audit_missing,
    audit_frequency,
    audit_stationarity,
    audit_point_anomalies,
    audit_contextual_anomalies,
]

TARGETED_DETECTORS = [
    audit_equivalence,
    audit_correlation_leakage,
    audit_temporal_leakage,
]


def _frame(**cols) -> pd.DataFrame:
    return pd.DataFrame(cols, index=IDX)


# ── Degenerate columns must not crash any detector ───────────────────────────

DEGENERATE_COLUMNS = {
    "all_nan": [np.nan] * N,
    "single_value_rest_nan": [np.nan] * (N - 1) + [1.0],
    "all_positive_inf": [np.inf] * N,
    "all_negative_inf": [-np.inf] * N,
    "mixed_inf": ([np.inf, -np.inf] * (N // 2)),
    "constant": [5.0] * N,
    "constant_zero": [0.0] * N,
    "all_nan_but_two": [np.nan] * (N - 2) + [1.0, 2.0],
}


@pytest.mark.parametrize(
    "name, values", DEGENERATE_COLUMNS.items(), ids=list(DEGENERATE_COLUMNS)
)
@pytest.mark.parametrize("detector", UNTARGETED_DETECTORS, ids=lambda d: d.__name__)
def test_degenerate_column_does_not_crash_detector(name, values, detector):
    """A degenerate column is skipped, never raised on."""
    df = _frame(degenerate=values, healthy=np.arange(float(N)))
    issues = detector(df)
    assert isinstance(issues, list)


@pytest.mark.parametrize(
    "name, values", DEGENERATE_COLUMNS.items(), ids=list(DEGENERATE_COLUMNS)
)
@pytest.mark.parametrize("detector", TARGETED_DETECTORS, ids=lambda d: d.__name__)
def test_degenerate_feature_does_not_crash_leakage(name, values, detector):
    """Same, for the leakage detectors, which need a target."""
    rng = np.random.default_rng(0)
    df = _frame(degenerate=values, target=rng.normal(0, 1, N))
    issues = detector(df, target="target")
    assert isinstance(issues, list)


@pytest.mark.parametrize(
    "name, values", DEGENERATE_COLUMNS.items(), ids=list(DEGENERATE_COLUMNS)
)
def test_degenerate_column_does_not_crash_scan(name, values):
    """End-to-end: scan survives a degenerate column and still reports metadata."""
    df = _frame(degenerate=values, healthy=np.arange(float(N)))
    report = tsa.scan(df, run_stationarity=False)
    assert report.metadata["rows"] == N


# ── Specific NaN / inf semantics ─────────────────────────────────────────────


def test_all_nan_column_is_flagged_as_missing_not_as_anomalous():
    """
    An all-NaN column is a *missing data* problem, not an anomaly. It should
    raise PRF002/PRF006 and be silently skipped by the anomaly detectors,
    which have no values to score.
    """
    df = _frame(empty=[np.nan] * N, healthy=np.arange(float(N)))

    missing_codes = {i.code for i in audit_missing(df) if i.column == "empty"}
    assert missing_codes == {"PRF002", "PRF006"}

    assert [i for i in audit_point_anomalies(df) if i.column == "empty"] == []
    assert [i for i in audit_contextual_anomalies(df) if i.column == "empty"] == []


def test_inf_column_is_invisible_to_the_missing_check():
    """
    Documents a real gap: `inf` is not NaN, so audit_missing does not see an
    all-inf column at all, while the anomaly and leakage detectors *do* treat
    inf as missing and skip it. An all-inf column therefore produces no issue
    from any detector.

    This is a known limitation, pinned so a future change to it is deliberate
    rather than accidental.
    """
    df = _frame(broken=[np.inf] * N)

    assert audit_missing(df) == []
    assert audit_point_anomalies(df) == []
    assert audit_contextual_anomalies(df) == []


def test_inf_does_not_mask_a_genuine_outlier_in_the_same_column():
    """
    Regression: audit_point_anomalies used to compute mean/std over raw values,
    so a single inf made mean=inf and std=NaN. Every comparison then evaluated
    False and the column was skipped whole — hiding real outliers sitting
    alongside the inf. inf is now neutralised first, as in every other detector.
    """
    rng = np.random.default_rng(42)
    values = rng.normal(100, 5, N)
    values[10] = 500.0  # a genuine outlier
    values[20] = np.inf  # ...next to an inf

    issues = audit_point_anomalies(_frame(x=values))

    assert len(issues) == 1
    assert issues[0].code == "ANO002"
    assert issues[0].evidence["worst_value"] == 500.0
    assert np.isfinite(issues[0].evidence["max_zscore"])


def test_point_anomalies_emits_no_numpy_warnings_on_degenerate_input():
    """
    Degenerate columns must be skipped cleanly, not by letting NaN arithmetic
    fall through and emit RuntimeWarnings from numpy/pandas internals.
    """
    for values in ([np.inf] * N, [-np.inf] * N, [np.nan] * N, [5.0] * N):
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            assert audit_point_anomalies(_frame(x=values)) == []


def test_esd_disambiguates_a_zero_agreement_count():
    """
    A zero `agreement_count` has two opposite causes — harmless skew, or
    contamination heavy enough to blind the z-score — and the counts alone
    cannot tell them apart. The ESD diagnostic can, because it recomputes the
    scale after each removal so masking cannot occur.
    """
    rng = np.random.default_rng(3)
    base = rng.normal(0, 1, 1000)
    idx = pd.date_range("2024-01-01", periods=1000, freq="D")

    # Contaminated: z-score blind, but 50 real outliers are there.
    contaminated = base.copy()
    contaminated[:50] = 10.0
    evidence = audit_point_anomalies(
        pd.DataFrame({"x": contaminated}, index=idx), zscore_threshold=5.0
    )[0].evidence

    assert evidence["zscore_outlier_count"] == 0
    assert evidence["agreement_count"] == 0
    assert evidence["esd_outlier_count"] == 50
    assert evidence["masking_suspected"] is True

    # Clean Gaussian: the IQR rule fires on ordinary tails, but nothing is wrong.
    clean = audit_point_anomalies(
        pd.DataFrame({"x": base}, index=idx), zscore_threshold=5.0
    )[0].evidence

    assert clean["agreement_count"] == 0
    assert clean["esd_outlier_count"] == 0
    assert clean["masking_suspected"] is False


def test_esd_does_not_fire_on_skewed_data():
    """Skew must not be reported as masking."""
    rng = np.random.default_rng(9)
    idx = pd.date_range("2024-01-01", periods=1000, freq="D")

    for values in (rng.lognormal(0, 1, 1000), rng.exponential(1, 1000)):
        issues = audit_point_anomalies(
            pd.DataFrame({"x": values}, index=idx), zscore_threshold=5.0
        )
        assert issues[0].evidence["masking_suspected"] is False


def test_esd_is_skipped_when_the_rules_already_agree():
    """
    ESD is O(k*n) — ~27ms on 1,000 points. It is only computed for the
    ambiguous case, and reported as None otherwise.
    """
    rng = np.random.default_rng(3)
    values = rng.normal(0, 1, 1000)
    values[:20] = 10.0
    idx = pd.date_range("2024-01-01", periods=1000, freq="D")

    evidence = audit_point_anomalies(
        pd.DataFrame({"x": values}, index=idx), zscore_threshold=5.0
    )[0].evidence

    assert evidence["zscore_outlier_count"] == 20
    assert evidence["esd_outlier_count"] is None
    assert evidence["masking_suspected"] is False


def test_esd_does_not_change_what_gets_flagged():
    """
    The whole point of adding ESD as evidence rather than as a rule: flagging is
    driven by the z-score and IQR masks, exactly as before.
    """
    rng = np.random.default_rng(3)
    base = rng.normal(0, 1, 1000)
    idx = pd.date_range("2024-01-01", periods=1000, freq="D")

    for values in (base, np.r_[np.full(50, 10.0), base[50:]]):
        issues = audit_point_anomalies(
            pd.DataFrame({"x": values}, index=idx), zscore_threshold=5.0
        )
        evidence = issues[0].evidence
        # An issue is raised iff at least one rule fired.
        assert bool(issues) == (
            evidence["zscore_outlier_count"] > 0 or evidence["iqr_outlier_count"] > 0
        )


def test_constant_column_is_skipped_by_stationarity_not_crashed_on():
    """
    `adfuller` raises "Invalid input, x is constant". A constant column is
    trivially (degenerately) stationary, so it is skipped rather than allowed
    to abort the scan.
    """
    df = _frame(flat=[5.0] * N)
    assert [i for i in audit_stationarity(df) if i.column == "flat"] == []


def test_constant_column_is_reported_as_stuck():
    """
    The flip side: a constant column *is* one long stuck run, so ANO001 fires.
    This is the library's most common false positive (binary flags, regime
    indicators) and is documented as such — pinned here so the behaviour is
    known rather than surprising.
    """
    df = _frame(flat=[5.0] * N)
    stuck = [i for i in audit_contextual_anomalies(df) if i.code == "ANO001"]
    assert len(stuck) == 1
    assert stuck[0].evidence["max_stuck_duration"] == N


def test_inf_is_treated_as_missing_for_min_obs_purposes():
    """
    Equivalence needs >= min_obs (30) pairwise-complete rows. With 40 of 60
    feature values set to inf only 20 remain, so the column is skipped even
    though it would otherwise be a perfect leak.
    """
    rng = np.random.default_rng(0)
    change = rng.normal(0, 1, N)
    feature = change.copy()
    feature[:40] = np.inf

    df = _frame(target=(change > 0).astype(int), feature=feature)
    assert audit_equivalence(df, target="target") == []

    # Without the infs the same column is caught immediately.
    clean = _frame(target=(change > 0).astype(int), feature=change)
    assert len(audit_equivalence(clean, target="target")) == 1


@pytest.mark.parametrize("n_rows, expected", [(29, 0), (30, 1), (31, 1)])
def test_equivalence_min_obs_boundary(n_rows, expected):
    """min_obs=30 is inclusive: exactly 30 observations is enough."""
    rng = np.random.default_rng(1)
    idx = pd.date_range("2024-01-01", periods=n_rows, freq="D")
    change = rng.normal(0, 1, n_rows)
    df = pd.DataFrame(
        {"target": (change > 0).astype(int), "feature": change}, index=idx
    )
    assert len(audit_equivalence(df, target="target")) == expected


# ── Degenerate targets ───────────────────────────────────────────────────────


def test_constant_target_returns_no_leakage_issues():
    """A constant target has nothing to reproduce; every check returns []."""
    rng = np.random.default_rng(0)
    df = _frame(target=[1] * N, feature=rng.normal(0, 1, N))

    assert audit_equivalence(df, target="target") == []
    assert audit_correlation_leakage(df, target="target") == []
    assert audit_temporal_leakage(df, target="target") == []


def test_all_nan_target_returns_no_leakage_issues():
    df = _frame(target=[np.nan] * N, feature=np.arange(float(N)))
    assert audit_equivalence(df, target="target") == []
    assert audit_correlation_leakage(df, target="target") == []


def test_binary_categorical_target_is_encoded_and_caught():
    """A string 'up'/'down' target must work exactly like a 0/1 target."""
    rng = np.random.default_rng(2)
    change = rng.normal(0, 1, N)
    df = _frame(
        target=np.where(change > 0, "up", "down"),
        feature=np.where(change > 0, 1.0, 0.0),
    )
    issues = audit_equivalence(df, target="target")
    assert len(issues) == 1
    assert issues[0].evidence["metric"] == "auc"
    assert issues[0].evidence["auc"] == 1.0


def test_non_numeric_multiclass_target_raises():
    """Three string categories cannot be correlated — fail loudly, not silently."""
    rng = np.random.default_rng(0)
    df = _frame(target=["a", "b", "c"] * (N // 3), feature=rng.normal(0, 1, N))

    with pytest.raises(ValueError):
        audit_equivalence(df, target="target")
    with pytest.raises(ValueError):
        audit_correlation_leakage(df, target="target")


def test_missing_target_column_raises():
    df = _frame(feature=np.arange(float(N)))
    with pytest.raises(ValueError, match="not found"):
        audit_equivalence(df, target="does_not_exist")


# ── Time index edge cases ────────────────────────────────────────────────────


def test_scan_sorts_an_unsorted_index():
    """Rows arriving newest-first must be sorted before any check runs."""
    df = pd.DataFrame({"a": np.arange(float(N))}, index=IDX[::-1])
    report = tsa.scan(df, run_stationarity=False)
    assert report.metadata["time_start"] == str(IDX.min().date())
    assert report.metadata["time_end"] == str(IDX.max().date())


def test_numeric_index_is_refused_not_coerced():
    """
    A RangeIndex would be silently read as nanosecond epochs (all near 1970),
    quietly corrupting every gap and frequency result. Refusing is correct.
    """
    df = pd.DataFrame({"a": np.arange(5.0)})
    with pytest.raises(ValueError, match="numeric"):
        tsa.scan(df)


def test_string_date_index_is_coerced():
    """Date-like string labels are a genuine convenience case and are coerced."""
    df = pd.DataFrame(
        {"a": np.arange(5.0)},
        index=["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
    )
    report = tsa.scan(df, run_stationarity=False)
    assert report.metadata["time_start"] == "2024-01-01"


def test_empty_dataframe_raises():
    with pytest.raises(ValueError, match="empty"):
        tsa.scan(pd.DataFrame())


def test_single_row_produces_no_gap_issues():
    """One row means no gaps exist to measure; frequency is 'unknown'."""
    df = pd.DataFrame({"a": [1.0]}, index=IDX[:1])
    report = tsa.scan(df, run_stationarity=False)
    assert report.metadata["frequency"] == "unknown"
    assert audit_frequency(df) == []


def test_two_rows_do_not_crash():
    df = pd.DataFrame({"a": [1.0, 2.0]}, index=IDX[:2])
    report = tsa.scan(df, run_stationarity=False)
    assert report.metadata["rows"] == 2


def test_timezone_aware_index_is_supported():
    df = pd.DataFrame({"a": np.arange(float(N))}, index=IDX.tz_localize("UTC"))
    report = tsa.scan(df, run_stationarity=False)
    assert report.metadata["frequency"] == "daily"


def test_dst_transition_does_not_produce_spurious_gaps():
    """
    Spring-forward in US/Eastern (2024-03-10) drops a wall-clock hour. Because
    gap maths runs on UTC-absolute timestamps, the spacing stays uniform and no
    PRF001 should appear.
    """
    idx = pd.date_range("2024-03-09", periods=72, freq="h", tz="US/Eastern")
    df = pd.DataFrame({"a": np.arange(72.0)}, index=idx)
    assert [i for i in audit_frequency(df) if i.code == "PRF001"] == []


def test_leap_day_is_handled():
    idx = pd.date_range("2024-02-25", periods=10, freq="D")
    assert pd.Timestamp("2024-02-29") in idx
    df = pd.DataFrame({"a": np.arange(10.0)}, index=idx)
    report = tsa.scan(df, run_stationarity=False)
    assert report.metadata["frequency"] == "daily"
    assert [i for i in audit_frequency(df) if i.code == "PRF001"] == []


def test_sub_second_frequency_is_labelled_sub_daily():
    idx = pd.date_range("2024-01-01", periods=100, freq="100ms")
    df = pd.DataFrame({"a": np.arange(100.0)}, index=idx)
    report = tsa.scan(df, run_stationarity=False)
    assert report.metadata["frequency"] == "sub-daily"


def test_duplicate_timestamps_raise_prf004_critical():
    idx = IDX[: N // 2].repeat(2)
    df = pd.DataFrame({"a": np.arange(float(N))}, index=idx)
    dupes = [i for i in audit_frequency(df) if i.code == "PRF004"]
    assert len(dupes) == 1
    assert dupes[0].severity == "critical"
    assert dupes[0].column is None  # dataset-level, not per-column


def test_adaptive_gap_threshold_scales_with_sampling_rate():
    """
    With domain=None the gap threshold is 3x the median gap, so the same
    *shaped* outage is caught at any sampling rate. A 10-minute series and a
    daily series both flag a single 10-step hole.
    """
    for freq, step in (
        ("10min", pd.Timedelta(minutes=10)),
        ("D", pd.Timedelta(days=1)),
    ):
        base = pd.date_range("2024-01-01", periods=40, freq=freq)
        idx = base[:20].append(base[20:] + step * 10)  # one large hole
        df = pd.DataFrame({"a": np.arange(40.0)}, index=idx)
        gaps = [i for i in audit_frequency(df) if i.code == "PRF001"]
        assert len(gaps) == 1, f"no PRF001 at freq={freq}"


# ── As-of edge cases ─────────────────────────────────────────────────────────


def test_asof_ignores_nan_values():
    """A NaN cannot be 'used early' — there is no value to use."""
    df = _frame(cpi=[np.nan] * N)
    assert audit_asof_leakage(df, available_at={"cpi": pd.Timedelta(days=30)}) == []


def test_asof_misaligned_series_raises_rather_than_reporting_clean():
    """
    A Series indexed by anything other than df.index would silently produce
    zero violations — indistinguishable from clean data, and the worst possible
    failure mode for a leakage check. It must raise.
    """
    df = _frame(cpi=np.arange(float(N)))
    misaligned = pd.Series(IDX, index=pd.RangeIndex(N))
    with pytest.raises(ValueError, match="align"):
        audit_asof_leakage(df, available_at={"cpi": misaligned})


def test_asof_zero_and_negative_lag_are_not_violations():
    """Available at or before the row timestamp is exactly what we want."""
    df = _frame(cpi=np.arange(float(N)))
    assert audit_asof_leakage(df, available_at={"cpi": pd.Timedelta(0)}) == []
    assert audit_asof_leakage(df, available_at={"cpi": pd.Timedelta(days=-5)}) == []


def test_asof_unlisted_columns_are_never_checked():
    df = _frame(cpi=np.arange(float(N)), other=np.arange(float(N)))
    issues = audit_asof_leakage(df, available_at={"cpi": pd.Timedelta(days=30)})
    assert {i.column for i in issues} == {"cpi"}


# ── Repair on degenerate input ───────────────────────────────────────────────


def test_apply_fixes_leaves_an_all_nan_column_alone():
    """
    There is nothing to interpolate from, so the column stays NaN rather than
    being filled with a fabricated constant.
    """
    df = _frame(empty=[np.nan] * N, healthy=np.arange(float(N)))
    report = tsa.scan(df, run_stationarity=False)
    clean = report.apply_fixes(df)

    assert clean["empty"].isna().all()
    assert clean.shape == df.shape  # no rows or columns dropped


def test_apply_fixes_never_mutates_the_input():
    rng = np.random.default_rng(3)
    values = rng.normal(100, 5, N)
    values[10] = 5000.0
    df = _frame(x=values)
    before = df.copy(deep=True)

    report = tsa.scan(df, run_stationarity=False)
    report.apply_fixes(df)

    pd.testing.assert_frame_equal(df, before)


def test_health_score_is_100_when_there_are_no_numeric_columns():
    df = pd.DataFrame({"label": ["a"] * N}, index=IDX)
    report = tsa.scan(df, run_stationarity=False)
    assert report.health_score(df) == 100.0
