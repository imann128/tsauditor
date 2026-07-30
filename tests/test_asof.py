import numpy as np
import pandas as pd
import pytest

from tsauditor.leakage.asof import audit_asof_leakage
from tsauditor.report.summary import CRITICAL


def _idx(n):
    return pd.date_range("2020-01-01", periods=n, freq="D")


# ── Clean / legitimate (must NOT flag) ────────────────────────────────────────


def test_no_availability_no_check():
    df = pd.DataFrame({"cpi": np.arange(50.0)}, index=_idx(50))
    assert audit_asof_leakage(df, available_at={}) == []


def test_zero_lag_is_clean():
    """A value available exactly at its row timestamp is not leakage."""
    df = pd.DataFrame({"cpi": np.arange(50.0)}, index=_idx(50))
    assert audit_asof_leakage(df, {"cpi": pd.Timedelta(0)}) == []


def test_already_shifted_series_is_clean():
    """Availability at or before each row timestamp -> no violation."""
    idx = _idx(50)
    df = pd.DataFrame({"cpi": np.arange(50.0)}, index=idx)
    # published one day BEFORE it is used -> safe
    avail = pd.Series(idx - pd.Timedelta(days=1), index=idx)
    assert audit_asof_leakage(df, {"cpi": avail}) == []


# ── Leakage (must flag) ───────────────────────────────────────────────────────


def test_fixed_lag_flags_whole_column():
    """A positive publication lag on an unshifted column is used early on every
    row -> the classic 'forgot to shift the macro series' bug."""
    df = pd.DataFrame({"cpi": np.arange(50.0)}, index=_idx(50))
    issues = audit_asof_leakage(df, {"cpi": pd.Timedelta(days=14)})
    assert len(issues) == 1
    iss = issues[0]
    assert iss.code == "LEK004"
    assert iss.severity == CRITICAL
    assert iss.column == "cpi"
    assert iss.evidence["n_violations"] == 50
    assert iss.evidence["max_lookahead_days"] == 14.0


def test_ragged_release_flags_only_early_rows():
    """Only the rows whose release timestamp is later than the row are flagged."""
    idx = _idx(10)
    df = pd.DataFrame({"macro": np.arange(10.0)}, index=idx)
    avail = pd.Series(idx, index=idx)
    avail.iloc[3] = idx[3] + pd.Timedelta(days=2)  # this one leaks
    avail.iloc[7] = idx[7] + pd.Timedelta(days=5)  # so does this
    issues = audit_asof_leakage(df, {"macro": avail})
    assert issues[0].evidence["n_violations"] == 2
    assert issues[0].evidence["max_lookahead_days"] == 5.0


def test_min_violations_threshold_suppresses():
    idx = _idx(10)
    df = pd.DataFrame({"macro": np.arange(10.0)}, index=idx)
    avail = pd.Series(idx, index=idx)
    avail.iloc[3] = idx[3] + pd.Timedelta(days=2)  # a single violation
    assert audit_asof_leakage(df, {"macro": avail}, min_violations=2) == []


def test_exactly_min_violations_fires():
    """The gate is `< min_violations`, so exactly min_violations early rows
    must still flag, not be suppressed."""
    idx = _idx(10)
    df = pd.DataFrame({"macro": np.arange(10.0)}, index=idx)
    avail = pd.Series(idx, index=idx)
    avail.iloc[3] = idx[3] + pd.Timedelta(days=2)
    avail.iloc[5] = idx[5] + pd.Timedelta(days=2)  # exactly 2 violations
    issues = audit_asof_leakage(df, {"macro": avail}, min_violations=2)
    assert len(issues) == 1
    assert issues[0].evidence["n_violations"] == 2


def test_nan_values_are_not_violations():
    """A missing value carries no information, so it cannot leak."""
    idx = _idx(10)
    v = np.arange(10.0)
    v[:] = np.nan
    df = pd.DataFrame({"macro": v}, index=idx)
    assert audit_asof_leakage(df, {"macro": pd.Timedelta(days=30)}) == []


# ── Integration through scan() ────────────────────────────────────────────────


def test_scan_runs_asof_when_metadata_supplied():
    import tsauditor as tsa

    idx = _idx(60)
    df = pd.DataFrame(
        {"cpi": np.linspace(1, 5, 60), "price": np.linspace(10, 20, 60)}, index=idx
    )
    report = tsa.scan(
        df, available_at={"cpi": pd.Timedelta(days=30)}, run_stationarity=False
    )
    assert "cpi" in report.leaky_columns()
    assert any(i.code == "LEK004" for i in report.critical)


def test_scan_without_metadata_skips_asof():
    import tsauditor as tsa

    idx = _idx(60)
    df = pd.DataFrame({"cpi": np.linspace(1, 5, 60)}, index=idx)
    report = tsa.scan(df, run_stationarity=False)
    assert all(i.code != "LEK004" for i in report.all_issues)


# ── Edge cases ────────────────────────────────────────────────────────────────


def test_missing_column_raises():
    df = pd.DataFrame({"cpi": np.arange(10.0)}, index=_idx(10))
    with pytest.raises(ValueError, match="not found"):
        audit_asof_leakage(df, {"nope": pd.Timedelta(days=1)})


def test_non_datetime_index_raises():
    df = pd.DataFrame({"cpi": np.arange(10.0)})  # default RangeIndex
    with pytest.raises(ValueError, match="DatetimeIndex"):
        audit_asof_leakage(df, {"cpi": pd.Timedelta(days=1)})


def test_bad_spec_type_raises():
    df = pd.DataFrame({"cpi": np.arange(10.0)}, index=_idx(10))
    with pytest.raises(ValueError, match="Series .* or a pandas Timedelta"):
        audit_asof_leakage(df, {"cpi": 14})  # int, not Timedelta/Series
