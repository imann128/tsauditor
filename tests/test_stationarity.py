import pytest
import pandas as pd
import numpy as np

import tsauditor as tsa
from tsauditor.profiler.stationarity import audit_stationarity


@pytest.fixture
def base_date_index():
    return pd.date_range("2026-01-01", periods=100, freq="D")


def test_max_lag_cap_runs(base_date_index):
    """Capping the ADF lag search still returns a list of Issues, no crash."""
    df = pd.DataFrame(
        {"rw": np.cumsum(np.random.default_rng(0).normal(0, 1, 100))},
        index=base_date_index,
    )
    issues = audit_stationarity(df, max_lag=4)
    assert isinstance(issues, list)


def test_scan_run_stationarity_toggle(base_date_index):
    """run_stationarity=False skips the ADF check (no PRF003)."""
    df = pd.DataFrame(
        {"rw": np.cumsum(np.random.default_rng(1).normal(0, 1, 100))},
        index=base_date_index,
    )
    on = tsa.scan(df, run_stationarity=True)
    off = tsa.scan(df, run_stationarity=False)
    assert any(i.code == "PRF003" for i in on.all_issues)  # random walk -> flagged
    assert all(i.code != "PRF003" for i in off.all_issues)  # skipped entirely


def test_audit_stationarity_scenarios(base_date_index):
    np.random.seed(42)

    # 1 & 2. Non-stationary and Stationary
    df = pd.DataFrame(
        {
            "rw_col": np.cumsum(np.random.randn(100)),
            "wn_col": np.random.randn(100),
        },
        index=base_date_index,
    )

    issues = audit_stationarity(df, alpha=0.05, min_obs=25)

    # Verify PRF003 for rw_col
    rw_issues = [i for i in issues if i.column == "rw_col"]
    assert len(rw_issues) == 1
    assert rw_issues[0].code == "PRF003"

    # Verify wn_col is not flagged
    wn_issues = [i for i in issues if i.column == "wn_col"]
    assert len(wn_issues) == 0


def test_audit_stationarity_short_col():
    # 3. Short column in separate DataFrame
    short_index = pd.date_range("2026-01-01", periods=10, freq="D")
    df_short = pd.DataFrame({"short_col": np.random.randn(10)}, index=short_index)
    short_issues = audit_stationarity(df_short, min_obs=25)
    assert len(short_issues) == 0


def test_exactly_min_obs_is_scored_not_skipped():
    """The gate is `< min_obs`, so exactly min_obs observations must still be
    tested, not skipped. (Distinct from test_audit_stationarity_short_col,
    which is below the boundary.)"""
    idx = pd.date_range("2026-01-01", periods=25, freq="D")
    rw = np.cumsum(np.random.default_rng(11).normal(0, 1, 25))
    df = pd.DataFrame({"rw_col": rw}, index=idx)
    issues = audit_stationarity(df, min_obs=25)
    assert any(i.column == "rw_col" and i.code == "PRF003" for i in issues)


def test_audit_stationarity_non_datetime_index():
    # 4. Non-DatetimeIndex
    df = pd.DataFrame({"a": [1, 2, 3]}, index=[1, 2, 3])
    with pytest.raises(ValueError, match="DataFrame index must be a pd.DatetimeIndex"):
        audit_stationarity(df)


def test_audit_stationarity_finance_mixed_columns(base_date_index):
    # 5. Finance domain test: Mixed types (numeric + categorical)
    df = pd.DataFrame(
        {
            "price": np.cumsum(np.random.randn(100)),  # Non-stationary
            "ticker": ["AAPL"] * 100,  # Non-numeric
            "volatility": np.random.randn(100),  # Stationary
        },
        index=base_date_index,
    )

    issues = audit_stationarity(df, domain="finance")

    # Only price should be flagged
    assert len(issues) == 1
    assert issues[0].column == "price"
    assert issues[0].code == "PRF003"


def test_audit_stationarity_with_nan_and_inf(base_date_index):
    # 6. Handling NaNs and Infs
    data = np.random.randn(100)
    data[0:5] = [np.nan, np.inf, -np.inf, np.nan, 1.0]

    df = pd.DataFrame({"dirty_col": data}, index=base_date_index)

    issues = audit_stationarity(df, min_obs=25)
    assert isinstance(issues, list)


def test_two_valued_column_is_still_tested(monkeypatch):
    """The constant-column guard is `nunique < 2`; a column with exactly two
    distinct values is not constant and must still reach adfuller(), not be
    silently skipped. An alternating +-1 series is stationary either way, so
    the output (no PRF003) can't distinguish tested-and-passed from
    skipped -- patch adfuller itself to observe whether it was called."""
    import tsauditor.profiler.stationarity as stat_mod

    calls = []
    real_adfuller = stat_mod.adfuller

    def spy(series, *args, **kwargs):
        calls.append(series)
        return real_adfuller(series, *args, **kwargs)

    monkeypatch.setattr(stat_mod, "adfuller", spy)

    idx = pd.date_range("2020-01-01", periods=40, freq="D")
    two_valued = np.where(np.arange(40) % 2 == 0, 1.0, -1.0)
    df = pd.DataFrame({"alt": two_valued}, index=idx)
    audit_stationarity(df, min_obs=25)
    assert len(calls) == 1


def test_constant_column_does_not_crash_scan():
    """A constant numeric column must not crash the ADF path. statsmodels'
    adfuller raises 'Invalid input, x is constant'; tsauditor skips it as
    trivially stationary and the scan still returns a report."""
    import numpy as np
    import pandas as pd
    import tsauditor as tsa

    idx = pd.date_range("2020-01-01", periods=80, freq="D")
    df = pd.DataFrame({"price": np.linspace(1, 5, 80), "flag": np.ones(80)}, index=idx)
    report = tsa.scan(df, run_stationarity=True)  # must not raise
    assert not any(i.code == "PRF003" and i.column == "flag" for i in report.all_issues)
