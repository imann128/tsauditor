import pytest
import pandas as pd
import numpy as np
from tsauditor.profiler.stationarity import audit_stationarity


@pytest.fixture
def base_date_index():
    return pd.date_range("2026-01-01", periods=100, freq="D")


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


def test_all_nan_column_skipped(base_date_index):
    """Column that is entirely NaN is skipped gracefully."""
    df = pd.DataFrame(
        {"all_nan": [np.nan] * 100, "valid": np.random.randn(100)},
        index=base_date_index,
    )
    issues = audit_stationarity(df, min_obs=25)
    nan_issues = [i for i in issues if i.column == "all_nan"]
    assert len(nan_issues) == 0


def test_single_row_df_raises_without_guard():
    """Single row DataFrame passes min_obs=0 but statsmodels adfuller
    rejects constant input — guard is missing (known issue)."""
    dates = pd.date_range("2026-01-01", periods=1, freq="D")
    df = pd.DataFrame({"val": [1.0]}, index=dates)
    # TODO: remove pytest.raises once a min_obs guard is added upstream
    with pytest.raises(ValueError, match="constant"):
        audit_stationarity(df, min_obs=0)
