import pytest
import pandas as pd
import numpy as np
from tsauditor.anomaly.contextual import audit_contextual_anomalies
from tsauditor.report.summary import WARNING


def test_clean_df_no_anomalies(clean_financial_df):
    """Case 1 — Clean df -> no issues on continuous market data columns."""
    market_cols = ["Price", "Open", "High", "Low", "Volume"]
    df_clean = clean_financial_df[market_cols]

    issues = audit_contextual_anomalies(df_clean, domain="finance")
    assert len(issues) == 0


def test_stuck_values_finance_trigger(clean_financial_df):
    """Case 2 — Column with 6 identical consecutive values, finance domain -> ANO001."""
    df = clean_financial_df.copy()
    df.iloc[20:26, df.columns.get_loc("Price")] = 150.0

    issues = audit_contextual_anomalies(df, domain="finance")

    stuck_issues = [i for i in issues if i.code == "ANO001" and i.column == "Price"]
    assert len(stuck_issues) == 1
    assert stuck_issues[0].severity == WARNING
    assert stuck_issues[0].evidence["max_stuck_duration"] == 6


def test_sharp_spike_and_recovery(clean_financial_df):
    """Case 3 — Column with a sharp spike and recovery -> ANO003."""
    df = clean_financial_df.copy()

    # Establish a highly stable local historical context with small variance
    # This prevents the local rolling std dev from being zero (which returns NaN)
    # while ensuring a sudden jump will flag as an anomaly
    base_idx = 50
    df.iloc[base_idx - 10 : base_idx + 10, df.columns.get_loc("Price")] = [
        100.0 + (i % 2) * 0.1 for i in range(20)
    ]

    # Inject the transient spike at the center point
    df.iloc[base_idx, df.columns.get_loc("Price")] = 150.0

    issues = audit_contextual_anomalies(df, domain="finance")

    spike_issues = [i for i in issues if i.code == "ANO003" and i.column == "Price"]
    assert len(spike_issues) >= 1
    assert spike_issues[0].severity == WARNING


def test_stuck_values_sensor_lower_threshold(sensor_df):
    """Case 4 — Sensor domain with 3 stuck values -> ANO001 (lower threshold)."""
    df = sensor_df.copy()
    df.iloc[100:104, df.columns.get_loc("temperature")] = 22.5

    issues = audit_contextual_anomalies(df, domain="sensor")

    stuck_issues = [
        i for i in issues if i.code == "ANO001" and i.column == "temperature"
    ]
    assert len(stuck_issues) == 1
    assert stuck_issues[0].evidence["max_stuck_duration"] == 4


def test_non_datetime_index_raises_value_error():
    """Case 5 — Non-DatetimeIndex -> raises ValueError."""
    df_bad_index = pd.DataFrame({"Price": [10.0, 11.0, 12.0]}, index=[0, 1, 2])

    with pytest.raises(ValueError, match="DataFrame index must be a pd.DatetimeIndex"):
        audit_contextual_anomalies(df_bad_index, domain="finance")


def test_local_spike_fails_global_zscore(clean_financial_df):
    """
    Case 6 — Spike that wouldn't trigger global z-score -> ANO003 only.
    Tests a spike within a low-volatility regime that is extreme locally,
    but falls within normal bounds when evaluated against global metrics.
    """
    df = clean_financial_df.copy()

    # Establish a stable, low-variance local baseline early in the sequence
    df.iloc[0:40, df.columns.get_loc("Price")] = [
        100.0 + (i % 2) * 0.05 for i in range(40)
    ]
    # Introduce a local spike that breaches this tight baseline context
    df.iloc[20, df.columns.get_loc("Price")] = 120.0

    # Introduce massive values later in the series to blow out the global standard deviation
    df.iloc[200:, df.columns.get_loc("Price")] = 5000.0

    issues = audit_contextual_anomalies(df, domain="finance")

    spike_issues = [i for i in issues if i.code == "ANO003" and i.column == "Price"]
    assert len(spike_issues) >= 1


def test_all_nan_column_skipped():
    """Column that is entirely NaN is skipped gracefully."""
    dates = pd.date_range("2026-01-01", periods=20, freq="D")
    df = pd.DataFrame(
        {"all_nan": [np.nan] * 20, "valid": range(20)},
        index=dates,
    )
    issues = audit_contextual_anomalies(df, domain="finance")
    nan_issues = [i for i in issues if i.column == "all_nan"]
    assert len(nan_issues) == 0


def test_single_row_df():
    """Single row DataFrame: returns empty list, no crash."""
    dates = pd.date_range("2026-01-01", periods=1, freq="D")
    df = pd.DataFrame({"val": [1.0]}, index=dates)
    issues = audit_contextual_anomalies(df, domain="finance")
    assert isinstance(issues, list)
    assert len(issues) == 0
