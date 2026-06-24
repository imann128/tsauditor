import pytest
import pandas as pd
import numpy as np
from tsauditor.anomaly.point import audit_point_anomalies


@pytest.fixture
def base_date_index():
    return pd.date_range("2026-01-01", periods=100, freq="D")


def test_audit_point_anomalies_cases(base_date_index):
    rng = np.random.default_rng(123)

    # 1. Clean df
    df_clean = pd.DataFrame({"val": rng.normal(0, 1, 100)}, index=base_date_index)
    assert len(audit_point_anomalies(df_clean)) == 0

    # 2. Extreme Z-score outlier
    df_z = pd.DataFrame({"val": rng.normal(0, 1, 100)}, index=base_date_index)
    df_z.iloc[0, 0] = 10.0
    issues_z = audit_point_anomalies(df_z)
    assert len(issues_z) == 1
    assert issues_z[0].evidence["zscore_outlier_count"] >= 1

    # 3. IQR outlier (but not Z-score)
    data = np.concatenate([rng.normal(0, 0.1, 97), [0.35, 0.36, 0.37]])
    df_iqr = pd.DataFrame({"val": data}, index=base_date_index)
    issues_iqr = audit_point_anomalies(df_iqr)

    assert len(issues_iqr) == 1
    assert issues_iqr[0].evidence["iqr_outlier_count"] >= 3
    assert issues_iqr[0].evidence["zscore_outlier_count"] == 0


def test_audit_point_anomalies_constant_col(base_date_index):
    df = pd.DataFrame({"val": [1.0] * 100}, index=base_date_index)
    assert len(audit_point_anomalies(df)) == 0


def test_audit_point_anomalies_invalid_index():
    df = pd.DataFrame({"a": [1, 2]}, index=[1, 2])
    with pytest.raises(ValueError, match="DataFrame index must be a pd.DatetimeIndex"):
        audit_point_anomalies(df)


def test_audit_point_anomalies_finance_threshold(base_date_index):
    rng = np.random.default_rng(123)
    data = rng.normal(0, 2, 100)
    data[0] = 4.5
    df = pd.DataFrame({"val": data}, index=base_date_index)

    # In finance (5.0), 4.5 is not an outlier
    issues = audit_point_anomalies(df, domain="finance")
    assert len(issues) == 0


def test_audit_point_anomalies_all_nan_column_skipped(base_date_index):
    """Column that is entirely NaN is skipped gracefully."""
    df = pd.DataFrame(
        {"all_nan": [np.nan] * 100, "valid": np.random.default_rng(42).normal(0, 1, 100)},
        index=base_date_index,
    )
    issues = audit_point_anomalies(df)
    nan_issues = [i for i in issues if i.column == "all_nan"]
    assert len(nan_issues) == 0


def test_single_row_df():
    """Single row DataFrame: returns empty list, no crash."""
    dates = pd.date_range("2026-01-01", periods=1, freq="D")
    df = pd.DataFrame({"val": [1.0]}, index=dates)
    issues = audit_point_anomalies(df)
    assert isinstance(issues, list)
    assert len(issues) == 0
