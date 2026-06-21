import pandas as pd
import numpy as np
from tsauditor.profiler.missing import audit_missing
from tsauditor.report.summary import WARNING


def test_clean_df_no_missing_values(clean_financial_df):
    # Case 1 — Clean df with no missing values -> no issues.
    issues = audit_missing(clean_financial_df, domain="finance")
    assert len(issues) == 0


def test_consecutive_nans_finance_trigger(clean_financial_df):
    # Case 2 — Column with 5 consecutive NaNs, finance domain -> PRF002.
    df = clean_financial_df.copy()
    # Inject exactly 5 consecutive NaNs into a target column
    df.iloc[10:15, df.columns.get_loc("Price")] = np.nan

    issues = audit_missing(df, domain="finance")

    cluster_issues = [i for i in issues if i.code == "PRF002" and i.column == "Price"]
    assert len(cluster_issues) == 1
    assert cluster_issues[0].severity == WARNING
    assert cluster_issues[0].evidence["longest_consecutive_run"] == 5


def test_high_missing_rate_only(clean_financial_df):
    # Case 3 — Column with 80% missing -> PRF006.
    df = clean_financial_df.copy()
    # Mask 80% of rows completely spread out (evenly spaced to prevent clustering)
    # Step size of 5 leaves 20% intact, removing 80%
    mask = np.arange(len(df)) % 5 != 0
    df.iloc[mask, df.columns.get_loc("Price")] = np.nan

    issues = audit_missing(df, domain="finance", missing_rate_threshold=0.30)

    # Expect PRF006 but NO PRF002 because cluster_threshold=5 and runs are length 4
    rate_issues = [i for i in issues if i.code == "PRF006" and i.column == "Price"]
    cluster_issues = [i for i in issues if i.code == "PRF002" and i.column == "Price"]

    assert len(rate_issues) == 1
    assert len(cluster_issues) == 0
    assert rate_issues[0].evidence["missing_percentage"] >= 79.0


def test_high_missing_rate_and_clustering(clean_financial_df):
    # Case 4 — Column with both high missing rate AND clustering -> both PRF002 and PRF006.
    df = clean_financial_df.copy()
    # Block-out the first 80% of the dataframe sequentially to trigger both rules
    cutoff = int(len(df) * 0.80)
    df.iloc[:cutoff, df.columns.get_loc("Price")] = np.nan

    issues = audit_missing(df, domain="finance")

    col_issues = [i for i in issues if i.column == "Price"]
    codes = {i.code for i in col_issues}

    assert "PRF002" in codes
    assert "PRF006" in codes
    assert len(col_issues) == 2


def test_scattered_nans_below_threshold(clean_financial_df):
    # Case 5 — Column with scattered single NaNs below cluster threshold -> no PRF002.
    df = clean_financial_df.copy()
    # Inject isolated NaNs separated by valid rows to prevent run accumulations
    df.iloc[10, df.columns.get_loc("Price")] = np.nan
    df.iloc[20, df.columns.get_loc("Price")] = np.nan
    df.iloc[30, df.columns.get_loc("Price")] = np.nan

    issues = audit_missing(df, domain="finance")

    cluster_issues = [i for i in issues if i.code == "PRF002" and i.column == "Price"]
    assert len(cluster_issues) == 0


def test_non_numeric_columns_ignored():
    # Case 6 — Non-numeric columns -> ignored, no issues.
    dates = pd.date_range("2026-05-22", periods=10, freq="B")
    df_str = pd.DataFrame(
        {"text_col": ["A", None, None, None, None, None, "B", "C", "D", "E"]},
        index=dates,
    )

    issues = audit_missing(df_str, domain="finance")
    assert len(issues) == 0


def test_sensor_domain_lower_threshold(sensor_df):
    # Case 7 — Sensor domain with 3 consecutive NaNs -> PRF002 (lower threshold).
    df = sensor_df.copy()
    # Inject exactly 3 consecutive NaNs into a sensor column
    df.iloc[50:53, df.columns.get_loc("temperature")] = np.nan

    issues = audit_missing(df, domain="sensor")

    cluster_issues = [
        i for i in issues if i.code == "PRF002" and i.column == "temperature"
    ]
    assert len(cluster_issues) == 1
    assert cluster_issues[0].evidence["cluster_threshold"] == 3
    assert cluster_issues[0].evidence["longest_consecutive_run"] == 3
