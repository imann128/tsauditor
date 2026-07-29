import numpy as np
import pytest
import pandas as pd
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


# ── Mutation-found gaps ────────────────────────────────────────────────────────


def _idx(n):
    return pd.date_range("2024-01-01", periods=n, freq="D")


def test_stuck_run_boundary_is_strictly_longer_than():
    """
    "A run longer than this is flagged" (the docstring's own words) means
    run > stuck_window, not run >= stuck_window. A run of exactly stuck_window
    must not fire; stuck_window + 1 must.

    Mutation-checked: `>` -> `>=` left every existing test in this file passing.
    """
    rng = np.random.default_rng(11)

    def _with_run(run_len):
        v = rng.normal(20, 1, 40)
        v[10 : 10 + run_len] = 7.0
        return pd.DataFrame({"x": v}, index=_idx(40))

    exactly = _with_run(5)
    over = _with_run(6)
    assert audit_contextual_anomalies(exactly, stuck_window=5) == []
    flagged = audit_contextual_anomalies(over, stuck_window=5)
    assert len(flagged) == 1
    assert flagged[0].code == "ANO001"


def test_interpolate_limit_does_not_bridge_wide_gaps_in_a_stuck_run():
    """
    handle_missing="interpolate" is documented as filling *single-row* gaps.
    With a wider gap sitting in the middle of an otherwise-stuck run, filling
    more than one row would bridge the gap with the same repeated value
    (linear interpolation between two equal endpoints is constant), merging two
    separate runs into one long enough to flag.

    Mutation-checked: raising the interpolation limit from 1 to 5 flips this
    fixture from not-flagged to flagged, and no existing test in this file
    caught it.
    """
    rng = np.random.default_rng(1)
    values = np.concatenate(
        [
            np.full(10, 50.0),
            [np.nan] * 3,
            np.full(10, 50.0),
            rng.normal(20, 1, 20),
        ]
    )
    df = pd.DataFrame({"x": values}, index=_idx(len(values)))
    issues = audit_contextual_anomalies(
        df, handle_missing="interpolate", stuck_window=15
    )
    assert [i for i in issues if i.code == "ANO001"] == []


def test_spike_zscore_boundary_is_strictly_greater_than():
    """
    A local z-score exactly at spike_threshold must not fire; the docstring
    describes the threshold as something a spike must exceed.

    An earlier version of this test used a fully flat series and
    spike_threshold=0.0, intending a point with z exactly 0 on the boundary.
    That does not work: a flat context gives local_std == 0, so z is 0/0 = NaN,
    not 0, and NaN is filtered out by `.fillna(False)` regardless of `>` or
    `>=`. The mutant survived that fixture.

    This version reads the detector's own reported z-score for a real spike,
    then re-runs with spike_threshold set to that exact float, so the point
    sits precisely on the boundary rather than falling into the NaN case.

    Mutation-checked: `>` -> `>=` on the z-score comparison flags the
    exact-boundary point that must stay unflagged, and this test catches it;
    the previous fixture did not.
    """
    rng = np.random.default_rng(5)
    values = rng.normal(50, 2, 60)
    values[30] = 58.0
    df = pd.DataFrame({"x": values}, index=_idx(60))

    # The exact local z-score of the spike at index 30, read from the
    # detector itself rather than hand-computed, so it can never drift out of
    # sync with the implementation.
    probe = audit_contextual_anomalies(df, spike_threshold=1e-9)
    exact_z = [i for i in probe if i.code == "ANO003"][0].evidence["max_spike_zscore"]
    # The evidence value is rounded for display; recover the unrounded z-score
    # the comparison actually used by re-deriving it the same way the detector
    # does, rather than trusting the rounded figure for a boundary test.
    sq = df["x"].pow(2)
    mp = max(3, 21 // 2)
    roll = df["x"].rolling(window=21, center=True, min_periods=mp)
    roll_sq = sq.rolling(window=21, center=True, min_periods=mp)
    n_excl = roll.count() - 1
    local_mean = (roll.sum() - df["x"]) / n_excl
    local_var = (roll_sq.sum() - sq) / n_excl - local_mean.pow(2)
    local_std = np.sqrt(local_var.clip(lower=0))
    exact_z = float((df["x"] - local_mean).abs().iloc[30] / local_std.iloc[30])

    at_boundary = audit_contextual_anomalies(df, spike_threshold=exact_z)
    just_below = audit_contextual_anomalies(df, spike_threshold=exact_z - 1e-9)

    assert [i for i in at_boundary if i.code == "ANO003"] == []
    assert len([i for i in just_below if i.code == "ANO003"]) == 1
