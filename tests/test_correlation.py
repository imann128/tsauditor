import numpy as np
import pandas as pd
import pytest

from tsauditor.leakage.correlation import audit_correlation_leakage
from tsauditor.report.summary import WARNING


def _idx(n):
    return pd.date_range("2020-01-01", periods=n, freq="B")


def _iid_target(n, seed=0):
    return pd.Series(np.random.default_rng(seed).normal(0, 1, n), index=_idx(n))


# ── Clean / legitimate features ───────────────────────────────────────────────


def test_clean_financial_no_positive_lag_peak(clean_financial_df):
    assert audit_correlation_leakage(clean_financial_df, target="Direction") == []


def test_past_lagged_feature_not_flagged():
    """A feature built from PAST target values peaks at a negative lag."""
    n = 300
    t = _iid_target(n, 1)
    df = pd.DataFrame({"target": t, "past": t.shift(2)}, index=_idx(n))
    assert audit_correlation_leakage(df, target="target") == []


def test_contemporaneous_feature_not_flagged():
    """A lag-0 association is not a positive-lag peak."""
    n = 300
    t = _iid_target(n, 2)
    df = pd.DataFrame(
        {"target": t, "same": t + np.random.default_rng(9).normal(0, 0.1, n)},
        index=_idx(n),
    )
    assert audit_correlation_leakage(df, target="target") == []


# ── Leakage cases ─────────────────────────────────────────────────────────────


def test_future_target_leak_caught():
    n = 300
    t = _iid_target(n, 3)
    df = pd.DataFrame({"target": t, "leak": t.shift(-1)}, index=_idx(n))
    issues = audit_correlation_leakage(df, target="target")
    leak = next(i for i in issues if i.column == "leak")
    assert leak.code == "LEK002"
    assert leak.severity == WARNING
    assert leak.evidence["peak_lag"] == 1
    assert leak.evidence["metric"] == "spearman"


def test_binary_target_peak_lag_preserved():
    """Encoding a binary target attenuates magnitude but keeps the peak lag."""
    n = 300
    b = pd.Series(
        (np.random.default_rng(4).normal(0, 1, n) > 0).astype(int), index=_idx(n)
    )
    df = pd.DataFrame({"label": b, "leak": b.shift(-1)}, index=_idx(n))
    issues = audit_correlation_leakage(df, target="label")
    assert "leak" in {i.column for i in issues}
    assert next(i for i in issues if i.column == "leak").evidence["peak_lag"] == 1


# ── Parameters ────────────────────────────────────────────────────────────────


def test_min_correlation_floor_suppresses():
    """A moderate future leak (~0.63 at +1) is flagged by default but
    suppressed once the correlation floor is raised above it."""
    n = 300
    t = _iid_target(n, 5)
    leak = t.shift(-1) + np.random.default_rng(99).normal(0, 1.2, n)
    df = pd.DataFrame({"target": t, "leak": leak}, index=_idx(n))
    assert "leak" in {i.column for i in audit_correlation_leakage(df, target="target")}
    assert audit_correlation_leakage(df, target="target", min_correlation=0.9) == []


def test_max_lag_window_respected():
    """A leak at +3 is missed when max_lag=2 and caught when max_lag=5."""
    n = 300
    t = _iid_target(n, 6)
    df = pd.DataFrame({"target": t, "leak": t.shift(-3)}, index=_idx(n))
    assert audit_correlation_leakage(df, target="target", max_lag=2) == []
    assert "leak" in {
        i.column for i in audit_correlation_leakage(df, target="target", max_lag=5)
    }


# ── Edge cases ────────────────────────────────────────────────────────────────


def test_missing_target_raises(clean_financial_df):
    with pytest.raises(ValueError, match="not found"):
        audit_correlation_leakage(clean_financial_df, target="Nope")


def test_constant_target_returns_empty():
    n = 100
    df = pd.DataFrame(
        {"const": np.ones(n), "x": np.arange(n, dtype=float)}, index=_idx(n)
    )
    assert audit_correlation_leakage(df, target="const") == []


def test_constant_feature_skipped():
    n = 200
    t = _iid_target(n, 7)
    df = pd.DataFrame(
        {"target": t, "flat": np.full(n, 3.0), "leak": t.shift(-1)}, index=_idx(n)
    )
    flagged = {i.column for i in audit_correlation_leakage(df, target="target")}
    assert "flat" not in flagged and "leak" in flagged


def test_nonnumeric_nonbinary_target_raises():
    n = 99
    df = pd.DataFrame(
        {"cat": np.array(["a", "b", "c"] * 33), "x": np.arange(n, dtype=float)},
        index=_idx(n),
    )
    with pytest.raises(ValueError, match="binary"):
        audit_correlation_leakage(df, target="cat")


def test_few_observations_skipped():
    n = 20
    t = _iid_target(n, 8)
    df = pd.DataFrame({"target": t, "leak": t.shift(-1)}, index=_idx(n))
    assert audit_correlation_leakage(df, target="target", min_obs=30) == []


def test_single_row_df():
    """Single row DataFrame with target: returns empty list, no crash."""
    dates = pd.date_range("2026-01-01", periods=1, freq="D")
    df = pd.DataFrame({"target": [1.0], "feat": [2.0]}, index=dates)
    issues = audit_correlation_leakage(df, target="target", min_obs=1)
    assert isinstance(issues, list)
    assert len(issues) == 0
