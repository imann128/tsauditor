import numpy as np
import pandas as pd
import pytest

from tsauditor.leakage.temporal import audit_temporal_leakage
from tsauditor.report.summary import WARNING


def _idx(n):
    return pd.date_range("2020-01-01", periods=n, freq="B")


def _ar1(n, phi=0.7, seed=0):
    """Strongly autocorrelated target — the case a naive detector false-flags."""
    e = np.random.default_rng(seed).normal(0, 1, n)
    v = np.zeros(n)
    for t in range(1, n):
        v[t] = phi * v[t - 1] + e[t]
    return pd.Series(v, index=_idx(n))


# ── Clean / legitimate (must NOT flag) ────────────────────────────────────────


def test_clean_financial_no_lookahead(clean_financial_df):
    assert audit_temporal_leakage(clean_financial_df, target="Direction") == []


def test_trailing_rolling_not_flagged():
    """KEY: a causal trailing mean on an autocorrelated target is not leakage,
    even though it correlates with the future through persistence."""
    t = _ar1(600, seed=1)
    df = pd.DataFrame({"target": t, "trailing": t.rolling(5).mean()}, index=_idx(600))
    assert audit_temporal_leakage(df, target="target") == []


def test_past_lagged_not_flagged():
    t = _ar1(600, seed=2)
    df = pd.DataFrame({"target": t, "past": t.shift(2)}, index=_idx(600))
    assert audit_temporal_leakage(df, target="target") == []


def test_noise_not_flagged():
    n = 400
    t = _ar1(n, seed=3)
    df = pd.DataFrame(
        {"target": t, "noise": np.random.default_rng(11).normal(0, 1, n)}, index=_idx(n)
    )
    assert audit_temporal_leakage(df, target="target") == []


# ── Lookahead leakage (must flag) ─────────────────────────────────────────────


def test_centered_rolling_caught():
    """A centered window pulls in future values -> excess over persistence."""
    t = _ar1(600, seed=4)
    df = pd.DataFrame(
        {"target": t, "centered": t.rolling(5, center=True).mean()}, index=_idx(600)
    )
    issues = audit_temporal_leakage(df, target="target")
    iss = next(i for i in issues if i.column == "centered")
    assert iss.code == "LEK003"
    assert iss.severity == WARNING
    assert iss.evidence["excess_over_persistence"] >= 0.1


def test_future_target_leak_caught():
    t = _ar1(600, seed=5)
    df = pd.DataFrame({"target": t, "leak": t.shift(-1)}, index=_idx(600))
    assert "leak" in {i.column for i in audit_temporal_leakage(df, target="target")}


# ── Parameters ────────────────────────────────────────────────────────────────


def test_excess_threshold_param_suppresses():
    t = _ar1(600, seed=6)
    df = pd.DataFrame(
        {"target": t, "centered": t.rolling(5, center=True).mean()}, index=_idx(600)
    )
    assert audit_temporal_leakage(df, target="target", excess_threshold=0.99) == []


# ── Edge cases ────────────────────────────────────────────────────────────────


def test_missing_target_raises(clean_financial_df):
    with pytest.raises(ValueError, match="not found"):
        audit_temporal_leakage(clean_financial_df, target="Nope")


def test_constant_target_returns_empty():
    n = 100
    df = pd.DataFrame(
        {"const": np.ones(n), "x": np.arange(n, dtype=float)}, index=_idx(n)
    )
    assert audit_temporal_leakage(df, target="const") == []


def test_constant_feature_skipped():
    t = _ar1(400, seed=7)
    df = pd.DataFrame(
        {"target": t, "flat": np.full(400, 2.0), "leak": t.shift(-1)}, index=_idx(400)
    )
    flagged = {i.column for i in audit_temporal_leakage(df, target="target")}
    assert "flat" not in flagged and "leak" in flagged


def test_few_observations_skipped():
    n = 20
    t = _ar1(n, seed=8)
    df = pd.DataFrame({"target": t, "leak": t.shift(-1)}, index=_idx(n))
    assert audit_temporal_leakage(df, target="target", min_obs=30) == []


def test_single_row_df():
    """Single row DataFrame with target: returns empty list, no crash."""
    dates = pd.date_range("2026-01-01", periods=1, freq="D")
    df = pd.DataFrame({"target": [1.0], "feat": [2.0]}, index=dates)
    issues = audit_temporal_leakage(df, target="target", min_obs=1)
    assert isinstance(issues, list)
    assert len(issues) == 0
