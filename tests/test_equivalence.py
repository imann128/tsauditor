import numpy as np
import pandas as pd
import pytest

from tsauditor.leakage.equivalence import audit_equivalence
from tsauditor.report.summary import CRITICAL


def _idx(n):
    return pd.date_range("2020-01-01", periods=n, freq="B")


# ── Headline OGDC cases ───────────────────────────────────────────────────────


def test_clean_financial_no_equivalence(clean_financial_df):
    """Clean data: no feature is equivalent to the Direction target."""
    issues = audit_equivalence(clean_financial_df, target="Direction")
    assert issues == []


def test_binary_equivalence_caught_via_auc(leaky_financial_df):
    """ChangeP defines Direction's sign -> AUC=1.0 even though Pearson tops out ~0.8."""
    issues = audit_equivalence(leaky_financial_df, target="Direction")
    flagged = {i.column for i in issues}
    assert "ChangeP" in flagged
    iss = next(i for i in issues if i.column == "ChangeP")
    assert iss.code == "LEK001"
    assert iss.severity == CRITICAL
    assert iss.evidence["metric"] == "auc"
    assert iss.evidence["separation"] >= 0.95


def test_pearson_would_have_missed_it(leaky_financial_df):
    """Document the ceiling: point-biserial Pearson < 0.80, AUC ~ 1.0."""
    df = leaky_financial_df
    pearson = abs(df["ChangeP"].corr(df["Direction"].astype(float)))
    assert pearson < 0.80  # below the spec's old cutoff
    iss = next(
        i for i in audit_equivalence(df, target="Direction") if i.column == "ChangeP"
    )
    assert iss.evidence["separation"] >= 0.99  # AUC catches it cleanly


# ── Continuous target (Spearman) ──────────────────────────────────────────────


def test_continuous_linear_equivalence_caught():
    n = 200
    t = np.random.default_rng(1).normal(0, 1, n)
    df = pd.DataFrame(
        {
            "target": t,
            "leak": 2 * t + 1e-9,
            "noise": np.random.default_rng(2).normal(0, 1, n),
        },
        index=_idx(n),
    )
    issues = audit_equivalence(df, target="target")
    flagged = {i.column for i in issues}
    assert "leak" in flagged and "noise" not in flagged
    assert (
        next(i for i in issues if i.column == "leak").evidence["metric"] == "spearman"
    )


def test_monotonic_nonlinear_equivalence_caught():
    """Spearman catches a monotonic transform that Pearson would underrate."""
    n = 200
    t = np.random.default_rng(3).normal(0, 1, n)
    df = pd.DataFrame({"target": t, "exp_leak": np.exp(t)}, index=_idx(n))
    issues = audit_equivalence(df, target="target")
    assert "exp_leak" in {i.column for i in issues}


def test_legit_weak_feature_not_flagged():
    n = 300
    rng = np.random.default_rng(4)
    t = rng.normal(0, 1, n)
    df = pd.DataFrame(
        {"target": t, "weak": 0.3 * t + rng.normal(0, 1, n)}, index=_idx(n)
    )
    assert audit_equivalence(df, target="target") == []


# ── Edge cases ────────────────────────────────────────────────────────────────


def test_missing_target_raises(clean_financial_df):
    with pytest.raises(ValueError, match="not found"):
        audit_equivalence(clean_financial_df, target="DoesNotExist")


def test_constant_target_returns_empty():
    n = 100
    df = pd.DataFrame(
        {"const": np.ones(n), "x": np.arange(n, dtype=float)}, index=_idx(n)
    )
    assert audit_equivalence(df, target="const") == []


def test_few_observations_skipped():
    """Below min_obs, even a perfect duplicate is not trusted."""
    n = 20
    t = np.arange(n, dtype=float)
    df = pd.DataFrame({"target": t, "leak": t}, index=_idx(n))
    assert audit_equivalence(df, target="target", min_obs=30) == []


def test_constant_feature_skipped():
    n = 100
    t = np.random.default_rng(5).normal(0, 1, n)
    df = pd.DataFrame({"target": t, "flat": np.full(n, 7.0), "leak": t}, index=_idx(n))
    flagged = {i.column for i in audit_equivalence(df, target="target")}
    assert "flat" not in flagged and "leak" in flagged


def test_categorical_binary_target_encoded():
    n = 200
    rng = np.random.default_rng(6)
    x = rng.normal(0, 1, n)
    df = pd.DataFrame({"label": np.where(x > 0, "up", "down"), "x": x}, index=_idx(n))
    issues = audit_equivalence(df, target="label")
    assert "x" in {i.column for i in issues}
    assert (
        next(i for i in issues if i.column == "x").evidence["target_type"] == "binary"
    )


def test_continuous_nonnumeric_target_raises():
    n = 100
    df = pd.DataFrame(
        {"cat": np.array(["a", "b", "c"] * 34)[:n], "x": np.arange(n, dtype=float)},
        index=_idx(n),
    )
    with pytest.raises(ValueError, match="numeric"):
        audit_equivalence(df, target="cat")


def test_target_never_flagged_against_itself():
    n = 100
    t = np.random.default_rng(7).normal(0, 1, n)
    df = pd.DataFrame({"target": t}, index=_idx(n))
    assert all(i.column != "target" for i in audit_equivalence(df, target="target"))


def test_scattered_nans_use_pairwise_overlap():
    n = 200
    t = np.random.default_rng(8).normal(0, 1, n)
    leak = t.copy()
    leak[::5] = np.nan  # 20% missing, still >min_obs overlap
    df = pd.DataFrame({"target": t, "leak": leak}, index=_idx(n))
    issues = audit_equivalence(df, target="target")
    assert "leak" in {i.column for i in issues}
    assert next(i for i in issues if i.column == "leak").evidence["n_obs"] < n


def test_single_row_df():
    """Single row DataFrame with target: returns empty list, no crash."""
    dates = pd.date_range("2026-01-01", periods=1, freq="D")
    df = pd.DataFrame({"target": [1.0], "feat": [2.0]}, index=dates)
    issues = audit_equivalence(df, target="target", min_obs=1)
    assert isinstance(issues, list)
    assert len(issues) == 0
