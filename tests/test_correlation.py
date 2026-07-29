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


# ── Spurious correlation between independent persistent series (#49) ──────────
#
# LEK002 fires when the argmax over lags lands at a positive lag. For two
# persistent series, spurious correlation is large by construction and which lag
# wins is close to a coin flip, so a low `min_correlation` reports leakage
# between columns that are statistically independent. These tests pin the
# false-positive rate so a future change to the gate cannot silently undo the
# 0.3.1 fix.


def _independent_random_walks(seed, n=400):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {"y": np.cumsum(rng.normal(0, 1, n)), "f": np.cumsum(rng.normal(0, 1, n))},
        index=_idx(n),
    )


def _independent_ar1(seed, n=400, rho=0.98):
    rng = np.random.default_rng(seed)
    a = np.zeros(n)
    b = np.zeros(n)
    for i in range(1, n):
        a[i] = rho * a[i - 1] + rng.normal(0, 0.3)
        b[i] = rho * b[i - 1] + rng.normal(0, 0.3)
    return pd.DataFrame({"y": a, "f": b}, index=_idx(n))


@pytest.mark.parametrize(
    "builder, max_false_positives",
    [
        (_independent_random_walks, 25),
        (_independent_ar1, 20),
    ],
    ids=["random_walk", "ar1_rho_0.98"],
)
def test_independent_persistent_series_rarely_flagged(builder, max_false_positives):
    """
    Both columns are generated from separate draws, so every flag is a false
    positive. At the pre-0.3.1 default of 0.1 these measured 37/100 and 51/100.

    The bounds are deliberately looser than the measured 13 and 8 so that
    ordinary sampling variation does not make the suite flaky; they are tight
    enough that a regression to the old gate fails.
    """
    flagged = sum(
        bool(audit_correlation_leakage(builder(s), target="y")) for s in range(100)
    )
    assert flagged <= max_false_positives


@pytest.mark.parametrize("persistent", [False, True], ids=["iid", "random_walk"])
def test_genuine_lookahead_still_detected(persistent):
    """
    The gate must not cost true positives, on an i.i.d. target or a persistent
    one. Both measured 100/100 before and after the change.

    The persistent case is the one that rules out a margin-over-lag-0 rule: a
    lookahead on a random walk correlates with the target at lag 0 almost as
    strongly as at lag 1, so a flat margin drops this to 0%.
    """
    n = 400
    detected = 0
    for s in range(20):
        rng = np.random.default_rng(5000 + s)
        y = np.cumsum(rng.normal(0, 1, n)) if persistent else rng.normal(0, 1, n)
        df = pd.DataFrame(
            {"y": y, "f": pd.Series(y).shift(-1).bfill().to_numpy()}, index=_idx(n)
        )
        detected += bool(audit_correlation_leakage(df, target="y"))
    assert detected == 20


def test_default_min_correlation_is_the_fixed_value():
    """Pins the default so #49 cannot be reverted without failing a test."""
    import inspect

    default = (
        inspect.signature(audit_correlation_leakage)
        .parameters["min_correlation"]
        .default
    )
    assert default == 0.5


def test_peak_correlation_keeps_its_sign():
    """
    `peak_correlation` is documented as signed, and the description prints it as
    a Spearman value. A leak built from the *negated* future target is just as
    much a leak, and reporting it as +1.0 would tell the user the feature tracks
    the target when it inverts it.

    Mutation-checked: reporting abs(r) instead of r leaves every other test in
    this file passing.
    """
    n = 300
    y = np.random.default_rng(0).normal(0, 1, n)
    df = pd.DataFrame(
        {"y": y, "f": -pd.Series(y).shift(-1).bfill().to_numpy()}, index=_idx(n)
    )
    issues = audit_correlation_leakage(df, target="y")
    assert len(issues) == 1
    assert issues[0].evidence["peak_correlation"] == -1.0
