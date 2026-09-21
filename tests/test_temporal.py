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
    assert iss.evidence["excess_scale"] == "fisher_z"


# ── Fisher-z reformulation (persistence-collapse fix) ──────────────────────────
#
# expected(k) = |r0| * |persistence(k)| and excess(k) = |observed(k)| -
# expected(k) used to be computed as a raw correlation-point difference.
# Correlation is bounded in [-1, 1] and compresses near the endpoints, so
# as the target's own persistence approaches 1 (an undifferenced price
# level -- this library's stated finance use case -- is close to a random
# walk), both an honest feature's and a leaky feature's observed(k) get
# pinned near the same ceiling as expected(k), and the raw subtraction
# loses almost all resolving power between them. Confirmed by direct
# simulation before fixing: 100% detection of an obvious centered-window
# leak through phi=0.9, then 13% at phi=0.95, 0% from phi=0.97 through a
# literal random walk. Fixed by comparing Fisher-z transforms
# (arctanh(observed) - arctanh(expected)) instead of raw correlations --
# arctanh stretches the space back out near +/-1 while being close to the
# identity for small-to-moderate correlations, so ordinary (non-extreme)
# persistence is unaffected: every pre-existing test in this file still
# passes with its original excess_threshold=0.1 default unchanged.


@pytest.mark.parametrize("phi", [0.7, 0.9, 0.95, 0.99, 1.0])
def test_centered_rolling_caught_across_persistence_levels(phi):
    """
    The core regression for the reformulation. Same construction as
    test_centered_rolling_caught, swept across persistence levels that
    span exactly where the raw-difference formula collapsed (verified
    against this exact seed/construction before the fix: raw excess was
    0.032, under the 0.1 threshold, at phi=1.0 -- an obvious leak that went
    completely undetected). Every level here must still be caught.
    """
    t = _ar1(600, phi=phi, seed=9)
    df = pd.DataFrame(
        {"target": t, "centered": t.rolling(5, center=True).mean()}, index=_idx(600)
    )
    issues = audit_temporal_leakage(df, target="target")
    flagged = {i.column for i in issues}
    assert "centered" in flagged, (
        f"an obvious centered-window leak went undetected at phi={phi} -- "
        f"the excess formula is likely back to a raw correlation "
        f"difference instead of a Fisher-z (arctanh) difference"
    )


def test_honest_trailing_feature_not_flagged_near_random_walk():
    """
    False-positive guard for the same reformulation. A ratio-based
    alternative (excess = (observed - expected) / (1 - expected)) was
    tried and rejected before landing on Fisher-z: it also restored
    recall, but its denominator approaches zero as persistence approaches
    1, which amplifies ordinary Spearman sampling noise on an honest
    feature into a false positive -- measured 12-30% false-positive rate
    on this exact kind of trailing feature at phi >= 0.95 across a
    60-trial sweep, versus 0% for the Fisher-z transform at matching
    recall. This pins that the shipped fix does not have that failure
    mode, on a target close enough to a literal random walk (phi=1.0) to
    expose it if it did.
    """
    t = _ar1(600, phi=1.0, seed=109)
    df = pd.DataFrame({"target": t, "trailing": t.rolling(5).mean()}, index=_idx(600))
    assert audit_temporal_leakage(df, target="target") == []


def test_future_target_leak_caught():
    t = _ar1(600, seed=5)
    df = pd.DataFrame({"target": t, "leak": t.shift(-1)}, index=_idx(600))
    assert "leak" in {i.column for i in audit_temporal_leakage(df, target="target")}


# ── Row-order dependence (full-sweep finding) ──────────────────────────────
#
# y.shift(-k) is positional, not timestamp-aware. Before
# audit_temporal_leakage validated and sorted its own input, a caller who
# passed rows out of chronological order (still a valid, non-duplicate
# DatetimeIndex) got an empty result -- no error, no warning -- instead of
# the leak that the same rows, sorted, correctly find.


def test_shuffled_but_valid_index_still_finds_the_leak():
    """Regression: see the identical-in-spirit test in test_correlation.py."""
    t = _ar1(600, seed=5)
    df_sorted = pd.DataFrame({"target": t, "leak": t.shift(-1)}, index=_idx(600))
    df_shuffled = df_sorted.sample(frac=1.0, random_state=3)

    assert "leak" in {
        i.column for i in audit_temporal_leakage(df_sorted, target="target")
    }
    assert "leak" in {
        i.column for i in audit_temporal_leakage(df_shuffled, target="target")
    }


def test_non_datetime_index_raises():
    df = pd.DataFrame({"target": np.arange(50.0), "x": np.arange(50.0)})
    with pytest.raises(ValueError, match="DatetimeIndex"):
        audit_temporal_leakage(df, target="target")


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


# ── Mismatched-sample expected(k): a feature with its own missingness ─────────


def _regime_switching_target(n, seed, phi_hi, phi_lo):
    """
    An AR(1) target whose persistence itself changes partway through: highly
    persistent (phi_hi) for the first half, close to white noise (phi_lo) for
    the second. Unlike `_ar1`, the target's own autocorrelation is not
    uniform across the series -- which is exactly what a feature recorded
    only in one half needs to expose a population mismatch.
    """
    rng = np.random.default_rng(seed)
    v = np.empty(n)
    v[0] = 0.0
    for t in range(1, n):
        phi = phi_hi if t < n // 2 else phi_lo
        v[t] = phi * v[t - 1] + rng.normal(0, 1)
    return rng, pd.Series(v, index=_idx(n))


def test_staggered_feature_leak_caught_despite_regime_change():
    """
    Regression for the mismatched-sample expected(k) bug: persistence,
    r0, and observed used to each be computed on their own independent
    pairwise-complete sample, which let persistence silently reflect a
    different population than the feature actually occupies.

    Here the target is highly persistent (phi=0.97) for the first half and
    near white noise (phi=0.0) for the second. `leaky` is a genuine lag -1
    leak (it peeks at tomorrow's target, moderately, with heavy noise) but
    is only recorded in the low-persistence second half -- e.g. a data
    source added partway through the series.

    Computing persistence on the full series (as before the fix) measures
    the *first* half's high persistence, which is not the population this
    feature ever occupies. That inflated persistence baseline was large
    enough to absorb this leak's observed future correlation entirely, so
    it went unflagged (best excess ~0.06, under the 0.1 threshold) even
    though the feature is a real, engineered leak. Aligning all three
    correlations to the rows where the feature actually exists (persistence
    ~0.0 there, not ~0.97) correctly exposes it (best excess ~0.14).
    """
    n = 500
    rng, target = _regime_switching_target(n, seed=4, phi_hi=0.97, phi_lo=0.0)

    leaky = pd.Series(np.nan, index=_idx(n))
    peek = target.shift(-1) * 0.15 + rng.normal(0, 0.8, n)
    leaky.iloc[n // 2 :] = peek.iloc[n // 2 :]

    df = pd.DataFrame({"target": target, "leaky": leaky}, index=_idx(n))
    issues = audit_temporal_leakage(df, target="target")
    flagged = {i.column for i in issues}
    assert "leaky" in flagged, (
        "a genuine lag -1 leak recorded only in the low-persistence half of "
        "the series went undetected; expected(k) is likely using an "
        "unaligned (whole-series) persistence estimate again"
    )
    ev = next(i for i in issues if i.column == "leaky").evidence
    assert ev["excess_over_persistence"] >= 0.1


def test_single_row_df():
    # With only one observation every feature column falls below min_obs (30),
    # so audit_temporal_leakage skips all of them and must return an empty
    # list without raising.
    dates = pd.date_range("2026-05-22", periods=1, freq="B")
    df_single = pd.DataFrame({"target": [1.0], "feature": [2.0]}, index=dates)

    issues = audit_temporal_leakage(df_single, target="target")
    assert isinstance(issues, list)
    assert len(issues) == 0
