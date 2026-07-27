"""
LEK005 — combination leakage: a *group* of two or three features reconstructs the
target while none does alone.

The canonical shape is a target defined as a difference. With x1 and x2
independent and target = x1 - x2, each feature correlates with the target at only
~0.7 — well under LEK001's 0.95 — so the leak is invisible to every univariate
check, while the pair explains the target exactly.

Also covered here:
- products and ratios, via the log form
- three-column identities, via residual extension from promising pairs
- the guards that stop this duplicating LEK001's findings
"""

import pathlib

import numpy as np
import pandas as pd
import pytest

import tsauditor as tsa
from tsauditor.leakage.combination import audit_combination_leakage

N = 400
IDX = pd.date_range("2024-01-01", periods=N, freq="D")

# Resolved from this file, not the working directory, so the suite passes
# regardless of where pytest is invoked from.
OGDC_CSV = (
    pathlib.Path(__file__).resolve().parent.parent
    / "examples"
    / "ogdc_leakage_case"
    / "ogdc_with_regimes.csv"
)


def _load_ogdc() -> pd.DataFrame:
    if not OGDC_CSV.exists():  # pragma: no cover - only if examples/ is stripped
        pytest.skip(f"OGDC example data not present at {OGDC_CSV}")
    return pd.read_csv(OGDC_CSV, index_col="Date", parse_dates=True).dropna(
        subset=["Direction"]
    )


@pytest.fixture
def difference_leak() -> pd.DataFrame:
    """target = x1 - x2. Neither input is near-deterministic on its own."""
    rng = np.random.default_rng(0)
    x1 = rng.normal(0, 1, N)
    x2 = rng.normal(0, 1, N)
    return pd.DataFrame(
        {
            "target": x1 - x2,
            "x1": x1,
            "x2": x2,
            "noise": rng.normal(0, 1, N),
        },
        index=IDX,
    )


def _groups(issues):
    return {tuple(sorted(i.evidence["group"])) for i in issues}


# ── Detection ────────────────────────────────────────────────────────────────


def test_detects_difference_leakage(difference_leak):
    issues = audit_combination_leakage(difference_leak, target="target")

    assert len(issues) == 1
    issue = issues[0]
    assert issue.code == "LEK005"
    assert issue.severity == "critical"
    assert tuple(sorted(issue.evidence["group"])) == ("x1", "x2")
    assert issue.evidence["group_adjusted_r2"] >= 0.99
    assert issue.evidence["metric"] == "adjusted_r2"


def test_the_leak_is_invisible_to_lek001(difference_leak):
    """
    The whole justification for LEK005: LEK001 does not and cannot catch this.
    Each feature alone correlates ~0.7 with the target.
    """
    from tsauditor.leakage.equivalence import audit_equivalence

    assert audit_equivalence(difference_leak, target="target") == []

    for col in ("x1", "x2"):
        rho = abs(difference_leak[col].corr(difference_leak["target"]))
        assert 0.5 < rho < 0.95


def test_reports_best_single_r2_as_evidence(difference_leak):
    """The user must be able to see that neither column explains it alone."""
    issue = audit_combination_leakage(difference_leak, target="target")[0]
    assert issue.evidence["best_single_adjusted_r2"] < 0.95
    assert (
        issue.evidence["group_adjusted_r2"] > issue.evidence["best_single_adjusted_r2"]
    )


def test_detects_a_mean_relationship():
    """target = (a + b) / 2 — the other common shape."""
    rng = np.random.default_rng(1)
    a = rng.normal(100, 10, N)
    b = rng.normal(100, 10, N)
    df = pd.DataFrame({"target": (a + b) / 2, "a": a, "b": b}, index=IDX)

    assert _groups(audit_combination_leakage(df, target="target")) == {("a", "b")}


# ── The single-feature guard ─────────────────────────────────────────────────


def test_single_column_leak_is_left_to_lek001():
    """
    Without the guard, one leaky column makes every pair containing it fire,
    producing k-1 duplicate findings for something LEK001 already reports once.
    """
    rng = np.random.default_rng(2)
    leak = rng.normal(0, 1, N)
    df = pd.DataFrame(
        {
            "target": leak,
            "leak": leak,
            **{f"f{i}": rng.normal(0, 1, N) for i in range(5)},
        },
        index=IDX,
    )

    assert audit_combination_leakage(df, target="target") == []


def test_guard_does_not_suppress_genuine_combination_leakage(difference_leak):
    """The guard must not throw the baby out: x1-x2 still fires."""
    assert _groups(audit_combination_leakage(difference_leak, target="target")) == {
        ("x1", "x2")
    }


# ── False positives ──────────────────────────────────────────────────────────


def test_no_false_positives_on_unrelated_features():
    rng = np.random.default_rng(3)
    df = pd.DataFrame(
        {
            "target": rng.normal(0, 1, N),
            **{f"f{i}": rng.normal(0, 1, N) for i in range(12)},
        },
        index=IDX,
    )
    assert audit_combination_leakage(df, target="target") == []


def test_no_false_positives_on_highly_correlated_innocent_features():
    """
    Collinear features (r ~ 0.95) are the obvious false-positive risk, and also
    the case that makes a naive normal-equation solve blow up.
    """
    rng = np.random.default_rng(4)
    f1 = rng.normal(0, 1, N)
    df = pd.DataFrame(
        {
            "target": rng.normal(0, 1, N),
            "f1": f1,
            "f2": f1 * 0.95 + rng.normal(0, 0.3, N),
            "f3": f1 * 0.90 + rng.normal(0, 0.4, N),
        },
        index=IDX,
    )
    assert audit_combination_leakage(df, target="target") == []


def test_real_ogdc_data_produces_no_false_positives_for_direction():
    """A real 24-column financial frame with many derived features."""
    assert audit_combination_leakage(_load_ogdc(), target="Direction") == []


def test_finds_the_real_macd_identity():
    """
    MACD_hist is exactly MACD - MACD_signal in the OGDC file. LEK005 should find
    it, and LEK001 should not (best single adjusted R² is only ~0.12).
    """
    df = _load_ogdc()

    identity = (df["MACD"] - df["MACD_signal"] - df["MACD_hist"]).abs().max()
    assert identity < 1e-9  # the identity really holds

    issues = audit_combination_leakage(df, target="MACD_hist")
    assert _groups(issues) == {("MACD", "MACD_signal")}
    assert issues[0].evidence["group_adjusted_r2"] >= 0.999
    assert issues[0].evidence["best_single_adjusted_r2"] < 0.5


# ── Degenerate input ─────────────────────────────────────────────────────────


def test_missing_target_raises(difference_leak):
    with pytest.raises(ValueError, match="not found"):
        audit_combination_leakage(difference_leak, target="nope")


def test_constant_target_returns_nothing():
    rng = np.random.default_rng(5)
    df = pd.DataFrame(
        {"target": [1.0] * N, "a": rng.normal(0, 1, N), "b": rng.normal(0, 1, N)},
        index=IDX,
    )
    assert audit_combination_leakage(df, target="target") == []


def test_fewer_than_two_features_returns_nothing():
    rng = np.random.default_rng(6)
    df = pd.DataFrame(
        {"target": rng.normal(0, 1, N), "a": rng.normal(0, 1, N)}, index=IDX
    )
    assert audit_combination_leakage(df, target="target") == []


def test_degenerate_columns_do_not_crash(difference_leak):
    df = difference_leak.copy()
    df["all_nan"] = np.nan
    df["constant"] = 5.0
    df["infinite"] = np.inf

    issues = audit_combination_leakage(df, target="target")
    assert _groups(issues) == {("x1", "x2")}


def test_too_few_rows_returns_nothing():
    rng = np.random.default_rng(7)
    idx = pd.date_range("2024-01-01", periods=10, freq="D")
    x1 = rng.normal(0, 1, 10)
    x2 = rng.normal(0, 1, 10)
    df = pd.DataFrame({"target": x1 - x2, "x1": x1, "x2": x2}, index=idx)

    assert audit_combination_leakage(df, target="target", min_obs=30) == []


def test_max_reported_caps_output():
    """A family of derived columns must not produce dozens of findings."""
    rng = np.random.default_rng(8)
    base = {f"x{i}": rng.normal(0, 1, N) for i in range(8)}
    target = base["x0"] - base["x1"]
    df = pd.DataFrame({"target": target, **base}, index=IDX)

    issues = audit_combination_leakage(df, target="target", max_reported=2)
    assert len(issues) <= 2


# ── Wiring ───────────────────────────────────────────────────────────────────


def test_runs_through_scan(difference_leak):
    report = tsa.scan(difference_leak, target="target", run_stationarity=False)
    found = report.filter(code="LEK005")

    assert len(found) == 1
    assert found[0].column in ("x1", "x2")
    assert "x1" in found[0].evidence["group"] and "x2" in found[0].evidence["group"]


def test_appears_in_leaky_columns(difference_leak):
    report = tsa.scan(difference_leak, target="target", run_stationarity=False)
    assert "x1" in report.leaky_columns()


def test_has_a_specific_suggestion(difference_leak):
    report = tsa.scan(difference_leak, target="target", run_stationarity=False)
    suggestion = report.filter(code="LEK005")[0].suggestion

    assert "Review this issue" not in suggestion  # not the generic fallback
    assert "target" in suggestion.lower()


def test_skipped_when_no_target(difference_leak):
    report = tsa.scan(difference_leak, run_stationarity=False)
    assert report.filter(code="LEK005") == []


# ── Multiplicative and ratio relationships (log form) ────────────────────────


@pytest.fixture
def positive_pair():
    """Strictly positive inputs, so the log form is defined."""
    rng = np.random.default_rng(0)
    return rng.uniform(2, 10, N), rng.uniform(2, 10, N)


def _frame(target, a, b, extra_noise=5):
    rng = np.random.default_rng(99)
    noise = {f"n{i}": rng.normal(0, 1, N) for i in range(extra_noise)}
    return pd.DataFrame({"target": target, "a": a, "b": b, **noise}, index=IDX)


def test_detects_a_product(positive_pair):
    """target = a * b. The linear form only reaches ~0.93; the log form is exact."""
    a, b = positive_pair
    issues = audit_combination_leakage(_frame(a * b, a, b), target="target")

    assert _groups(issues) == {("a", "b")}
    assert issues[0].evidence["form"] == "log"
    assert issues[0].evidence["group_adjusted_r2"] >= 0.99


def test_detects_a_ratio(positive_pair):
    """target = a / b. Neither the linear form (~0.83) nor an interaction term
    catches this; log-space does, because log(a/b) = log a - log b."""
    a, b = positive_pair
    issues = audit_combination_leakage(_frame(a / b, a, b), target="target")

    assert _groups(issues) == {("a", "b")}
    assert issues[0].evidence["form"] == "log"


def test_additive_relationship_still_uses_the_linear_form(positive_pair):
    """The log form must not displace the linear one where linear is correct."""
    a, b = positive_pair
    issues = audit_combination_leakage(_frame(a - b, a, b), target="target")

    assert _groups(issues) == {("a", "b")}
    assert issues[0].evidence["form"] == "linear"


def test_log_form_is_skipped_when_values_are_not_positive():
    """A log is only meaningful on strictly positive data; negatives fall back
    to the linear form rather than erroring or producing NaN."""
    rng = np.random.default_rng(1)
    a = rng.normal(0, 1, N)  # spans zero
    b = rng.normal(0, 1, N)

    issues = audit_combination_leakage(_frame(a - b, a, b), target="target")
    assert issues[0].evidence["form"] == "linear"


def test_no_false_positives_in_log_space():
    """Positive random data, no relationship."""
    rng = np.random.default_rng(2)
    df = pd.DataFrame(
        {
            "target": rng.uniform(1, 100, N),
            **{f"f{i}": rng.uniform(1, 100, N) for i in range(12)},
        },
        index=IDX,
    )
    assert audit_combination_leakage(df, target="target") == []


# ── Triples via residual extension ───────────────────────────────────────────


@pytest.fixture
def triple_leak():
    """target = a + b + c. No *pair* reaches the threshold (best is ~0.71)."""
    rng = np.random.default_rng(2)
    a = rng.normal(0, 1, N)
    b = rng.normal(0, 1, N)
    c = rng.normal(0, 1, N)
    noise = {f"n{i}": rng.normal(0, 1, N) for i in range(6)}
    return pd.DataFrame(
        {"target": a + b + c, "a": a, "b": b, "c": c, **noise}, index=IDX
    )


def test_detects_a_three_column_identity(triple_leak):
    issues = audit_combination_leakage(triple_leak, target="target")

    assert _groups(issues) == {("a", "b", "c")}
    assert issues[0].evidence["group_size"] == 3
    assert issues[0].evidence["group_adjusted_r2"] >= 0.99


def test_no_pair_alone_reaches_the_threshold(triple_leak):
    """Confirms the triple is genuinely out of reach of pairwise detection."""
    pairs_only = audit_combination_leakage(
        triple_leak, target="target", max_group_size=2
    )
    assert pairs_only == []


def test_search_depth_is_configurable(triple_leak):
    """max_group_size=2 restricts the search to pairs."""
    assert (
        audit_combination_leakage(triple_leak, target="target", max_group_size=2) == []
    )
    assert _groups(audit_combination_leakage(triple_leak, target="target")) == {
        ("a", "b", "c")
    }


# -- Signed products and ratios, and larger groups ---------------------------


@pytest.fixture
def signed_pair():
    """Inputs that span zero, so a plain positive-only log is undefined."""
    rng = np.random.default_rng(11)
    return rng.normal(0, 3, N), rng.normal(0, 3, N)


def test_detects_a_signed_product(signed_pair):
    """
    The log form uses |values|, since |a*b| = |a|*|b| holds regardless of sign.
    On signed inputs the linear form scores ~0.01 for a product - completely
    blind - so without this the leak is missed entirely.
    """
    a, b = signed_pair
    issues = audit_combination_leakage(_frame(a * b, a, b), target="target")

    assert _groups(issues) == {("a", "b")}
    assert issues[0].evidence["form"] == "log"
    assert issues[0].evidence["group_adjusted_r2"] >= 0.99


def test_detects_a_signed_ratio(signed_pair):
    a, b = signed_pair
    issues = audit_combination_leakage(_frame(a / b, a, b), target="target")

    assert _groups(issues) == {("a", "b")}
    assert issues[0].evidence["form"] == "log"


def test_columns_touching_zero_skip_the_log_form():
    """log of a near-zero would dominate the fit, so those columns use linear."""
    rng = np.random.default_rng(12)
    a = rng.normal(0, 3, N)
    a[5] = 0.0
    b = rng.normal(0, 3, N)

    issues = audit_combination_leakage(_frame(a - b, a, b), target="target")
    assert issues[0].evidence["form"] == "linear"


def test_four_column_identity_needs_the_depth_raised():
    """
    Reachable, but off by default: each level is free on clean data yet costs
    time on frames where many features partially explain the target.
    """
    rng = np.random.default_rng(13)
    a, b, c, d = (rng.normal(0, 1, N) for _ in range(4))
    noise = {f"n{i}": rng.normal(0, 1, N) for i in range(4)}
    df = pd.DataFrame(
        {"target": a + b + c + d, "a": a, "b": b, "c": c, "d": d, **noise}, index=IDX
    )

    assert audit_combination_leakage(df, target="target") == []
    assert _groups(
        audit_combination_leakage(df, target="target", max_group_size=4)
    ) == {("a", "b", "c", "d")}


def test_deeper_search_adds_no_false_positives():
    rng = np.random.default_rng(14)
    df = pd.DataFrame(
        {
            "target": rng.normal(0, 1, N),
            **{f"f{i}": rng.normal(0, 1, N) for i in range(15)},
        },
        index=IDX,
    )
    assert audit_combination_leakage(df, target="target", max_group_size=4) == []


def test_candidate_cap_bounds_the_cost():
    """
    Without a cap, a frame of mutually correlated features explodes: 40 such
    features took 21s at depth 4. The cap keeps it near a second.
    """
    import time

    rng = np.random.default_rng(15)
    base = rng.normal(0, 1, N)
    df = pd.DataFrame(
        {
            "target": base + rng.normal(0, 0.3, N),
            **{f"f{i}": 0.7 * base + rng.normal(0, 0.7, N) for i in range(30)},
        },
        index=IDX,
    )

    start = time.time()
    audit_combination_leakage(df, target="target", max_group_size=4)
    assert time.time() - start < 10.0


def test_triple_gate_blocks_nothing_genuine():
    """
    The gate must not reject real three-way identities. Checked across equal,
    moderately unequal, cancelling and collinear component shapes.
    """
    rng = np.random.default_rng(4)
    a, b, c = (rng.normal(0, 1, N) for _ in range(3))
    p = rng.normal(0, 1, N)
    q = p + rng.normal(0, 0.1, N)
    s = p + rng.normal(0, 0.1, N)

    shapes = {
        "equal": (a + b + c, a, b, c),
        "moderately unequal": (3 * a + b + c, a, b, c),
        "cancelling": (a + b - c, a, b, c),
        "collinear": (p + q - 2 * s, p, q, s),
    }
    for name, (target, x, y, z) in shapes.items():
        noise = {f"n{i}": rng.normal(0, 1, N) for i in range(4)}
        df = pd.DataFrame(
            {"target": target, "x": x, "y": y, "z": z, **noise}, index=IDX
        )
        assert _groups(audit_combination_leakage(df, target="target")) == {
            ("x", "y", "z")
        }, name


def test_a_dominated_identity_is_left_to_lek001():
    """
    `target = 100a + b + c` looks like a three-way identity but is not one in any
    useful sense: `a` alone explains 99.98% of it, so LEK001 catches it and the
    single-feature guard correctly makes LEK005 defer. Reporting it here too
    would be a duplicate finding.
    """
    from tsauditor.leakage.equivalence import audit_equivalence

    rng = np.random.default_rng(4)
    a, b, c = (rng.normal(0, 1, N) for _ in range(3))
    df = pd.DataFrame({"target": 100 * a + b + c, "x": a, "y": b, "z": c}, index=IDX)

    assert [i.column for i in audit_equivalence(df, target="target")] == ["x"]
    assert audit_combination_leakage(df, target="target") == []


def test_no_triple_false_positives_on_random_data():
    """
    On random data no pair clears the gate, so no triple is ever evaluated —
    which is what keeps the implicit C(k,3) comparisons from mattering.
    """
    rng = np.random.default_rng(5)
    df = pd.DataFrame(
        {
            "target": rng.normal(0, 1, N),
            **{f"f{i}": rng.normal(0, 1, N) for i in range(20)},
        },
        index=IDX,
    )
    assert audit_combination_leakage(df, target="target") == []


def test_triple_not_reported_when_a_pair_already_explains_it(difference_leak):
    """
    x1-x2 is a pair identity. Adding any third column keeps R² at 1.0, but that
    triple is the same finding with a redundant column bolted on.
    """
    df = difference_leak.copy()
    df["extra"] = np.random.default_rng(6).normal(0, 1, N)

    groups = _groups(audit_combination_leakage(df, target="target"))
    assert groups == {("x1", "x2")}
    assert all(len(g) == 2 for g in groups)


def test_cost_stays_low_with_triples_enabled():
    """Triples must not turn the check into a hot spot on clean data."""
    import time

    rng = np.random.default_rng(7)
    df = pd.DataFrame(
        {
            "target": rng.normal(0, 1, N),
            **{f"f{i}": rng.normal(0, 1, N) for i in range(50)},
        },
        index=IDX,
    )
    start = time.time()
    audit_combination_leakage(df, target="target")
    assert time.time() - start < 3.0
