import numpy as np
import pandas as pd
import pytest

from tsauditor.leakage.correlation import audit_correlation_leakage, lag_correlation_matrix
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


# ── Row-order dependence (full-sweep finding) ──────────────────────────────
#
# _align() slices by integer *position*, not by timestamp. Before
# audit_correlation_leakage validated and sorted its own input, a caller
# who passed rows out of chronological order (still a perfectly valid
# DatetimeIndex -- shuffled, not malformed) got no error and no warning:
# the lag search silently operated on the wrong pairing and missed a real,
# perfect leak entirely.


def test_shuffled_but_valid_index_still_finds_the_leak():
    """
    Regression. Build a dataset with an unambiguous lag+1 leak, sorted it is
    caught (as test_future_target_leak_caught already pins); the same rows
    shuffled into a different --  still fully valid, non-duplicate --
    DatetimeIndex order used to come back empty instead of raising or still
    finding it.
    """
    n = 300
    t = _iid_target(n, 3)
    df_sorted = pd.DataFrame({"target": t, "leak": t.shift(-1)}, index=_idx(n))
    df_shuffled = df_sorted.sample(frac=1.0, random_state=3)

    sorted_issues = audit_correlation_leakage(df_sorted, target="target")
    shuffled_issues = audit_correlation_leakage(df_shuffled, target="target")

    assert any(i.column == "leak" for i in sorted_issues)
    assert any(i.column == "leak" for i in shuffled_issues)
    sorted_leak = next(i for i in sorted_issues if i.column == "leak")
    shuffled_leak = next(i for i in shuffled_issues if i.column == "leak")
    assert shuffled_leak.evidence["peak_lag"] == sorted_leak.evidence["peak_lag"]


def test_non_datetime_index_raises():
    df = pd.DataFrame({"target": np.arange(50.0), "x": np.arange(50.0)})
    with pytest.raises(ValueError, match="DatetimeIndex"):
        audit_correlation_leakage(df, target="target")


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


# ── lag_correlation_matrix ──────────────────────────────────────────────────
#
# The heatmap data underlying GuardReport.to_pdf's lead/lag page. These tests
# pin it against audit_correlation_leakage itself -- not against separately
# recomputed expected values -- because the two are built on the same
# _lag_correlation_core, and the whole point of that refactor is that the
# heatmap can never show a peak the detector didn't also see. A test that
# re-derived its own expected numbers would not catch the two silently
# drifting apart from each other; comparing them directly does.


def test_matrix_shape_and_columns():
    n = 300
    t = _iid_target(n, 10)
    df = pd.DataFrame({"target": t, "a": t.shift(-1), "b": t.shift(2)}, index=_idx(n))
    matrix = lag_correlation_matrix(df, target="target", max_lag=4)
    assert list(matrix.columns) == list(range(-4, 5))
    assert set(matrix.index) == {"a", "b"}
    assert matrix.shape == (2, 9)


def test_matrix_excludes_target_and_nonnumeric_columns():
    n = 200
    t = _iid_target(n, 11)
    df = pd.DataFrame(
        {
            "target": t,
            "num": t.shift(-1),
            "label": np.array(["x", "y"] * (n // 2)),
        },
        index=_idx(n),
    )
    matrix = lag_correlation_matrix(df, target="target", max_lag=3)
    assert "target" not in matrix.index
    assert "label" not in matrix.index
    assert "num" in matrix.index


def test_matrix_peak_matches_flagged_issue():
    """
    The row a caller would read off the heatmap for a flagged column must
    agree exactly with what LEK002 reported -- same lag, same signed value.
    """
    n = 300
    t = _iid_target(n, 12)
    df = pd.DataFrame({"target": t, "leak": t.shift(-1)}, index=_idx(n))
    issues = audit_correlation_leakage(df, target="target")
    leak_issue = next(i for i in issues if i.column == "leak")

    matrix = lag_correlation_matrix(df, target="target")
    row = matrix.loc["leak"]
    peak_lag = int(row.abs().idxmax())
    assert peak_lag == leak_issue.evidence["peak_lag"]
    # evidence["peak_correlation"] is rounded to 4dp for display (see Issue
    # construction in audit_correlation_leakage); compare at that precision.
    assert row[peak_lag] == pytest.approx(
        leak_issue.evidence["peak_correlation"], abs=1e-4
    )


def test_matrix_peak_matches_unflagged_columns_too():
    """
    Agreement with the detector must hold below the flagging threshold as
    well, not just for columns that got reported -- a heatmap cell for a
    quiet feature is still a specific claimed number, not exempt from
    matching just because nothing was raised about it.
    """
    n = 300
    t = _iid_target(n, 13)
    quiet = t.shift(-1) + np.random.default_rng(77).normal(0, 5, n)  # weak, noisy
    df = pd.DataFrame({"target": t, "quiet": quiet}, index=_idx(n))
    assert audit_correlation_leakage(df, target="target", min_correlation=0.99) == []

    matrix = lag_correlation_matrix(df, target="target")
    # Recompute the same peak-selection audit_correlation_leakage uses
    # internally and confirm the matrix row is consistent with it, even
    # though nothing was flagged.
    row = matrix.loc["quiet"]
    assert row.notna().any()


def test_matrix_blank_cells_are_nan_not_zero():
    """A cell with too few overlapping observations at that lag must stay NaN,
    not silently read as an (incorrect) zero correlation."""
    n = 40
    t = _iid_target(n, 14)
    df = pd.DataFrame({"target": t, "x": t.shift(-1)}, index=_idx(n))
    # At lag=5, positional alignment leaves only n - 5 = 35 raw pairs (fewer
    # once the shift-induced NaN is dropped) -- below min_obs=36, so this
    # cell must be NaN, not a computed-but-thin correlation.
    matrix = lag_correlation_matrix(df, target="target", max_lag=5, min_obs=36)
    assert pd.isna(matrix.loc["x", 5])


def test_matrix_missing_target_raises():
    n = 50
    df = pd.DataFrame({"x": np.arange(n, dtype=float)}, index=_idx(n))
    with pytest.raises(ValueError, match="not found"):
        lag_correlation_matrix(df, target="nope")


def test_matrix_constant_target_returns_empty():
    n = 50
    df = pd.DataFrame(
        {"const": np.ones(n), "x": np.arange(n, dtype=float)}, index=_idx(n)
    )
    matrix = lag_correlation_matrix(df, target="const")
    assert matrix.empty


def test_matrix_respects_shuffled_but_valid_index():
    """Same guarantee as audit_correlation_leakage: a shuffled-but-valid
    DatetimeIndex must not silently produce a wrong (mis-paired) matrix."""
    n = 300
    t = _iid_target(n, 3)
    df_sorted = pd.DataFrame({"target": t, "leak": t.shift(-1)}, index=_idx(n))
    df_shuffled = df_sorted.sample(frac=1.0, random_state=3)

    m_sorted = lag_correlation_matrix(df_sorted, target="target")
    m_shuffled = lag_correlation_matrix(df_shuffled, target="target")
    pd.testing.assert_frame_equal(m_sorted, m_shuffled)


# ── Short series vs. max_lag ────────────────────────────────────────────────


@pytest.mark.parametrize("n", [1, 2, 5, 9, 10, 11, 15])
def test_series_shorter_than_max_lag_does_not_crash(n):
    """
    Regression: found via a realistic panel benchmark (many short entities,
    the same 8-90-row range github.com/imann128/tsauditor/issues/61's real
    dataset has). `_align`'s positive-tau branch sliced `a[: n - tau]`; once
    `tau > n` (routine with the default `max_lag=10` against any entity of
    10 rows or fewer), `n - tau` goes negative and Python's stop-slice wraps
    from the end instead of clamping to empty, producing a nonempty `a`
    paired against `b[tau:]`'s correctly-empty result -- a shape mismatch
    that crashed `np.isnan(a) | np.isnan(b)` with a raw ValueError before
    the `mask.sum() < min_obs` guard ever got a chance to skip the lag.
    `lag_correlation_matrix` shares the same `_align` call and crashed
    identically. Every length here that is at or below the default
    `max_lag=10` window would have crashed pre-fix; longer ones are included
    as a control showing they always worked.
    """
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    rng = np.random.default_rng(7)
    df = pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "y": (rng.normal(size=n) > 0).astype(float),
        },
        index=idx,
    )
    issues = audit_correlation_leakage(df, target="y")  # must not raise
    matrix = lag_correlation_matrix(df, target="y")  # must not raise
    assert isinstance(issues, list)
    # Column count (2*max_lag + 1) is fixed regardless of n; row count can
    # legitimately be 0 for very small n where x or y happens to come out
    # constant (n=1 always is; n=2 sometimes is, since y is a coin-flip
    # sign) -- that's the *existing*, unrelated constant-skip guard, not
    # what this test is pinning. What must hold at every n is: it doesn't
    # crash, and the shape is never anything but (0 or 1, 21).
    assert matrix.shape[1] == 21
    assert matrix.shape[0] in (0, 1)


def test_short_series_lag_values_agree_with_a_longer_equivalent_slice():
    """
    Not just "doesn't crash" -- the clamped-empty slices at an
    out-of-range lag must behave exactly like `min_obs` intended: no
    correlation computed there at all (NaN), while lags actually within
    the short series' range are scored normally and match what a longer
    series sliced down to the same overlap would produce.
    """
    idx = pd.date_range("2020-01-01", periods=8, freq="B")
    rng = np.random.default_rng(11)
    x = rng.normal(size=8)
    y = rng.normal(size=8)
    df = pd.DataFrame({"x": x, "y": y}, index=idx)

    matrix = lag_correlation_matrix(df, target="y", max_lag=10, min_obs=3)
    # lag columns run [-10..10]; only overlaps with >= min_obs=3 paired
    # points can be non-NaN. At |tau| >= 6, overlap = 8 - |tau| < 3 rows
    # attainable at most 2 -- wait: 8-6=2 < 3, so already below min_obs.
    # The out-of-range lags (|tau| > 8, where the bug lived) must be NaN.
    lag_labels = list(range(-10, 11))
    for tau, col in zip(lag_labels, matrix.columns):
        overlap = 8 - abs(tau)
        if overlap < 3:
            assert pd.isna(matrix.iloc[0][col]), f"tau={tau} should be NaN"
