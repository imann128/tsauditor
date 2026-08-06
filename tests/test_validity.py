import numpy as np
import pandas as pd
import pytest

from tsauditor.validity import audit_validity
from tsauditor.report.summary import WARNING, CRITICAL


def _idx(n):
    return pd.date_range("2020-01-01", periods=n, freq="D")


# ── Bounds (VAL001) ───────────────────────────────────────────────────────────


def test_clean_within_bounds_no_issue():
    df = pd.DataFrame({"sentiment": np.linspace(-1, 1, 50)}, index=_idx(50))
    assert audit_validity(df, bounds={"sentiment": {"min": -1, "max": 1}}) == []


def test_out_of_range_flagged():
    s = np.linspace(-1, 1, 50)
    s[10] = 1.5  # above max
    s[20] = -2.0  # below min
    df = pd.DataFrame({"sentiment": s}, index=_idx(50))
    issues = audit_validity(df, bounds={"sentiment": {"min": -1, "max": 1}})
    assert len(issues) == 1
    iss = issues[0]
    assert iss.code == "VAL001" and iss.severity == WARNING
    assert iss.evidence["n_violations"] == 2
    assert iss.evidence["observed_max"] == 1.5
    assert iss.evidence["observed_min"] == -2.0


def test_exclusive_min_catches_zero_spread():
    """A spread must be strictly positive: 0 and negatives are glitches."""
    spread = np.full(30, 0.5)
    spread[5] = 0.0  # locked book
    spread[6] = -0.1  # crossed (negative spread)
    df = pd.DataFrame({"spread": spread}, index=_idx(30))
    issues = audit_validity(df, bounds={"spread": {"min": 0, "min_exclusive": True}})
    assert issues[0].evidence["n_violations"] == 2


def test_inclusive_bound_allows_boundary_value():
    spread = np.full(10, 0.0)  # exactly at the inclusive lower bound
    df = pd.DataFrame({"spread": spread}, index=_idx(10))
    assert audit_validity(df, bounds={"spread": {"min": 0}}) == []


def test_inclusive_max_allows_boundary_value():
    """Mirrors test_inclusive_bound_allows_boundary_value for the upper bound,
    which nothing previously exercised."""
    spread = np.full(10, 1.0)  # exactly at the inclusive upper bound
    df = pd.DataFrame({"spread": spread}, index=_idx(10))
    assert audit_validity(df, bounds={"spread": {"max": 1.0}}) == []


def test_exclusive_max_catches_boundary_value():
    """
    Mirrors test_exclusive_min_catches_zero_spread for the upper bound.

    Mutation-checked: before this test, changing the max_exclusive comparison
    from `>=` to `>` (making it behave as if always inclusive) left every test
    in this file passing, because nothing used max_exclusive at all.
    """
    spread = np.full(30, 0.5)
    spread[5] = 1.0  # exactly at the bound: a violation only if exclusive
    df = pd.DataFrame({"spread": spread}, index=_idx(30))
    issues = audit_validity(df, bounds={"spread": {"max": 1.0, "max_exclusive": True}})
    assert len(issues) == 1
    assert issues[0].evidence["n_violations"] == 1
    assert issues[0].evidence["observed_max"] == 1.0


def test_nan_not_counted_as_violation():
    s = np.linspace(-1, 1, 20)
    s[3] = np.nan
    df = pd.DataFrame({"sentiment": s}, index=_idx(20))
    assert audit_validity(df, bounds={"sentiment": {"min": -1, "max": 1}}) == []


# ── Relations (VAL002) ────────────────────────────────────────────────────────


def test_crossed_book_flagged():
    bid = np.full(40, 100.0)
    ask = np.full(40, 100.2)
    ask[15] = 99.8  # ask below bid -> crossed book
    df = pd.DataFrame({"bid": bid, "ask": ask}, index=_idx(40))
    issues = audit_validity(df, relations=[("bid", "ask")])
    assert len(issues) == 1
    iss = issues[0]
    assert iss.code == "VAL002" and iss.severity == CRITICAL
    assert iss.column == "ask"
    assert iss.evidence["n_violations"] == 1
    assert iss.evidence["low_col"] == "bid" and iss.evidence["high_col"] == "ask"


def test_equal_values_not_a_violation():
    df = pd.DataFrame(
        {"bid": np.full(10, 5.0), "ask": np.full(10, 5.0)}, index=_idx(10)
    )
    assert audit_validity(df, relations=[("bid", "ask")]) == []


# ── Integration through scan() ────────────────────────────────────────────────


def test_scan_runs_validity_and_excludes_from_leaky():
    import tsauditor as tsa

    bid = np.full(60, 100.0)
    ask = np.full(60, 100.2)
    ask[30] = 99.0
    df = pd.DataFrame({"bid": bid, "ask": ask}, index=_idx(60))
    report = tsa.scan(
        df, constraints={"relations": [("bid", "ask")]}, run_stationarity=False
    )
    assert any(i.code == "VAL002" for i in report.critical)
    assert report.leaky_columns() == []  # validity is not leakage


def test_scan_flat_bounds_mapping_treated_as_bounds():
    import tsauditor as tsa

    s = np.full(60, 0.5)
    s[10] = 5.0
    df = pd.DataFrame({"vol": s}, index=_idx(60))
    report = tsa.scan(df, constraints={"vol": {"max": 1}}, run_stationarity=False)
    assert any(i.code == "VAL001" for i in report.all_issues)


# ── Flat/nested dispatch: "bounds"/"relations" as real column names ────────────
#
# scan() has to guess whether `constraints` is the nested {"bounds": ...,
# "relations": ...} form or a flat {col: spec} shorthand. It used to guess by
# key presence alone (`.get("bounds")` / `.get("relations")` both None =>
# flat), which broke the moment a real column happened to be named "bounds"
# or "relations" -- a flat dict bounding such a column was misread as the
# nested form, and audit_validity crashed unpacking a spec dict as a
# (low, high) pair. These pin the structural fix: distinguish by shape
# (nested "bounds" is dict-of-dicts, nested "relations" is a list of pairs;
# neither shape a flat per-column spec can ever take), not by key name.


def test_scan_flat_bounds_column_named_relations_no_longer_crashes():
    """
    Regression. {"spread": {...}, "relations": {"min": 0}} is a flat bounds
    dict for two columns, one of them unfortunately named "relations". This
    used to crash with `ValueError: too many values to unpack (expected 2)`
    because opts.constraints.get("relations") returned the spec dict
    {"min": 0}, which is not None, so the flat-dict fallback never fired and
    audit_validity received it as a relations pair to unpack.
    """
    import tsauditor as tsa

    spread = np.full(20, 0.5)
    spread[3] = -0.5  # violates spread's min
    relations_col = np.full(20, 5.0)
    relations_col[7] = -1.0  # violates relations_col's min
    df = pd.DataFrame({"spread": spread, "relations": relations_col}, index=_idx(20))

    report = tsa.scan(
        df,
        constraints={
            "spread": {"min": 0, "min_exclusive": True},
            "relations": {"min": 0},
        },
        run_stationarity=False,
    )
    val001 = report.filter(code="VAL001")
    assert {i.column for i in val001} == {"spread", "relations"}
    assert report.filter(code="VAL002") == []  # no relation pairs were declared


def test_scan_flat_bounds_column_named_bounds_no_longer_crashes():
    """
    Mirrors the "relations" case for a column literally named "bounds". This
    used to fail differently: opts.constraints.get("bounds") returned the
    spec dict for that column, the flat-dict fallback never fired, and
    audit_validity's bounds loop tried to call .get() on the spec's own
    scalar values instead of a per-column spec dict.
    """
    import tsauditor as tsa

    volume = np.full(20, 100.0)
    volume[2] = -5.0  # violates volume's min
    bounds_col = np.full(20, 1.0)
    bounds_col[9] = -1.0  # violates bounds_col's min
    df = pd.DataFrame({"volume": volume, "bounds": bounds_col}, index=_idx(20))

    report = tsa.scan(
        df,
        constraints={"volume": {"min": 0}, "bounds": {"min": 0}},
        run_stationarity=False,
    )
    val001 = report.filter(code="VAL001")
    assert {i.column for i in val001} == {"volume", "bounds"}


def test_scan_nested_bounds_and_relations_together_still_works():
    """
    The genuine nested form, with both keys present at once, must still
    dispatch correctly now that detection is structural rather than
    presence-based.
    """
    import tsauditor as tsa

    bid = np.full(30, 100.0)
    ask = np.full(30, 100.2)
    ask[5] = 99.0  # crossed book
    spread = ask - bid
    spread[10] = -0.5  # out-of-range spread (separate from the crossed row)
    df = pd.DataFrame({"bid": bid, "ask": ask, "spread": spread}, index=_idx(30))

    report = tsa.scan(
        df,
        constraints={
            "bounds": {"spread": {"min": 0, "min_exclusive": True}},
            "relations": [("bid", "ask")],
        },
        run_stationarity=False,
    )
    assert any(i.code == "VAL001" and i.column == "spread" for i in report.all_issues)
    assert any(i.code == "VAL002" for i in report.critical)


# ── Edge cases ────────────────────────────────────────────────────────────────


def test_missing_column_raises():
    df = pd.DataFrame({"a": np.arange(10.0)}, index=_idx(10))
    with pytest.raises(ValueError, match="not found"):
        audit_validity(df, bounds={"nope": {"min": 0}})


def test_non_numeric_column_raises():
    df = pd.DataFrame({"a": ["x"] * 10}, index=_idx(10))
    with pytest.raises(ValueError, match="not numeric"):
        audit_validity(df, bounds={"a": {"min": 0}})


def test_empty_rules_return_empty():
    df = pd.DataFrame({"a": np.arange(10.0)}, index=_idx(10))
    assert audit_validity(df) == []


def test_multiple_relation_pairs_are_independent():
    """Each declared pair is checked independently; a violation in one does not
    suppress or affect another."""
    df = pd.DataFrame(
        {"a": np.full(20, 1.0), "b": np.full(20, 2.0), "c": np.full(20, 0.5)},
        index=_idx(20),
    )
    issues = audit_validity(df, relations=[("a", "b"), ("b", "c")])
    assert len(issues) == 1
    assert issues[0].evidence["low_col"] == "b"
    assert issues[0].evidence["high_col"] == "c"
