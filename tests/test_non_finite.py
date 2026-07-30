"""
PRF007 — infinite values (#46).

Before this check existed, an inf was reported by nothing and repaired by
nothing: `isna()` is False for it so the missing-data checks never saw it, and
every anomaly and leakage detector replaced it with NaN on its own working copy
so its arithmetic would not break. A user could run scan(), see no relevant
issue, run fix(), and still hand infinities to their model.
"""

import numpy as np
import pandas as pd
import pytest

import tsauditor as tsa
from tsauditor.profiler.missing import audit_non_finite
from tsauditor.report.summary import CRITICAL


def _idx(n):
    return pd.date_range("2024-01-01", periods=n, freq="D")


def _col(values):
    return pd.DataFrame({"x": np.asarray(values, dtype=float)}, index=_idx(len(values)))


def _clean(n=200, seed=0):
    return np.random.default_rng(seed).normal(100, 5, n)


# ── Detection ─────────────────────────────────────────────────────────────────


def test_no_false_positive_on_clean_column():
    assert audit_non_finite(_col(_clean())) == []


def test_nan_alone_is_not_reported_as_non_finite():
    """PRF007 must not duplicate the missing-data checks. isna() and isinf()
    are disjoint, and only the latter belongs here."""
    v = _clean()
    v[10:20] = np.nan
    assert audit_non_finite(_col(v)) == []


@pytest.mark.parametrize("bad", [np.inf, -np.inf], ids=["posinf", "neginf"])
def test_single_infinity_is_flagged(bad):
    """No rate threshold: one infinity is a defect, because an infinity is
    never a measurement."""
    v = _clean()
    v[42] = bad
    issues = audit_non_finite(_col(v))
    assert len(issues) == 1
    assert issues[0].code == "PRF007"
    assert issues[0].severity == CRITICAL
    assert issues[0].evidence["non_finite_count"] == 1


def test_signs_counted_separately():
    v = _clean()
    v[5] = np.inf
    v[6] = np.inf
    v[7] = -np.inf
    ev = audit_non_finite(_col(v))[0].evidence
    assert ev["positive_inf_count"] == 2
    assert ev["negative_inf_count"] == 1
    assert ev["non_finite_count"] == 3


def test_counts_are_additive_with_nan():
    """A column with both must report only the infs, and the finite count must
    exclude both categories."""
    v = _clean(n=100)
    v[0:10] = np.nan
    v[20:25] = np.inf
    ev = audit_non_finite(_col(v))[0].evidence
    assert ev["non_finite_count"] == 5
    assert ev["n_finite_remaining"] == 85


def test_below_leakage_min_obs_reported():
    """The key downstream consequence: under 30 finite observations the leakage
    detectors skip the column silently rather than degrading."""
    v = np.full(40, np.inf)
    v[:15] = _clean(n=15)
    ev = audit_non_finite(_col(v))[0].evidence
    assert ev["n_finite_remaining"] == 15
    assert ev["below_leakage_min_obs"] is True

    v2 = _clean(n=200).copy()
    v2[0] = np.inf
    assert audit_non_finite(_col(v2))[0].evidence["below_leakage_min_obs"] is False


def test_exactly_leakage_min_obs_is_not_below():
    """The gate is `finite_remaining < _LEAKAGE_MIN_OBS` (30), so exactly 30
    finite values remaining must read as NOT below the minimum."""
    v = np.full(31, np.inf)
    v[:30] = _clean(n=30)
    ev = audit_non_finite(_col(v))[0].evidence
    assert ev["n_finite_remaining"] == 30
    assert ev["below_leakage_min_obs"] is False


def test_multiple_columns_all_reported():
    n = 100
    a, b, c = _clean(n), _clean(n, 1), _clean(n, 2)
    a[3] = np.inf
    b[4] = -np.inf
    df = pd.DataFrame({"a": a, "b": b, "c": c}, index=_idx(n))
    assert {i.column for i in audit_non_finite(df)} == {"a", "b"}


def test_non_datetime_index_raises():
    with pytest.raises(ValueError, match="DatetimeIndex"):
        audit_non_finite(pd.DataFrame({"x": [np.inf, 1.0]}))


def test_empty_frame_returns_empty():
    assert audit_non_finite(pd.DataFrame(index=pd.DatetimeIndex([]))) == []


# ── Wiring and repair ─────────────────────────────────────────────────────────


def test_scan_reports_it():
    v = _clean()
    v[100:110] = np.inf
    assert any(
        i.code == "PRF007" for i in tsa.scan(_col(v), run_stationarity=False).all_issues
    )


def test_apply_fixes_removes_infinities():
    v = _clean()
    v[100:110] = np.inf
    v[50] = -np.inf
    df = _col(v)
    report = tsa.scan(df, run_stationarity=False)
    fixed = report.apply_fixes(df)
    assert np.isinf(fixed["x"]).sum() == 0
    assert fixed["x"].isna().sum() == 0
    assert np.isinf(df["x"]).sum() == 11, "input must not be mutated"


def test_fix_removes_infinities():
    v = _clean()
    v[100:110] = np.inf
    clean_df, _ = tsa.fix(_col(v))
    assert np.isinf(clean_df["x"]).sum() == 0


def test_infinities_become_nan_when_imputation_disabled():
    """With missing=None the cell is left NaN rather than inf. NaN is honest
    about the value being unknown; inf is a false statement about its size."""
    v = _clean()
    v[100:110] = np.inf
    df = _col(v)
    fixed = tsa.scan(df, run_stationarity=False).apply_fixes(df, missing=None)
    assert np.isinf(fixed["x"]).sum() == 0
    assert fixed["x"].isna().sum() == 10


def test_unflagged_column_untouched():
    n = 200
    a, b = _clean(n), _clean(n, 1)
    a[10] = np.inf
    df = pd.DataFrame({"a": a, "b": b}, index=_idx(n))
    report = tsa.scan(df, run_stationarity=False)
    fixed = report.apply_fixes(df, outliers=None, stuck=None)
    pd.testing.assert_series_equal(fixed["b"], df["b"])


def test_remediation_template_renders():
    v = _clean()
    v[7] = np.inf
    issue = audit_non_finite(_col(v))[0]
    text = issue.suggestion
    assert "infinite" in text
    assert "{" not in text, "template placeholder left unrendered"


def test_interpolation_does_not_spread_infinities():
    """
    The strongest reason the conversion runs before imputation.

    `interpolate` filling a NaN that neighbours an infinity carries the infinity
    into the gap, so repairing a frame could end with *more* infinities than it
    started with. Measured on real OGDC-derived features, 19 became 35.
    """
    v = _clean(n=200)
    v[100] = np.inf
    v[101:110] = np.nan  # a gap sitting directly against the infinity
    df = _col(v)

    fixed = tsa.scan(df, run_stationarity=False).apply_fixes(df)

    assert np.isinf(fixed["x"]).sum() == 0
    assert fixed["x"].isna().sum() == 0
    assert np.isfinite(fixed["x"]).all()
