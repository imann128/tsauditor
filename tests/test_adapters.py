import numpy as np
import pandas as pd
import pytest

from tsauditor.adapters import to_timesfm
from tsauditor.report.summary import GuardReport


def _idx(n):
    return pd.date_range("2020-01-01", periods=n, freq="D")


# ── Happy path ────────────────────────────────────────────────────────────────


def test_returns_1d_float32_array():
    df = pd.DataFrame({"y": np.linspace(1, 2, 100)}, index=_idx(100))
    arr = to_timesfm(df, "y")
    assert isinstance(arr, np.ndarray)
    assert arr.ndim == 1
    assert arr.dtype == np.float32
    assert len(arr) == 100


def test_cleans_the_target_column():
    """A clustered missing gap in the forecast series is repaired to finite,
    with a sane (interpolated) value, not merely to *some* finite value."""
    v = np.linspace(10, 20, 200)
    v[40:60] = np.nan  # clustered gap -> flagged and imputed
    df = pd.DataFrame({"y": v}, index=_idx(200))
    arr = to_timesfm(df, "y")
    assert np.isfinite(arr).all()
    # The gap sits on a linear ramp roughly between v[39] (~12.0) and
    # v[60] (~13.0); a correct time-interpolation lands inside that band. A
    # broken fill (e.g. dropping to 0.0, or extrapolating unboundedly) would
    # fail this without failing the isfinite check above.
    gap = arr[40:60]
    assert gap.min() > 11.5
    assert gap.max() < 13.5


def test_truncates_to_context_len_keeping_most_recent():
    df = pd.DataFrame({"y": np.arange(2000.0)}, index=_idx(2000))
    arr = to_timesfm(df, "y", context_len=1024)
    assert len(arr) == 1024
    assert arr[-1] == 1999.0  # most recent kept
    assert arr[0] == 2000.0 - 1024.0  # oldest inside the window


def test_return_report_gives_audit_trail():
    v = np.linspace(10, 20, 200)
    v[40:60] = np.nan
    df = pd.DataFrame({"y": v}, index=_idx(200))
    arr, report = to_timesfm(df, "y", return_report=True)
    assert isinstance(report, GuardReport)
    assert np.isfinite(arr).all()
    assert any(e["column"] == "y" for e in report.last_fixes)  # repair recorded


# ── Guards ────────────────────────────────────────────────────────────────────


def test_stray_unflagged_nan_raises_rather_than_leaking_to_model():
    """A lone NaN isn't a flagged cluster, so fix() leaves it; the adapter must
    refuse rather than hand a NaN to TimesFM."""
    v = np.linspace(1, 2, 100)
    v[50] = np.nan
    df = pd.DataFrame({"y": v}, index=_idx(100))
    with pytest.raises(ValueError, match="non-finite"):
        to_timesfm(df, "y")


def test_too_short_raises():
    df = pd.DataFrame({"y": np.arange(20.0)}, index=_idx(20))
    with pytest.raises(ValueError, match="minimum"):
        to_timesfm(df, "y", min_context=32)


def test_missing_column_raises():
    df = pd.DataFrame({"y": np.arange(50.0)}, index=_idx(50))
    with pytest.raises(KeyError):
        to_timesfm(df, "nope")


def test_non_numeric_target_col_raises_clearly():
    """
    Regression. A non-numeric target_col used to fail much later and much
    less clearly: remediate.py's own numeric-dtype guards silently pass a
    string/categorical column through unrepaired, and
    `clean[target_col].to_numpy(dtype=np.float32)` then raised a raw
    `ValueError: could not convert string to float: '...'` with no mention
    of target_col or what to do about it -- inconsistent with the adapter's
    own stated design (raise clearly rather than let something unusable
    reach the model silently, which is exactly what the finite-value guard
    already does for NaNs).
    """
    df = pd.DataFrame({"y": ["a", "b", "c"] * 20}, index=_idx(60))
    with pytest.raises(TypeError, match="not numeric"):
        to_timesfm(df, "y")


def test_accessible_via_top_level_namespace():
    import tsauditor as tsa

    df = pd.DataFrame({"y": np.linspace(0, 1, 60)}, index=_idx(60))
    arr = tsa.adapters.to_timesfm(df, "y")
    assert len(arr) == 60
