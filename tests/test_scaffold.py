"""
Smoke tests: verify the package imports, dataclasses, validation,
and scanner entry point work correctly before any algorithms are implemented.
"""

import json
import tempfile
import os

import pandas as pd
import pytest

import tsauditor as tsa
from tsauditor.report.summary import GuardReport, Issue, CRITICAL, WARNING, INFO
from tsauditor.utils.validation import validate_dataframe, infer_frequency


# ── Import smoke test ─────────────────────────────────────────────────────────


def test_package_imports():
    assert hasattr(tsa, "scan")
    assert hasattr(tsa, "GuardReport")
    assert hasattr(tsa, "Issue")
    assert hasattr(tsa, "fix")
    assert hasattr(tsa, "adapters")


def test_leakage_package_reexports_every_detector():
    """
    Regression: `audit_combination_leakage` (LEK005) was importable only via
    `tsauditor.leakage.combination`, not `tsauditor.leakage` -- unlike every
    other detector in the package, which is reachable both ways. A user
    following the documented `from tsauditor.leakage import ...` pattern
    would hit an ImportError specifically for LEK005.
    """
    from tsauditor.leakage import (
        audit_correlation_leakage,
        audit_equivalence,
        audit_temporal_leakage,
        audit_asof_leakage,
        audit_combination_leakage,
    )

    for fn in (
        audit_correlation_leakage,
        audit_equivalence,
        audit_temporal_leakage,
        audit_asof_leakage,
        audit_combination_leakage,
    ):
        assert callable(fn)

    import tsauditor.leakage as leakage_pkg

    assert "audit_combination_leakage" in leakage_pkg.__all__


def test_version_is_well_formed_and_consistent():
    """
    The version lives in two places — ``tsauditor/__init__.py`` and
    ``pyproject.toml`` — and they must agree. Checking consistency rather than a
    hardcoded literal means this catches a real bug class (bumping one and
    forgetting the other) without needing an edit every release.
    """
    import pathlib
    import re

    version = tsa.__version__
    assert re.fullmatch(r"\d+\.\d+\.\d+", version), version

    pyproject = pathlib.Path(__file__).resolve().parent.parent / "pyproject.toml"
    if not pyproject.exists():  # pragma: no cover - installed without sources
        pytest.skip("pyproject.toml not available")

    declared = re.search(
        r'^version\s*=\s*"([^"]+)"', pyproject.read_text(encoding="utf-8"), re.M
    )
    assert declared is not None, "no version found in pyproject.toml"
    assert declared.group(1) == version, (
        f"pyproject.toml says {declared.group(1)}, tsauditor.__version__ says {version}"
    )


# ── Issue dataclass ───────────────────────────────────────────────────────────


def test_issue_creation():
    issue = Issue(
        module="leakage",
        code="LEK001",
        severity=CRITICAL,
        description="Target equivalence detected.",
        column="ChangeP",
        evidence={"lag0_corr": 0.99},
    )
    assert issue.code == "LEK001"
    assert issue.severity == CRITICAL
    assert issue.evidence["lag0_corr"] == 0.99


def test_issue_to_dict():
    issue = Issue(
        module="profiler",
        code="PRF001",
        severity=WARNING,
        description="Irregular frequency.",
        column=None,
        evidence={},
    )
    d = issue.to_dict()
    assert d["code"] == "PRF001"
    assert d["column"] is None


# ── GuardReport dataclass ─────────────────────────────────────────────────────


def test_guard_report_empty():
    report = GuardReport(metadata={"rows": 100})
    assert report.critical == []
    assert report.warnings == []
    assert report.info == []
    assert report.all_issues == []


def test_guard_report_filter():
    issues = [
        Issue("leakage", "LEK001", CRITICAL, "Target equivalence.", "ChangeP"),
        Issue("leakage", "LEK002", WARNING, "Positive lag peak.", "RSI"),
        Issue("profiler", "PRF001", WARNING, "Large timestamp gap.", "Price"),
    ]
    report = GuardReport(
        critical=[issues[0]],
        warnings=[issues[1], issues[2]],
        metadata={},
    )
    assert len(report.filter(module="leakage")) == 2
    assert len(report.filter(code="LEK001")) == 1
    assert len(report.filter(severity=CRITICAL)) == 1


def test_guard_report_to_json():
    report = GuardReport(
        critical=[Issue("leakage", "LEK001", CRITICAL, "Test.", "Col")],
        metadata={"rows": 50},
    )
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        report.to_json(path)
        with open(path) as f:
            data = json.load(f)
        assert data["counts"]["critical"] == 1
        assert data["issues"][0]["code"] == "LEK001"
    finally:
        os.unlink(path)


def test_guard_report_to_dict():
    report = GuardReport(metadata={"rows": 10})
    d = report.to_dict()
    assert "metadata" in d
    assert "issues" in d
    assert "counts" in d


def test_guard_report_to_dict_counts_match_to_json():
    report = GuardReport(
        critical=[Issue("leakage", "LEK001", CRITICAL, "Test.", "Col")],
        warnings=[
            Issue("profiler", "PRF001", WARNING, "Irregular frequency.", "Price")
        ],
        info=[Issue("profiler", "PRF003", INFO, "Note.", "Price")],
        metadata={"rows": 10},
    )
    d = report.to_dict()
    assert d["counts"] == {"critical": 1, "warnings": 1, "info": 1}

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        report.to_json(path)
        with open(path) as f:
            json_counts = json.load(f)["counts"]
        assert d["counts"] == json_counts
    finally:
        os.unlink(path)


# ── Validation ────────────────────────────────────────────────────────────────


def test_validate_dataframe_with_datetime_index(clean_financial_df):
    out = validate_dataframe(clean_financial_df, target="Direction", time_col=None)
    assert isinstance(out.index, pd.DatetimeIndex)
    assert out.index.is_monotonic_increasing


def test_validate_dataframe_with_time_col():
    df = pd.DataFrame(
        {
            "Date": pd.date_range("2020-01-01", periods=10),
            "Price": range(10),
            "Target": range(10),
        }
    )
    out = validate_dataframe(df, target="Target", time_col="Date")
    assert isinstance(out.index, pd.DatetimeIndex)
    assert "Date" not in out.columns


def test_validate_dataframe_sort_is_stable_for_duplicate_timestamps():
    """
    Regression. validate_dataframe used to call df.sort_index() with pandas'
    default kind="quicksort", which is not a stable sort -- rows sharing an
    identical duplicate timestamp could come out reordered relative to the
    input in an order that depends on the input's own row order in an
    unspecified way, rather than preserving it. Concretely: the exact same
    logical rows, supplied in two different (equally valid) orders, could
    end up with a different row surviving downstream keep="first" dedup
    (e.g. audit_frequency's PRF004 handling), making repair nondeterministic
    across otherwise-equivalent inputs.

    kind="mergesort" is the sort pandas/numpy document as stable; this pins
    that it's actually used, by checking relative tie order survives a
    shuffle exactly as a stable sort guarantees.
    """
    rng_order = [3, 1, 4, 0, 2]  # an arbitrary shuffle of 5 same-timestamp rows
    dates = [pd.Timestamp("2024-01-01")] * 5 + [pd.Timestamp("2024-01-02")]
    tags = [f"row{i}" for i in range(5)] + ["later"]
    df = pd.DataFrame({"tag": tags}, index=pd.DatetimeIndex(dates))

    shuffled = df.iloc[rng_order + [5]]  # keep the distinct-day row last
    out = validate_dataframe(shuffled, target=None, time_col=None)

    same_day = out.loc["2024-01-01", "tag"].tolist()
    expected = [f"row{i}" for i in rng_order]
    assert same_day == expected


def test_validate_dataframe_rejects_non_dataframe():
    with pytest.raises(TypeError):
        validate_dataframe([1, 2, 3], target=None, time_col=None)


def test_validate_dataframe_rejects_missing_target(clean_financial_df):
    with pytest.raises(ValueError, match="target="):
        validate_dataframe(clean_financial_df, target="NonExistent", time_col=None)


def test_validate_dataframe_rejects_missing_time_col(clean_financial_df):
    df = clean_financial_df.reset_index(drop=True)  # strip datetime index
    with pytest.raises(ValueError):
        validate_dataframe(df, target="Direction", time_col="NonExistent")


def test_validate_dataframe_rejects_empty():
    with pytest.raises(ValueError, match="empty"):
        validate_dataframe(pd.DataFrame(), target=None, time_col=None)


# ── infer_frequency ───────────────────────────────────────────────────────────


def test_infer_frequency_daily(clean_financial_df):
    freq = infer_frequency(clean_financial_df.index)
    assert freq == "daily"


def test_infer_frequency_subhourly(sensor_df):
    freq = infer_frequency(sensor_df.index)
    assert freq == "sub-daily"


def test_infer_frequency_weekly():
    """
    Regression: coverage gap. Every existing infer_frequency test used daily
    or sub-daily data, so the "weekly" branch (median gap 140-196 hours) had
    never actually been exercised by any test in this suite -- verified
    correct by direct call, not just assumed from reading the boundary
    constants.
    """
    idx = pd.date_range("2024-01-01", periods=20, freq="W")
    assert infer_frequency(idx) == "weekly"


def test_infer_frequency_monthly():
    """Regression: coverage gap, same as test_infer_frequency_weekly -- the
    "monthly" branch (median gap 600-960 hours) had never been hit."""
    idx = pd.date_range("2024-01-01", periods=20, freq="MS")
    assert infer_frequency(idx) == "monthly"


def test_infer_frequency_irregular():
    """Regression: coverage gap. The final `return "irregular"` fallback --
    a median gap outside every named bucket -- had never been hit either."""
    idx = pd.DatetimeIndex(["2024-01-01", "2024-01-03", "2024-02-20", "2024-06-01"])
    assert infer_frequency(idx) == "irregular"


# ── Scanner entry point ───────────────────────────────────────────────────────


def test_scan_rejects_invalid_domain(clean_financial_df):
    with pytest.raises(ValueError, match="domain"):
        tsa.scan(clean_financial_df, domain="stock_exchange")


def test_scan_skips_leakage_without_target(clean_financial_df):
    """scan() should run cleanly and skip leakage when target is None."""
    # Leakage is the only module still stubbed (raises NotImplementedError).
    # With target=None the leakage block is skipped, so scan() must complete
    # and return a GuardReport containing no leakage-module issues.
    report = tsa.scan(clean_financial_df, target=None, domain="finance")
    assert isinstance(report, GuardReport)
    assert all(issue.module != "leakage" for issue in report.all_issues)


def test_scan_returns_guard_report(clean_financial_df):
    """With all modules implemented, scan() returns a populated GuardReport."""
    report = tsa.scan(clean_financial_df, target="Direction", domain="finance")
    assert isinstance(report, GuardReport)


def test_scan_forwards_anomaly_tuning_params():
    """
    Regression: previously only `domain=` crossed the boundary into the
    anomaly detectors from scan() -- zscore_threshold, stuck_window,
    spike_threshold, spike_window, and handle_missing were only reachable
    by importing audit_point_anomalies/audit_contextual_anomalies directly.
    A caller with, say, known 1-row sensor dropouts had no way to turn on
    interpolation short of bypassing scan() entirely.

    Proven here via stuck_window specifically: a 4-long run is invisible at
    the default threshold but must be flagged once a caller explicitly
    tightens stuck_window through scan() itself.
    """
    idx = pd.date_range("2024-01-01", periods=20, freq="D")
    values = [100.0 + i * 0.01 for i in range(20)]
    values[5:9] = [7.0, 7.0, 7.0, 7.0]  # a 4-long stuck run
    df = pd.DataFrame({"x": values}, index=idx)

    default_report = tsa.scan(df, run_leakage=False, run_stationarity=False)
    assert default_report.filter(code="ANO001") == []  # 4 < default stuck_window (5)

    tuned_report = tsa.scan(
        df, run_leakage=False, run_stationarity=False, stuck_window=3
    )
    assert len(tuned_report.filter(code="ANO001")) == 1


def test_scan_forwards_handle_missing_to_ano001_bridging():
    """
    handle_missing is documented on scan() now; confirm it actually reaches
    audit_contextual_anomalies rather than being silently dropped somewhere
    in _run_checks. ANO001 bridges a single-row gap regardless of
    handle_missing (see contextual.py), so this checks the parameter is
    wired through by observing scan() doesn't raise and still finds the
    bridged run under either setting.
    """
    idx = pd.date_range("2024-01-01", periods=11, freq="D")
    values = [1.0] * 5 + [float("nan")] + [1.0] * 5
    df = pd.DataFrame({"x": values}, index=idx)

    for hm in ("strict", "interpolate"):
        report = tsa.scan(
            df,
            run_leakage=False,
            run_stationarity=False,
            stuck_window=5,
            handle_missing=hm,
        )
        stuck = report.filter(code="ANO001")
        assert len(stuck) == 1
        assert stuck[0].evidence["max_stuck_duration"] == 11
