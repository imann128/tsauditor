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
    assert tsa.__version__ == "0.1.0"


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
        Issue("leakage",  "LEK001", CRITICAL, "Target equivalence.", "ChangeP"),
        Issue("leakage",  "LEK002", WARNING,  "Positive lag peak.",  "RSI"),
        Issue("profiler", "PRF003", WARNING,  "Non-stationary.",     "Price"),
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


# ── Validation ────────────────────────────────────────────────────────────────

def test_validate_dataframe_with_datetime_index(clean_financial_df):
    out = validate_dataframe(clean_financial_df, target="Direction", time_col=None)
    assert isinstance(out.index, pd.DatetimeIndex)
    assert out.index.is_monotonic_increasing


def test_validate_dataframe_with_time_col():
    df = pd.DataFrame({
        "Date":   pd.date_range("2020-01-01", periods=10),
        "Price":  range(10),
        "Target": range(10),
    })
    out = validate_dataframe(df, target="Target", time_col="Date")
    assert isinstance(out.index, pd.DatetimeIndex)
    assert "Date" not in out.columns


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


# ── Scanner entry point ───────────────────────────────────────────────────────

def test_scan_rejects_invalid_domain(clean_financial_df):
    with pytest.raises(ValueError, match="domain"):
        tsa.scan(clean_financial_df, domain="stock_exchange")


def test_scan_skips_leakage_without_target(clean_financial_df):
    """scan() should not crash when target is None and run_leakage=True."""
    # All modules will raise NotImplementedError once implemented;
    # for now we just verify the domain/target guard works cleanly.
    with pytest.raises(NotImplementedError):
        tsa.scan(clean_financial_df, target=None, domain="finance")


def test_scan_returns_guard_report(clean_financial_df):
    """Once profiler is implemented this will return a real report.
    For now verify the return type when modules are stubbed out."""
    with pytest.raises(NotImplementedError):
        tsa.scan(clean_financial_df, target="Direction", domain="finance")
