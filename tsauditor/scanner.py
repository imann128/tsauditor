"""
tsauditor.scanner
-----------------
The main entry point. scan() orchestrates all audit modules and
assembles a GuardReport.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from tsauditor.report.summary import GuardReport, Issue, CRITICAL, WARNING, INFO
from tsauditor.utils.validation import validate_dataframe, infer_frequency


def scan(
    df: pd.DataFrame,
    target: Optional[str] = None,
    time_col: Optional[str] = None,
    domain: Optional[str] = None,
    # Fine-grained toggles — all enabled by default
    run_profiler:  bool = True,
    run_anomaly:   bool = True,
    run_leakage:   bool = True,
) -> GuardReport:
    """
    Audit a time-series DataFrame for data quality issues.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame. Must have a DatetimeIndex or a datetime column
        specified via time_col.
    target : Optional[str]
        Name of the target/label column. Required for leakage detection.
        If None, leakage checks are skipped.
    time_col : Optional[str]
        Name of a datetime column to use as the index. If None, the
        DataFrame index must already be a DatetimeIndex.
    domain : Optional[str]
        Domain hint for domain-specific heuristics.
        One of: "finance", "sensor", None.
    run_profiler : bool
        Run structural profiling checks. Default True.
    run_anomaly : bool
        Run anomaly detection checks. Default True.
    run_leakage : bool
        Run leakage detection checks. Default True.
        Silently skipped if target is None.

    Returns
    -------
    GuardReport
        Structured report with critical issues, warnings, and info.

    Examples
    --------
    >>> import tsauditor as tsa
    >>> report = tsa.scan(df, target="Direction", domain="finance")
    >>> report.summary()
    >>> report.to_json("report.json")
    """
    # ── Validate domain argument ──────────────────────────────────────────────
    valid_domains = {"finance", "sensor", None}
    if domain not in valid_domains:
        raise ValueError(
            f"domain must be one of {valid_domains}, got '{domain}'."
        )

    # ── Validate and normalize input ──────────────────────────────────────────
    df = validate_dataframe(df, target=target, time_col=time_col)

    # ── Build metadata ────────────────────────────────────────────────────────
    metadata = {
        "rows":       len(df),
        "columns":    len(df.columns),
        "time_start": str(df.index.min().date()),
        "time_end":   str(df.index.max().date()),
        "frequency":  infer_frequency(df.index),
        "target":     target,
        "domain":     domain,
    }

    report = GuardReport(metadata=metadata)

    # ── Profiler ──────────────────────────────────────────────────────────────
    if run_profiler:
        from tsauditor.profiler import (
            audit_frequency,
            audit_stationarity,
            audit_missing,
        )
        report.critical += [
            i for i in audit_frequency(df, domain=domain)
            if i.severity == CRITICAL
        ]
        report.warnings += [
            i for i in audit_frequency(df, domain=domain)
            if i.severity == WARNING
        ]
        report.info += [
            i for i in audit_frequency(df, domain=domain)
            if i.severity == INFO
        ]

        for issue in audit_stationarity(df, domain=domain):
            _append_issue(report, issue)

        for issue in audit_missing(df, domain=domain):
            _append_issue(report, issue)

    # ── Anomaly ───────────────────────────────────────────────────────────────
    if run_anomaly:
        from tsauditor.anomaly import (
            audit_point_anomalies,
            audit_contextual_anomalies,
        )
        for issue in audit_point_anomalies(df, domain=domain):
            _append_issue(report, issue)

        for issue in audit_contextual_anomalies(df, domain=domain):
            _append_issue(report, issue)

    # ── Leakage ───────────────────────────────────────────────────────────────
    if run_leakage and target is not None:
        from tsauditor.leakage import (
            audit_equivalence,
            audit_correlation_leakage,
            audit_temporal_leakage,
        )
        for issue in audit_equivalence(df, target=target, domain=domain):
            _append_issue(report, issue)

        for issue in audit_correlation_leakage(df, target=target, domain=domain):
            _append_issue(report, issue)

        for issue in audit_temporal_leakage(df, target=target, domain=domain):
            _append_issue(report, issue)

    return report


def _append_issue(report: GuardReport, issue: Issue) -> None:
    """Route an Issue to the correct severity bucket in the report."""
    if issue.severity == CRITICAL:
        report.critical.append(issue)
    elif issue.severity == WARNING:
        report.warnings.append(issue)
    else:
        report.info.append(issue)
