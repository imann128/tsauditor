"""
tsauditor
---------
A data quality auditing library for time-series tabular data
in financial and sensor domains.

Basic usage::

    import tsauditor as tsa

    report = tsa.scan(df, target="Direction", domain="finance")
    report.summary()
    issues = report.critical
    report.to_json("report.json")

    # one-shot scan-and-repair, keeping the audit trail:
    clean, report = tsa.fix(df, target="Direction", domain="finance")

    # format a series for Google TimesFM inference:
    array = tsa.adapters.to_timesfm(df, target_col="close_price")
"""

from tsauditor.scanner import scan
from tsauditor.remediate import fix
from tsauditor.report.summary import GuardReport, Issue
from tsauditor import adapters

__version__ = "0.3.0"
__all__ = ["scan", "fix", "adapters", "GuardReport", "Issue"]
