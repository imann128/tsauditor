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
"""

from tsauditor.scanner import scan
from tsauditor.report.summary import GuardReport, Issue

__version__ = "0.1.0"
__all__ = ["scan", "GuardReport", "Issue"]
