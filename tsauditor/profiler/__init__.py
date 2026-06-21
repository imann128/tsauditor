"""
tsauditor.profiler
------------------
Structural time-series profiling.

Modules
-------
frequency   Timestamp regularity, gap detection, duplicate timestamps.
stationarity ADF/KPSS stationarity tests per numeric column.
missing     Temporal missing value pattern analysis (clustered vs random).
"""

from tsauditor.profiler.frequency import audit_frequency
from tsauditor.profiler.stationarity import audit_stationarity
from tsauditor.profiler.missing import audit_missing

__all__ = ["audit_frequency", "audit_stationarity", "audit_missing"]
