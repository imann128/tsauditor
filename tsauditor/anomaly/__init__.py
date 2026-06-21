"""
tsauditor.anomaly
-----------------
Time-series aware anomaly detection.

Modules
-------
point        Statistical outliers (z-score, IQR).
contextual   Stuck values, implausible jumps relative to neighbors.
classifier   Routes anomalies to probable cause categories.
"""

from tsauditor.anomaly.point import audit_point_anomalies
from tsauditor.anomaly.contextual import audit_contextual_anomalies

__all__ = ["audit_point_anomalies", "audit_contextual_anomalies"]
