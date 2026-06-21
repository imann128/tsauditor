"""
tsauditor.leakage
-----------------
Temporal leakage detection for time-series feature pipelines.

Modules
-------
correlation  Cross-correlation across lags to detect future information in features.
equivalence  Target equivalence detection (feature mathematically identical to target).
temporal     Rolling window lookahead detection.

Issue codes raised
------------------
LEK001  Target equivalence: feature is mathematically identical or near-identical
        to the target (lag-0 correlation >= threshold).
LEK002  Positive-lag peak: feature's peak correlation with target occurs at a
        positive lag, indicating future information leakage.
LEK003  Rolling window lookahead: suspected forward-looking window in feature
        construction.
"""

from tsauditor.leakage.correlation import audit_correlation_leakage
from tsauditor.leakage.equivalence import audit_equivalence
from tsauditor.leakage.temporal import audit_temporal_leakage

__all__ = [
    "audit_correlation_leakage",
    "audit_equivalence",
    "audit_temporal_leakage",
]
