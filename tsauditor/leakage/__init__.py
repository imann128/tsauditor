"""
tsauditor.leakage
-----------------
Temporal leakage detection for time-series feature pipelines.

Modules
-------
correlation  Cross-correlation across lags to detect future information in features.
equivalence  Target equivalence detection (feature mathematically identical to target).
temporal     Rolling window lookahead detection.
asof         Point-in-time availability leakage (values used before release).
combination  Group-of-features leakage (no single feature leaks, but a small
             combination reconstructs the target).

Issue codes raised
------------------
LEK001  Target equivalence: feature is mathematically identical or near-identical
        to the target (lag-0 correlation >= threshold).
LEK002  Positive-lag peak: feature's peak correlation with target occurs at a
        positive lag, indicating future information leakage.
LEK003  Rolling window lookahead: suspected forward-looking window in feature
        construction.
LEK004  As-of leakage: a value is used before it was available (its release
        timestamp is later than the row it occupies). Opt-in; needs availability
        metadata.
LEK005  Combination leakage: a small group of features jointly reconstructs the
        target even though no individual feature does.
"""

from tsauditor.leakage.correlation import audit_correlation_leakage
from tsauditor.leakage.equivalence import audit_equivalence
from tsauditor.leakage.temporal import audit_temporal_leakage
from tsauditor.leakage.asof import audit_asof_leakage
from tsauditor.leakage.combination import audit_combination_leakage

__all__ = [
    "audit_correlation_leakage",
    "audit_equivalence",
    "audit_temporal_leakage",
    "audit_asof_leakage",
    "audit_combination_leakage",
]
