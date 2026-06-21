"""
tsauditor.leakage.equivalence
------------------------------
Detects features that are mathematically equivalent or near-equivalent
to the target variable at lag 0.

This is the canonical OGDC ChangeP case: a feature derived from the
same-day closing price being used to predict a target also derived from
the same-day closing price. The two columns may have different names and
come from different sources, yet be functionally identical.

Detection method (rank-based)
-----------------------------
Equivalence to a target is a *determinism* question — "does this feature
near-perfectly reproduce the target?" — not a linearity question. Linear
(Pearson) correlation answers the wrong question and, worse, collapses on
the exact case this module exists for: against a binary 0/1 target the
Pearson point-biserial correlation has a hard ceiling of sqrt(2/pi) ~ 0.798,
so a feature whose *sign defines* the target (Direction = 1{ChangeP > 0})
scores only ~0.80 and routinely slips under a 0.80 cutoff.

So we use rank-based metrics, chosen by target type:

    Continuous target  ->  |Spearman rho|              flag if >= 0.95
    Binary target      ->  AUC separation              flag if >= 0.95
                           ( max(AUC, 1 - AUC), where AUC is the
                             Mann-Whitney rank statistic )

Spearman catches any *monotonic* equivalence (including non-linear ones a
log or square transform would hide from Pearson) and is robust to outliers.
AUC scores
1.0 for a feature that perfectly separates the two classes — exactly the
sign-derived leakage above — while a legitimate weak predictor sits near
0.5. Both metrics live on a comparable [0, 1] scale, so a single 0.95
"near-equivalence" threshold is meaningful for either target type.

Issue codes raised
------------------
LEK001  Target equivalence detected.  (CRITICAL)
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from tsauditor.report.summary import Issue, CRITICAL


def _auc(feature: pd.Series, y01: np.ndarray) -> Optional[float]:
    """
    Area under the ROC curve via the Mann-Whitney rank statistic.

    Uses average ranks so ties are handled correctly. Returns None if one
    of the two classes is absent (AUC undefined). Equivalent to the
    probability that a randomly chosen class-1 point ranks above a randomly
    chosen class-0 point.
    """
    n1 = float(y01.sum())
    n0 = float(len(y01) - n1)
    if n1 == 0 or n0 == 0:
        return None
    ranks = feature.rank()  # average ranks for ties
    rank_sum_pos = ranks.to_numpy()[y01 == 1].sum()
    return (rank_sum_pos - n1 * (n1 + 1) / 2) / (n1 * n0)


def audit_equivalence(
    df: pd.DataFrame,
    target: str,
    continuous_threshold: float = 0.95,
    binary_threshold: float = 0.95,
    min_obs: int = 30,
    domain: Optional[str] = None,
) -> List[Issue]:
    """
    Detect features that near-deterministically reproduce the target (lag 0).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    target : str
        Name of the target column. Must exist in ``df``.
    continuous_threshold : float
        Absolute Spearman correlation threshold for a continuous target.
        Default 0.95.
    binary_threshold : float
        AUC-separation threshold for a binary target, applied to
        ``max(AUC, 1 - AUC)``. Default 0.95. Loosen toward 0.90 to tolerate
        a leak that carries some label noise — still far above any
        legitimate predictor (~0.5-0.65).
    min_obs : int
        Minimum number of pairwise-complete (feature, target) observations
        required to trust a score. Below this the column is skipped, because
        a high score from a handful of points is spurious. Default 30.
    domain : Optional[str]
        Accepted for API consistency; thresholds are driven by target type,
        not domain.

    Returns
    -------
    List[Issue]
        One LEK001 Issue per flagged feature column.
    """
    issues: List[Issue] = []

    # 1. Validate the target exists.
    if target not in df.columns:
        raise ValueError(f"target '{target}' not found in DataFrame columns.")

    target_raw = df[target]
    n_unique = target_raw.dropna().nunique()

    # A constant (or all-null) target has no variance: equivalence is
    # undefined and there is nothing to reproduce. Skip cleanly.
    if n_unique < 2:
        return issues

    # 2. Determine target type and pick the metric + threshold.
    is_binary = n_unique == 2
    if is_binary:
        # Encode the two categories to 0/1 deterministically so the method
        # works for numeric (0/1) and categorical ("up"/"down") binaries alike.
        categories = sorted(target_raw.dropna().unique(), key=str)
        mapping = {categories[0]: 0.0, categories[1]: 1.0}
        y = target_raw.map(mapping)
        threshold = binary_threshold
        target_type = "binary"
    else:
        if not pd.api.types.is_numeric_dtype(target_raw):
            raise ValueError(
                f"continuous target '{target}' must be numeric to correlate."
            )
        y = target_raw.astype(float)
        threshold = continuous_threshold
        target_type = "continuous"

    # 3. Score each numeric feature (excluding the target) against the target.
    for col in df.select_dtypes(include=["number"]).columns:
        if col == target:
            continue

        # Pairwise-complete observations only; treat inf as missing.
        pair = (
            pd.concat([df[col], y], axis=1, keys=["x", "y"])
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        if len(pair) < min_obs:
            continue

        # A zero-variance feature cannot reproduce anything; its score is
        # undefined (constant ranks). Skip.
        if pair["x"].nunique() < 2:
            continue

        if target_type == "binary":
            auc = _auc(pair["x"], pair["y"].to_numpy())
            if auc is None:
                continue  # only one class present here
            score = max(auc, 1.0 - auc)  # direction-agnostic separation
            evidence = {
                "metric": "auc",
                "auc": round(float(auc), 4),
                "separation": round(float(score), 4),
                "threshold": threshold,
                "target_type": target_type,
                "n_obs": int(len(pair)),
            }
        else:
            rho = pair["x"].corr(pair["y"], method="spearman")
            if pd.isna(rho):
                continue
            score = abs(float(rho))
            evidence = {
                "metric": "spearman",
                "spearman_rho": round(float(rho), 4),
                "threshold": threshold,
                "target_type": target_type,
                "n_obs": int(len(pair)),
            }

        if score >= threshold:
            issues.append(
                Issue(
                    module="leakage",
                    code="LEK001",
                    severity=CRITICAL,
                    description=(
                        f"Feature '{col}' near-deterministically reproduces target "
                        f"'{target}' ({evidence['metric']} score={score:.4f} >= "
                        f"{threshold} for {target_type} target). Likely data "
                        f"leakage — review before modeling."
                    ),
                    column=col,
                    evidence=evidence,
                )
            )

    return issues
