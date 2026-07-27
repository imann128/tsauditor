"""
tsauditor.leakage.combination
------------------------------
Combination leakage: no single feature reproduces the target, but a *group* of
features together does.

Every other leakage check in tsauditor is univariate — each feature is scored
against the target on its own. That misses a whole class of real bug::

    target = (high + low) / 2
    target = revenue - costs
    target = price * quantity
    target = numerator / denominator
    target = a + b + c

Here no single input is near-deterministic, so LEK001 stays silent, but the
group reconstructs the target exactly. The canonical shape is a target defined
as a *difference*: with ``x1`` and ``x2`` independent and ``target = x1 - x2``,
each correlates with the target at only ~0.7 — far below LEK001's 0.95 — while
the pair explains it perfectly.

Detection method
----------------
For a candidate group of columns, fit ``target ~ 1 + columns`` by ordinary least
squares and take the **adjusted** R². Two algebraic forms are tried:

- **linear** — catches sums, differences and weighted combinations
- **log** — the same fit on ``log`` of the target and columns, which catches
  *products and ratios*, since ``log(a*b) = log a + log b`` and
  ``log(a/b) = log a - log b``. Only attempted when the target and both columns
  are strictly positive.

Measured coverage (adjusted R², n=500):

===========================  ========  =========
target                       linear    log
===========================  ========  =========
``x1 - x2``                  1.0000    0.0112
``x1 * x2``                  0.9287    1.0000
``x1 / x2``                  0.8304    1.0000
unrelated control            -0.0026   -0.0038
===========================  ========  =========

Neither form alone is sufficient; together they cover the four shapes that
account for almost all real combination leakage. An interaction term
(``x_i * x_j`` as a third predictor) was tested as an alternative and rejected:
it catches products but not ratios, and roughly doubles the chance-level R².

Adjusted (not raw) R² is used throughout because it penalises extra predictors,
which keeps the null distribution tight across many candidate groups.

Triples, without the cost of O(k^3)
-----------------------------------
Scanning every triple would be C(k,3) fits — 161,700 for 100 features — and
would badly inflate the multiple-comparison problem.

Instead, triples are reached by **residual extension**: if ``target = a+b+c``,
then any *pair* drawn from those three already explains a large share of the
target (0.71 measured for equal contributions), even though it falls short of
the flagging threshold. So only pairs scoring at least ``triple_gate`` are
extended with a third column.

On random data **no pair clears the gate at all**, so triples cost nothing and
contribute no false positives. The gate was verified not to block genuine
three-way identities across equal, very unequal, cancelling and collinear
component shapes (best pair 0.71-1.00 in every case).

The single-feature guard
------------------------
Without it, one leaky column poisons the whole report: if ``leak`` alone
reproduces the target, then *every* group containing ``leak`` also reaches R²
1.0, producing k-1 findings for a leak LEK001 already reported once. So a group
is skipped when any of its columns *alone* reaches the threshold. That case
belongs to LEK001; this check is only for leakage that emerges from combination.

False-positive profile
----------------------
Measured on random targets with independent random features, the largest
adjusted R² reached by chance was 0.075 for pairs (50 features, 1225 pairs, 100
rows) and typically below 0.03; the log form behaves the same (max 0.028). No
triple was ever evaluated on random data because no pair cleared the gate.
Innocent but highly correlated feature pairs (r ~ 0.96) score ~0.00 against an
unrelated target.

Issue codes raised
------------------
LEK005  Combination leakage: a group of features reconstructs the target. CRITICAL.
"""

from __future__ import annotations

import itertools
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from tsauditor.report.summary import Issue, CRITICAL

# Values at or below this are treated as non-positive for the log form. Using a
# small positive floor rather than 0 avoids log() blowing up on values that are
# positive only by floating-point accident.
_POSITIVE_FLOOR = 1e-12


def _adjusted_r2(y: np.ndarray, X: np.ndarray) -> float:
    """
    Adjusted R² of ``y ~ 1 + X`` by least squares.

    ``lstsq`` is used rather than a normal-equation solve because candidate
    groups are frequently collinear (``high``/``low``, a level and its lag),
    which makes ``X'X`` singular; ``lstsq`` handles that via the pseudo-inverse
    instead of raising.
    """
    n = len(y)
    p = X.shape[1]
    if n <= p + 1:
        return 0.0

    design = np.column_stack([np.ones(n), X])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ beta

    ss_tot = float(((y - y.mean()) ** 2).sum())
    if ss_tot <= 0:
        return 0.0

    r2 = 1.0 - float(residual @ residual) / ss_tot
    return 1.0 - (1.0 - r2) * (n - 1) / (n - p - 1)


def _score_arrays(y: np.ndarray, X: np.ndarray) -> Tuple[float, str]:
    """
    Best adjusted R² for ``y ~ X``, over the linear and log forms.

    Returns ``(score, form)`` where form is ``"linear"`` or ``"log"``.

    The log form fits ``log|y| ~ log|X|``. Absolute values rather than raw ones,
    so that products and ratios of *signed* data are still recovered:
    ``|a*b| = |a|*|b|`` holds regardless of sign. On signed inputs the linear
    form scores 0.009 for ``a*b`` — completely blind — while the absolute-log
    form scores 1.000. It is skipped when any value sits at or below
    ``_POSITIVE_FLOOR`` in magnitude, since ``log`` of a near-zero would produce
    a huge negative that dominates the fit.
    """
    best = _adjusted_r2(y, X)
    form = "linear"

    abs_y = np.abs(y)
    abs_X = np.abs(X)
    if bool((abs_y > _POSITIVE_FLOOR).all() and (abs_X > _POSITIVE_FLOOR).all()):
        log_score = _adjusted_r2(np.log(abs_y), np.log(abs_X))
        if log_score > best:
            best, form = log_score, "log"

    return best, form


class _Matrix:
    """
    Column-major view of the numeric frame with a precomputed NaN mask.

    Building a ``pd.concat`` per candidate group was the dominant cost — over a
    second for 50 features. Extracting arrays once and slicing with a boolean
    mask brings the same scan down to well under a tenth of that.
    """

    __slots__ = ("y", "columns", "y_ok", "col_ok")

    def __init__(self, y: pd.Series, numeric: pd.DataFrame, features: Sequence[str]):
        self.y = y.to_numpy(dtype=float)
        self.y_ok = ~np.isnan(self.y)
        self.columns = {c: numeric[c].to_numpy(dtype=float) for c in features}
        self.col_ok = {c: ~np.isnan(v) for c, v in self.columns.items()}

    def block(self, names: Sequence[str]):
        """Complete-case ``(y, X)`` for these columns, or ``(None, None)``."""
        mask = self.y_ok
        for name in names:
            mask = mask & self.col_ok[name]
        if not mask.any():
            return None, None
        X = np.column_stack([self.columns[n][mask] for n in names])
        return self.y[mask], X


def audit_combination_leakage(
    df: pd.DataFrame,
    target: str,
    threshold: float = 0.95,
    min_obs: int = 30,
    max_features: Optional[int] = None,
    max_reported: int = 10,
    max_group_size: int = 3,
    gate: float = 0.30,
    max_candidates_per_level: int = 200,
    domain: Optional[str] = None,
) -> List[Issue]:
    """
    Detect groups of features that jointly reconstruct the target (LEK005).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    target : str
        Name of the target column. Must exist in ``df``.
    threshold : float
        Adjusted R² at or above which a group is flagged. Default 0.95, matching
        LEK001's near-determinism threshold.
    min_obs : int
        Minimum complete rows required to score a group. Default 30.
    max_features : Optional[int]
        Cap on how many numeric features to consider. ``None`` (default) means no
        cap. The pair scan is O(k²) — roughly 0.2s for 100 features — so set this
        if you have several hundred columns.
    max_reported : int
        Maximum number of findings, best first. Default 10. Prevents a family of
        derived columns producing dozens of near-identical results.
    max_group_size : int
        Largest group to search. Default 3 (pairs and triples). Set 2 for pairs
        only, or 4+ to find larger identities — each extra level is free on clean
        data but can cost time on frames where many features partially explain
        the target. See the module docstring for measured costs.
    gate : float
        A group must reach this adjusted R² before it is extended by one more
        column. Default 0.30. On random data nothing clears it, so deeper levels
        add no false positives; genuine identities produce sub-groups well above
        it (0.71 for a pair inside a 3-way, 0.49 inside a 4-way).
    max_candidates_per_level : int
        Cap on how many sub-threshold groups are carried forward to the next
        level, best first. Default 200. Without a cap, a frame of 40 mutually
        correlated features took 21s at ``max_group_size=4``; with it, 0.7s.
    domain : Optional[str]
        Accepted for API consistency; has no effect.

    Returns
    -------
    List[Issue]
        Zero or more LEK005 Issues (CRITICAL), strongest group first. A triple is
        not reported when one of its own pairs was already reported.
    """
    issues: List[Issue] = []

    if target not in df.columns:
        raise ValueError(f"target '{target}' not found in DataFrame columns.")

    numeric = df.select_dtypes(include=["number"]).replace([np.inf, -np.inf], np.nan)
    if target not in numeric.columns:
        # A binary categorical target is encodable, but reconstructing a value
        # arithmetically is a numeric question; skip rather than guess.
        return issues

    features = [c for c in numeric.columns if c != target]
    if max_features is not None:
        features = features[:max_features]
    if len(features) < 2:
        return issues

    y_full = numeric[target]
    if y_full.dropna().nunique() < 2:
        return issues

    matrix = _Matrix(y_full, numeric, features)

    # Single-column explanatory power, computed once. Used to skip groups whose
    # leakage is already attributable to one column (LEK001's job).
    single: Dict[str, Optional[float]] = {}
    for col in features:
        y_vals, X_vals = matrix.block([col])
        if y_vals is None or len(y_vals) < min_obs or len(np.unique(X_vals)) < 2:
            single[col] = None
            continue
        single[col] = _score_arrays(y_vals, X_vals)[0]

    # Columns that already explain the target alone belong to LEK001, and any
    # group containing one would trivially score high.
    usable = [c for c in features if single[c] is not None and single[c] < threshold]
    found: List[dict] = []
    reported: List[frozenset] = []

    # ── Iterative deepening ───────────────────────────────────────────────────
    # Level 2 is every pair. Each subsequent level extends the surviving groups
    # from the level below by one column. A group survives when it reaches
    # `gate` without reaching `threshold` — i.e. it explains a real share of the
    # target but is not yet an identity, which is exactly the signature of a
    # sub-group of a larger one.
    candidates: List[Tuple[float, frozenset]] = [
        (0.0, frozenset(pair)) for pair in itertools.combinations(usable, 2)
    ]

    for size in range(2, max(max_group_size, 2) + 1):
        survivors: List[Tuple[float, frozenset]] = []
        seen: set = set()

        for _, key in candidates:
            if key in seen:
                continue
            seen.add(key)

            # A superset of something already reported is the same finding with
            # a redundant column bolted on.
            if any(prior <= key for prior in reported):
                continue

            columns = sorted(key)
            y_vals, X_vals = matrix.block(columns)
            if y_vals is None or len(y_vals) < min_obs:
                continue
            if any(len(np.unique(X_vals[:, k])) < 2 for k in range(X_vals.shape[1])):
                continue

            score, form = _score_arrays(y_vals, X_vals)
            if score >= threshold:
                found.append(
                    {
                        "score": score,
                        "form": form,
                        "columns": columns,
                        "n_obs": len(y_vals),
                    }
                )
                reported.append(key)
            elif score >= gate:
                survivors.append((score, key))

        if size >= max_group_size or not survivors:
            break

        # Carry forward only the strongest sub-groups. Unbounded expansion is
        # what turns a correlated frame into a 21-second scan.
        survivors.sort(key=lambda item: -item[0])
        survivors = survivors[:max_candidates_per_level]

        candidates = [
            (score, key | {col})
            for score, key in survivors
            for col in usable
            if col not in key
        ]

    found.sort(key=lambda item: (-item["score"], len(item["columns"])))

    for item in found[:max_reported]:
        columns = item["columns"]
        best_single = max(single[c] for c in columns)
        joined = ", ".join(f"'{c}'" for c in columns)
        relation = (
            "an additive combination (a sum, difference or weighted mix)"
            if item["form"] == "linear"
            else "a multiplicative combination (a product or ratio)"
        )

        issues.append(
            Issue(
                module="leakage",
                code="LEK005",
                severity=CRITICAL,
                description=(
                    f"Features {joined} together reconstruct target '{target}' "
                    f"(adjusted R²={item['score']:.4f} >= {threshold}, "
                    f"{item['form']} form), while none does alone (best single "
                    f"adjusted R²={best_single:.4f}). This is combination "
                    f"leakage — the target is likely {relation} of these columns. "
                    f"Review how it was constructed."
                ),
                column=columns[0],
                evidence={
                    "metric": "adjusted_r2",
                    "form": item["form"],
                    "group": columns,
                    "group_size": len(columns),
                    "group_adjusted_r2": round(float(item["score"]), 4),
                    "best_single_adjusted_r2": round(float(best_single), 4),
                    "threshold": threshold,
                    "n_obs": int(item["n_obs"]),
                },
            )
        )

    return issues
