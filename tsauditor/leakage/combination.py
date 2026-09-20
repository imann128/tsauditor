"""
tsauditor.leakage.combination
------------------------------
Combination leakage: no single feature reproduces the target, but a group of
features together does.

Every other leakage check in tsauditor is univariate: each feature is scored
against the target on its own. That misses a whole class of real bug::

    target = (high + low) / 2
    target = revenue - costs
    target = price * quantity
    target = numerator / denominator
    target = a + b + c

Here no single input is near-deterministic, so LEK001 stays silent, but the
group reconstructs the target exactly. The canonical shape is a target
defined as a difference: with ``x1`` and ``x2`` independent and
``target = x1 - x2``, each correlates with the target at only ~0.7 (far
below LEK001's 0.95) while the pair explains it perfectly.

Detection method
----------------
For a candidate group of columns, fit ``target ~ 1 + columns`` by ordinary
least squares and take the adjusted R². Two algebraic forms are tried:

- **linear**: catches sums, differences and weighted combinations
- **log**: the same fit on ``log`` of the target and columns, which catches
  products and ratios, since ``log(a*b) = log a + log b`` and
  ``log(a/b) = log a - log b``. Only attempted when the target and both
  columns are strictly positive.

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
(``x_i * x_j`` as a third predictor) was tested and rejected: it catches
products but not ratios, and roughly doubles the chance-level R².

Adjusted (not raw) R² is used throughout because it penalises extra
predictors, which keeps the null distribution tight across many candidate
groups.

Triples, without the cost of O(k^3)
-----------------------------------
Scanning every triple would be C(k,3) fits (161,700 for 100 features) and
would badly inflate the multiple-comparison problem.

Instead, triples are reached by residual extension: if ``target = a+b+c``,
any pair drawn from those three already explains a large share of the
target (0.71 measured for equal contributions), even though it falls short
of the flagging threshold. So only pairs scoring at least ``triple_gate``
are extended with a third column.

On random data no pair clears the gate at all, so triples cost nothing and
contribute no false positives. Verified not to block genuine three-way
identities across equal, very unequal, cancelling and collinear component
shapes (best pair 0.71-1.00 in every case).

The single-feature guard
------------------------
Without it, one leaky column poisons the whole report: if ``leak`` alone
reproduces the target, every group containing ``leak`` also reaches R²
1.0, producing k-1 findings for a leak LEK001 already reported once. So a
group is skipped when any of its columns alone reaches the threshold. That
case belongs to LEK001; this check is only for leakage that emerges from
combination.

False-positive profile
----------------------
Measured on random targets with independent random features, the largest
adjusted R² reached by chance was 0.075 for pairs (50 features, 1225 pairs,
100 rows) and typically below 0.03; the log form behaves the same (max
0.028). No triple was ever evaluated on random data because no pair cleared
the gate. Innocent but highly correlated feature pairs (r ~ 0.96) score
~0.00 against an unrelated target.

Issue codes raised
------------------
LEK005  Combination leakage: a group of features reconstructs the target. CRITICAL.
"""

from __future__ import annotations

import itertools
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from tsauditor.leakage.equivalence import _auc as _pairwise_auc
from tsauditor.leakage.equivalence import _score_feature as _equivalence_score
from tsauditor.report.summary import Issue, CRITICAL

# Values at or below this are treated as non-positive for the log form. A
# small positive floor rather than 0 avoids log() blowing up on values that
# are positive only by floating-point accident.
_POSITIVE_FLOOR = 1e-12


def _adjusted_r2(y: np.ndarray, X: np.ndarray) -> float:
    """
    Adjusted R² of ``y ~ 1 + X`` by least squares.

    ``lstsq`` is used rather than a normal-equation solve because candidate
    groups are frequently collinear (``high``/``low``, a level and its lag),
    which makes ``X'X`` singular; ``lstsq`` handles that via the
    pseudo-inverse instead of raising.
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

    The log form fits ``log|y| ~ log|X|``. Absolute values, not raw ones, so
    products and ratios of signed data are still recovered: ``|a*b| =
    |a|*|b|`` holds regardless of sign. On signed inputs the linear form
    scores 0.009 for ``a*b`` (blind) while the absolute-log form scores
    1.000. Skipped when any value sits at or below ``_POSITIVE_FLOOR`` in
    magnitude, since ``log`` of a near-zero would dominate the fit.

    This is the group's R²/log score only. For a binary target it is
    ceiling-limited (see ``_binary_combination_auc`` below) and is
    supplemented, not replaced, by that separate check in
    ``audit_combination_leakage``.
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


# Fold count and permutation count for _binary_combination_auc. Fixed
# private constants rather than public parameters: they trade off runtime
# against resolution/robustness in ways a caller has no principled basis to
# tune per-dataset (unlike max_group_size, which trades off runtime against
# what shapes of leak can be found at all). See _binary_combination_auc's
# docstring for how they were chosen.
_CV_FOLDS = 5
_PERM_ITERATIONS = 200
_PERM_ALPHA = 0.01


def _kfold_fitted(y: np.ndarray, X: np.ndarray, k: int, seed: int) -> np.ndarray:
    """
    K-fold cross-validated fitted values of ``y ~ 1 + X``: row ``i``'s value
    is predicted by a model fit only on the folds not containing row ``i``.

    Deliberately literal refitting per fold, not the closed-form
    leave-one-out (LOOCV) shortcut used elsewhere in statistics for OLS
    (``fitted[i] = y[i] - residual[i]/(1-h[i][i])``). That shortcut is exact
    for full LOOCV but was rejected here: it is explicitly linear in point
    ``i``'s own label ``y[i]``, so when leverage ``h`` is small and roughly
    uniform (2-4 predictors over dozens-to-thousands of rows, the ordinary
    case), leave-one-out "predictions" become a near-deterministic,
    monotonic function of each row's own label even when X carries zero
    information about y, and a rank-based statistic (AUC) reads that
    monotonicity as apparent near-perfect separation. Verified directly:
    two independent random features against an independent random binary
    target, n=200. The closed-form LOOCV-fitted values correlated with y
    itself at r=-0.27 (should be ~0 if X carries nothing), and over
    permutations of the same data that artifact alone produced a
    fitted-value AUC of 1.00 on 1 of 200 draws, the fingerprint of the
    formula leaking each point's own label into its own "held out"
    prediction rather than a rare tail event.

    K-fold with actual refitting does not have this property: row ``i``'s
    prediction depends on the fold's excluded rows as a group, not
    algebraically on row ``i``'s own label. Swept the same way (independent
    features vs independent binary target, k=5): mean |corr(cv_fitted, y)|
    dropped from the LOOCV shortcut's 0.27 to under 0.05.

    ``k=5`` (``_CV_FOLDS``) is a moderate choice: fewer folds dilutes the
    recentering artifact further but costs estimation precision from
    smaller training folds; more folds approaches the biased LOOCV
    shortcut's own behavior as k -> n. 5 was measured to keep the residual
    bias small across n in {30, 50, 100, 200} while still resolving a
    genuine signal cleanly.

    A degenerate fold (fewer training rows than the design needs, i.e.
    ``n_train <= p + 1``) falls back to the training fold's own mean rather
    than raising. This can only happen with ``min_obs`` close to its floor
    and a large group size, and a constant fallback prediction cannot itself
    manufacture separation.
    """
    n = len(y)
    p = X.shape[1]
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    folds = np.array_split(order, k)
    fitted = np.empty(n, dtype=float)

    for fold in folds:
        train = np.setdiff1d(order, fold, assume_unique=False)
        if len(train) <= p + 1:
            fitted[fold] = y[train].mean() if len(train) else float(y.mean())
            continue
        design_train = np.column_stack([np.ones(len(train)), X[train]])
        beta, *_ = np.linalg.lstsq(design_train, y[train], rcond=None)
        design_test = np.column_stack([np.ones(len(fold)), X[fold]])
        fitted[fold] = design_test @ beta

    return fitted


def _fitted_auc_score(y01: np.ndarray, fitted: np.ndarray) -> Optional[float]:
    """Direction-agnostic AUC separation of a continuous score against a
    binary target, via ``equivalence._auc`` (the same rank statistic LEK001
    uses). ``None`` if one class is absent on this sample."""
    auc = _pairwise_auc(pd.Series(fitted), y01)
    if auc is None:
        return None
    return max(auc, 1.0 - auc)


def _binary_combination_auc(
    y: np.ndarray, X: np.ndarray, y01: np.ndarray, threshold: float, seed: int = 0
) -> Optional[Tuple[float, str]]:
    """
    Test whether a group's fitted linear combination separates a binary
    target far more than chance, closing a ceiling plain R² cannot reach for
    this target type.

    Why R² alone is not enough: adjusted R² of an OLS fit against a binary
    target has the same hard ceiling documented in ``leakage.equivalence``'s
    module docstring. A point-biserial-type correlation between a
    continuous score and a binary variable cannot exceed sqrt(2/pi) ~=
    0.798, i.e. R² cannot exceed ~0.637, however perfectly the group
    actually determines the class. A group of features that jointly
    reconstruct a binary target via a threshold rule on their linear
    combination (e.g. ``target = 1{a - b > 0}``) is exactly this shape, and
    would never cross a threshold above that ceiling under R² alone, the
    same failure LEK001 (``audit_equivalence``) already had to fix for
    single columns by switching to AUC separation. This mirrors that fix,
    applied to a group's fitted combination rather than a single raw column.

    Two safeguards beyond a plain AUC-of-fit, both necessary (see
    ``_kfold_fitted`` for the first in detail):

    1. The continuous score is a K-fold cross-validated fit
       (``_kfold_fitted``), not the in-sample fit. An in-sample OLS fit,
       scored by AUC, is biased well above 0.5 by chance alone with a
       handful of unrelated predictors at this module's ``min_obs`` floor
       (measured mean 0.63, max 0.77 over 50 trials of two independent
       random features against an independent random binary target at
       n=30), enough to trigger a flag directly by chance and clear
       ``gate=0.30`` on essentially every random pair.

    2. Even with cross-validation, the observed AUC-of-fit is compared
       against its own null distribution by permutation
       (``_PERM_ITERATIONS`` shuffles of which row each label belongs to,
       refitting and rescoring identically each time) rather than against a
       fixed number. AUC has substantially higher small-sample sampling
       variance than adjusted R²: at n=30 the standard error of an AUC
       estimate under the null is large enough that ``max(AUC, 1-AUC)``
       alone routinely exceeds 0.30 (and occasionally exceeds 0.95) purely
       from estimation noise, so no fixed cutoff is well-calibrated the way
       it is for adjusted R² (whose null distribution concentrates tightly
       near 0). The permutation reference distribution is exact for this
       group's own sample size and feature geometry, so it stays valid
       regardless of any residual bias left over from (1); flagging
       requires the observed score to clear ``threshold`` and land in the
       most extreme ``_PERM_ALPHA`` (default 1%) of that distribution.

    Only called from ``audit_combination_leakage`` for a binary target, and
    only on a candidate that has already cleared ``gate`` under plain R²,
    which keeps the moderately expensive permutation test off most
    candidates. A genuine binary combination clears ``gate`` easily under
    plain R² alone (adjusted R² ~0.60-0.65 for a threshold-rule
    combination), so this does not weaken recall; an independent random
    pair essentially never reaches ``gate`` under R² (max 0.075 by chance),
    so this check almost never runs on noise.

    Returns ``(score, "linear-auc")`` if the observed AUC separation clears
    both ``threshold`` and permutation significance, else ``None``.

    Verified by direct simulation: a two-feature exact reconstruction of a
    balanced binary target (``target = 1{a - b > 0}``) was flagged in 30/30
    trials with this check (0/30 under R² alone, since adjusted R² tops out
    near 0.63-0.65 and never reaches the default 0.95 threshold); a
    false-positive sweep of 30 trials of 10 mutually independent random
    features against an independent random binary target at n=30, testing
    all 45 candidate pairs through the real gate/threshold/permutation
    pipeline, produced 0 flags.
    """
    fitted = _kfold_fitted(y, X, k=_CV_FOLDS, seed=seed)
    observed = _fitted_auc_score(y01, fitted)
    if observed is None or observed < threshold:
        return None

    rng = np.random.default_rng(seed + 1)
    n = len(y)
    at_least_as_extreme = 0
    for i in range(_PERM_ITERATIONS):
        perm = rng.permutation(n)
        y_perm, y01_perm = y[perm], y01[perm]
        perm_fitted = _kfold_fitted(y_perm, X, k=_CV_FOLDS, seed=seed + 2 + i)
        perm_score = _fitted_auc_score(y01_perm, perm_fitted)
        if perm_score is not None and perm_score >= observed:
            at_least_as_extreme += 1

    p_value = (at_least_as_extreme + 1) / (_PERM_ITERATIONS + 1)
    if p_value < _PERM_ALPHA:
        return observed, "linear-auc"
    return None


class _Matrix:
    """
    Column-major view of the numeric frame with a precomputed NaN mask.

    Building a ``pd.concat`` per candidate group was the dominant cost:
    over a second for 50 features. Extracting arrays once and slicing with a
    boolean mask brings the same scan down to well under a tenth of that.
    """

    __slots__ = ("y", "columns", "y_ok", "col_ok", "y01")

    def __init__(
        self,
        y: pd.Series,
        numeric: pd.DataFrame,
        features: Sequence[str],
        y01: Optional[pd.Series] = None,
    ):
        self.y = y.to_numpy(dtype=float)
        self.y_ok = ~np.isnan(self.y)
        self.columns = {c: numeric[c].to_numpy(dtype=float) for c in features}
        self.col_ok = {c: ~np.isnan(v) for c, v in self.columns.items()}
        # y01 (the target re-encoded to {0.0, 1.0}) is only set for a binary
        # target (see audit_combination_leakage). It shares y's index and
        # therefore y's NaN pattern exactly, so masking it with
        # y_ok/col_ok below is valid.
        self.y01 = y01.to_numpy(dtype=float) if y01 is not None else None

    def block(self, names: Sequence[str]):
        """Complete-case ``(y, X, y01)`` for these columns; ``y01`` is None
        unless this matrix was built with one, and ``(None, None, None)`` if
        no rows survive the mask."""
        mask = self.y_ok
        for name in names:
            mask = mask & self.col_ok[name]
        if not mask.any():
            return None, None, None
        X = np.column_stack([self.columns[n][mask] for n in names])
        y01_masked = self.y01[mask] if self.y01 is not None else None
        return self.y[mask], X, y01_masked


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
    seed: int = 0,
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
        Adjusted R² at or above which a group is flagged. Default 0.95,
        matching LEK001's near-determinism threshold. For a binary target,
        a group that reaches ``gate`` under R² but not ``threshold`` is
        additionally tested via cross-validated AUC separation of its
        fitted combination (``_binary_combination_auc``), against this same
        ``threshold``. Adjusted R² alone cannot exceed ~0.64 against a
        binary target (the point-biserial ceiling; see
        ``leakage.equivalence``'s module docstring), so without this a
        genuine binary combination leak can never be flagged.
    min_obs : int
        Minimum complete rows required to score a group. Default 30.
    max_features : Optional[int]
        Cap on how many numeric features to consider. ``None`` (default)
        means no cap. The pair scan is O(k²) (roughly 0.2s for 100
        features), so set this if you have several hundred columns.
    max_reported : int
        Maximum number of findings, best first. Default 10. Prevents a
        family of derived columns producing dozens of near-identical
        results.
    max_group_size : int
        Largest group to search. Default 3 (pairs and triples). Set 2 for
        pairs only, or 4+ to find larger identities. See the module
        docstring for measured costs.
    gate : float
        A group must reach this adjusted R² before it is extended by one
        more column. Default 0.30. On random data nothing clears it, so
        deeper levels add no false positives; genuine identities produce
        sub-groups well above it (0.71 for a pair inside a 3-way, 0.49
        inside a 4-way).
    max_candidates_per_level : int
        Cap on how many sub-threshold groups are carried forward to the
        next level, best first. Default 200. Without a cap, a frame of 40
        mutually correlated features took 21s at ``max_group_size=4``; with
        it, 0.7s.
    domain : Optional[str]
        Accepted for API consistency; has no effect.
    seed : int
        Seed for the binary-target AUC path's K-fold splits and permutation
        draws (``_binary_combination_auc``); has no effect on a continuous
        target, which is scored by plain OLS with no randomness. Default 0,
        matching this function's previous unconditional behavior, so this
        parameter is purely additive. Exists so a result can be checked
        against a second seed as an independent draw (the flag/no-flag
        outcome should agree, even though the exact score will differ
        slightly).

    Returns
    -------
    List[Issue]
        Zero or more LEK005 Issues (CRITICAL), strongest group first. A
        triple is not reported when one of its own pairs was already
        reported.

    Notes
    -----
    The single-feature guard (see module docstring) excludes a column from
    every candidate group once it already explains the target alone, so a
    column LEK001 already flagged does not also flood the report with every
    group it appears in. That guard checks two metrics, not just this
    module's own adjusted R²: it also checks the column against
    ``audit_equivalence``'s own AUC/Spearman score, taking whichever is
    higher. A column with a strong monotonic but non-linear relationship to
    the target (AUC/Spearman near 1.0, adjusted R² well below the LEK005
    threshold) previously slipped past a guard based on R² alone and was
    reported a second time inside a LEK005 group, claiming no single column
    in that group explains the target when LEK001's own metric said
    otherwise on the same data. ``best_single_adjusted_r2`` in the evidence
    below still reports the pure R² value (falling back to 0.0 if every
    column in the group only qualified via the equivalence-score half of
    the guard).
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

    # Target type/encoding, matching equivalence.py's own rule exactly, so
    # the guard below agrees with LEK001 about what "explains the target
    # alone" means. Resolved before building the matrix, since the matrix
    # now carries this same 0/1 encoding for binary targets (see _Matrix's
    # y01 and _score_arrays' binary-target AUC branch below).
    target_n_unique = y_full.dropna().nunique()
    if target_n_unique == 2:
        categories = sorted(y_full.dropna().unique(), key=str)
        y_encoded = y_full.map({categories[0]: 0.0, categories[1]: 1.0})
        target_type = "binary"
    else:
        y_encoded = y_full.astype(float)
        target_type = "continuous"

    matrix = _Matrix(
        y_full, numeric, features, y01=y_encoded if target_type == "binary" else None
    )

    # Single-column explanatory power, computed once. Used to skip groups
    # whose leakage is already attributable to one column (LEK001's job).
    #
    # Two metrics are checked, not one. `single` (R^2, linear/log OLS, or
    # for a binary target, AUC-of-fit; see _score_arrays) is this module's
    # own scoring and is what gets reported as best_single_adjusted_r2
    # below. But relying on it alone for the guard missed columns LEK001
    # already flags via a different metric (AUC/Spearman). See
    # equivalence._score_feature's docstring for the concrete case.
    # `single_guard` is the max of both, and decides what counts as usable;
    # `single` keeps its original meaning for reporting.
    single: Dict[str, Optional[float]] = {}
    single_guard: Dict[str, Optional[float]] = {}
    for col in features:
        # y01 deliberately not passed here: `single`/`best_single_adjusted_r2`
        # keeps meaning pure R² for reporting. single_guard already covers
        # the binary-target ceiling for the guard's purposes via eq_score
        # (AUC/Spearman) below; the group-level ceiling fix lives in the
        # iterative-deepening loop's own _score_arrays calls further down,
        # which is where a group's combined score, not a single column's,
        # decides whether LEK005 fires.
        y_vals, X_vals, _y01_vals = matrix.block([col])
        r2_score = None
        if (
            y_vals is not None
            and len(y_vals) >= min_obs
            and len(np.unique(X_vals)) >= 2
        ):
            r2_score = _score_arrays(y_vals, X_vals)[0]
        single[col] = r2_score

        eq_result = _equivalence_score(numeric[col], y_encoded, target_type, min_obs)
        eq_score = eq_result["score"] if eq_result is not None else None

        scores = [s for s in (r2_score, eq_score) if s is not None]
        single_guard[col] = max(scores) if scores else None

    # Columns that already explain the target alone (by either metric)
    # belong to LEK001, and any group containing one would trivially score
    # high.
    usable = [
        c
        for c in features
        if single_guard[c] is not None and single_guard[c] < threshold
    ]
    found: List[dict] = []
    reported: List[frozenset] = []

    # ── Iterative deepening ────────────────────────────────────────────────
    # Level 2 is every pair. Each subsequent level extends the surviving
    # groups from the level below by one column. A group survives when it
    # reaches `gate` without reaching `threshold`: it explains a real share
    # of the target but is not yet an identity, the signature of a
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

            # A superset of something already reported is the same finding
            # with a redundant column bolted on.
            if any(prior <= key for prior in reported):
                continue

            columns = sorted(key)
            y_vals, X_vals, y01_vals = matrix.block(columns)
            if y_vals is None or len(y_vals) < min_obs:
                continue
            if any(len(np.unique(X_vals[:, k])) < 2 for k in range(X_vals.shape[1])):
                continue

            score, form = _score_arrays(y_vals, X_vals)

            # Binary-target ceiling fix: plain R² cannot exceed ~0.637
            # against a binary target (see _binary_combination_auc's
            # docstring), so a genuine binary combination clears `gate`
            # (0.30) comfortably but can never reach `threshold` (0.95)
            # through `score` alone. Only attempted once R² has already
            # cleared `gate`, which an independent random pair essentially
            # never does (max 0.075 by chance), so the extra
            # permutation-validated work is paid almost exclusively on
            # candidates already worth the attention.
            if y01_vals is not None and gate <= score < threshold:
                boosted = _binary_combination_auc(
                    y_vals, X_vals, y01_vals, threshold, seed=seed
                )
                if boosted is not None:
                    score, form = boosted

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

        # Carry forward only the strongest sub-groups. Unbounded expansion
        # is what turns a correlated frame into a 21-second scan.
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
        # `single[c]` (pure R^2) can be None for a column that only entered
        # `usable` via the equivalence-score half of the guard above (e.g.
        # too few obs for the OLS fit specifically). Falls back to 0.0
        # rather than crashing on max() over a None.
        best_single = max(
            (single[c] for c in columns if single[c] is not None), default=0.0
        )
        joined = ", ".join(f"'{c}'" for c in columns)
        if item["form"] == "linear":
            relation = "an additive combination (a sum, difference or weighted mix)"
        elif item["form"] == "log":
            relation = "a multiplicative combination (a product or ratio)"
        else:  # "linear-auc": binary target, flagged via _binary_combination_auc
            relation = (
                "a threshold rule on a linear combination (binary target; flagged "
                "by cross-validated AUC separation of the fitted combination, not "
                "by R², since R² alone cannot exceed ~0.64 against a binary "
                "target however perfectly the group determines it)"
            )

        is_auc = item["form"] == "linear-auc"
        metric_name = "cv_auc_separation" if is_auc else "adjusted_r2"
        score_label = "cross-validated AUC separation" if is_auc else "adjusted R²"

        issues.append(
            Issue(
                module="leakage",
                code="LEK005",
                severity=CRITICAL,
                description=(
                    f"Features {joined} together reconstruct target '{target}' "
                    f"({score_label}={item['score']:.4f} >= {threshold}, "
                    f"{item['form']} form), while none does alone (best single "
                    f"adjusted R²={best_single:.4f}). This is combination "
                    f"leakage: the target is likely {relation} of these columns. "
                    f"Review how it was constructed."
                ),
                column=columns[0],
                evidence={
                    "metric": metric_name,
                    "form": item["form"],
                    "group": columns,
                    "group_size": len(columns),
                    "group_score": round(float(item["score"]), 4),
                    # Kept under its original key for continuity with every
                    # non-binary-target LEK005 finding (the overwhelming
                    # majority); for a "linear-auc" finding this key holds
                    # the same AUC-separation value as group_score, not an
                    # R², since no adjusted R² cleared threshold for this
                    # group. See "metric" above for which one it actually is.
                    "group_adjusted_r2": round(float(item["score"]), 4),
                    "best_single_adjusted_r2": round(float(best_single), 4),
                    "threshold": threshold,
                    "n_obs": int(item["n_obs"]),
                },
            )
        )

    return issues
