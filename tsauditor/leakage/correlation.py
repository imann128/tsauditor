"""
tsauditor.leakage.correlation
------------------------------
Cross-correlation leakage detection across a range of lags.

A legitimate feature should carry its information from the past or present:
its association with the target should peak at lag <= 0. If a feature's
peak cross-correlation with the target occurs at a *positive* lag, the
feature aligns most strongly with *future* target values: a signature of
lookahead leakage.

Detection method
----------------
For each numeric feature, compute the rank (Spearman) cross-correlation
with the target across lags in [-max_lag, +max_lag], where

    r(tau) = corr( feature_t , target_{t+tau} )

so tau > 0 means the feature is being compared against the target's future.
If the peak |r| occurs at a positive lag AND exceeds ``min_correlation``,
raise LEK002.

Spearman is used (not Pearson) for consistency with the equivalence module
and because it is robust and captures monotonic association; binary targets
are encoded 0/1 (the rank correlation is attenuated but the *lag* of the
peak (the actual signal here) is preserved).

Important limitation
--------------------
In pure cross-correlation a genuine strong predictor and a lookahead leak
both peak at a positive lag. The separator is magnitude: real one-step
predictive power is weak, whereas leakage is strong. LEK002 is therefore a
WARNING-level *suspicion* flag for review, not a proof of leakage.

Issue codes raised
------------------
LEK002  Positive-lag peak detected.  (WARNING)

Visualizing this check
-----------------------
``lag_correlation_matrix()`` exposes the full feature x lag correlation grid
that ``audit_correlation_leakage`` searches over (used by
``GuardReport.to_pdf``'s lead/lag heatmap). Both functions are built on the
same ``_lag_correlation_core`` so the heatmap can never show a peak the
detector didn't also see, or vice versa. A visualization silently drifting
out of sync with the detector it's supposed to represent is its own class of
bug, the same failure mode as feeding a model a leaky column nobody
double-checked against the audit that flagged it.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from tsauditor.leakage._common import encode_target as _encode_target
from tsauditor.report.summary import Issue, WARNING
from tsauditor.utils.validation import ensure_sorted_datetime_index


def _align(a: np.ndarray, b: np.ndarray, tau: int):
    """Slice ``a`` and ``b`` so element i pairs a_t with b_{t+tau}."""
    n = len(a)
    # Both branches below produce two same-length slices except when
    # abs(tau) > n (a lag larger than the whole series; routine once
    # max_lag is compared against a short panel entity, e.g. this
    # package's own default max_lag=10 against any series of 10 rows or
    # fewer). A start-slice (`a[s:]`) already clamps to length 0 in that
    # case; Python's stop-slice (`a[:n-tau]`) does not: a negative
    # stop wraps from the end instead, silently returning `abs(n-tau)`
    # elements instead of 0. That produced two differently-shaped arrays
    # downstream (`np.isnan(a) | np.isnan(b)` then raises a raw
    # ValueError) instead of the empty-but-shape-matched pair `mask.sum()
    # < min_obs` was written to skip over. Clamping the stop bound to 0
    # makes both slices agree at length 0 exactly when the start-slice
    # side already would.
    if tau >= 0:
        end = max(n - tau, 0)
        return a[:end], b[tau:]
    s = -tau
    end = max(n - s, 0)
    return a[s:], b[:end]


def _lag_correlation_core(
    df: pd.DataFrame, target: str, ry: np.ndarray, max_lag: int, min_obs: int
) -> Tuple[List[str], np.ndarray]:
    """
    Compute the Spearman correlation of every numeric non-target column
    against the (already rank-transformed) target, at every lag in
    ``[-max_lag, +max_lag]``.

    Shared core for ``audit_correlation_leakage`` (which reduces each row to
    its peak) and ``lag_correlation_matrix`` (which returns the grid as-is).
    Factored out so there is exactly one place that decides what "the
    correlation of this feature with the target at this lag" means.
    Duplicating this loop for the heatmap would risk the two silently
    disagreeing (e.g. a rounding or masking difference) about the same
    underlying number.

    Returns
    -------
    (feature_names, matrix) : (List[str], np.ndarray)
        ``matrix`` has shape ``(len(feature_names), 2 * max_lag + 1)``, column
        ``k`` corresponding to lag ``k - max_lag``. Entries are ``NaN`` where
        fewer than ``min_obs`` paired observations exist at that lag, or
        either side is constant on the overlap (undefined correlation), the
        same skip conditions ``audit_correlation_leakage`` has always
        applied, now visible instead of silently discarded.
    """
    n_lags = 2 * max_lag + 1
    feature_cols = [
        c for c in df.select_dtypes(include=["number"]).columns if c != target
    ]
    matrix = np.full((len(feature_cols), n_lags), np.nan)

    for i, col in enumerate(feature_cols):
        x = df[col].astype(float).replace([np.inf, -np.inf], np.nan)
        if x.nunique() < 2:
            continue
        rx = x.rank().to_numpy(dtype=float)

        for k, tau in enumerate(range(-max_lag, max_lag + 1)):
            a, b = _align(rx, ry, tau)
            mask = ~(np.isnan(a) | np.isnan(b))
            if int(mask.sum()) < min_obs:
                continue
            aa, bb = a[mask], b[mask]
            if aa.std() == 0 or bb.std() == 0:  # constant subset -> undefined
                continue
            r = float(np.corrcoef(aa, bb)[0, 1])
            if np.isnan(r):
                continue
            matrix[i, k] = r

    return feature_cols, matrix


def lag_correlation_matrix(
    df: pd.DataFrame,
    target: str,
    max_lag: int = 10,
    min_obs: int = 30,
    domain: Optional[str] = None,
) -> pd.DataFrame:
    """
    Full feature x lag Spearman cross-correlation grid against ``target``.

    This is the same computation ``audit_correlation_leakage`` (LEK002) runs
    internally, exposed in full instead of reduced to each feature's peak,
    e.g. to drive ``GuardReport.to_pdf``'s lead/lag heatmap, or your own
    plotting. A cell is ``NaN`` where there were fewer than ``min_obs``
    overlapping observations at that lag, or one side was constant on the
    overlap; that is not the same as a correlation of 0, so do not fill it
    with 0 before plotting or aggregating.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with rows in time order (sorted DatetimeIndex).
    target : str
        Name of the target/label column. Must exist in ``df``.
    max_lag : int
        Maximum lag (in periods) to compute in each direction. Default 10,
        matching ``audit_correlation_leakage``'s default.
    min_obs : int
        Minimum overlapping observations at a lag for it to be computed.
        Default 30, matching ``audit_correlation_leakage``'s default.
    domain : Optional[str]
        Accepted for API consistency; unused.

    Returns
    -------
    pd.DataFrame
        Index: numeric feature columns (target and non-numeric columns
        excluded). Columns: integer lags from ``-max_lag`` to ``+max_lag``.
        Empty (no rows) if the target has fewer than two distinct values
        after encoding.
    """
    if target not in df.columns:
        raise ValueError(f"target '{target}' not found in DataFrame columns.")

    df = ensure_sorted_datetime_index(df, "lag_correlation_matrix")

    y = _encode_target(df[target], target)
    lags = list(range(-max_lag, max_lag + 1))
    if y.dropna().nunique() < 2:
        return pd.DataFrame(columns=lags)

    ry = y.rank().to_numpy(dtype=float)
    feature_cols, matrix = _lag_correlation_core(df, target, ry, max_lag, min_obs)
    return pd.DataFrame(matrix, index=feature_cols, columns=lags)


def audit_correlation_leakage(
    df: pd.DataFrame,
    target: str,
    max_lag: int = 10,
    min_correlation: float = 0.5,
    min_obs: int = 30,
    domain: Optional[str] = None,
) -> List[Issue]:
    """
    Detect leakage via a cross-correlation peak at positive lags.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with rows in time order (sorted DatetimeIndex).
    target : str
        Name of the target column. Must exist in ``df``.
    max_lag : int
        Maximum lag (in periods) to test in each direction. Default 10.
    min_correlation : float
        Minimum absolute correlation for a peak at a positive lag to be
        reported. Default 0.5.

        This gate carries more weight than it appears to. The rule below fires
        whenever the argmax over lags lands at a positive lag, and for two
        persistent series (a price level, a random walk, a slow AR process)
        spurious correlation is large by construction while *which* lag wins is
        close to a coin flip. A low gate therefore reports leakage between
        columns that are statistically independent.

        Measured over 100 trials per cell on 400-point series, where FP columns
        are two independently generated series and TP columns are a genuine
        t+1 lookahead::

            min_correlation   FP walk   FP AR(.98)   TP iid   TP walk
            0.1 (until 0.3.1)    37%          51%     100%      100%
            0.5 (current)        13%           8%     100%      100%

        Raising the gate removed no true positive in 200 trials.

        Note for anyone tempted to replace this with a margin over the lag-0
        correlation, which is what LEK003 does: it does not work here. On a
        persistent target a genuine lookahead correlates with the target at
        lag 0 almost as strongly as at lag 1, so a flat margin suppresses real
        leaks too. Measured, a 0.10 margin cut false positives to 3% but
        dropped true detection on a random-walk target from 100% to 0%.
        LEK003 escapes this by dividing by the target's *measured*
        autocorrelation rather than subtracting a constant.
    min_obs : int
        Minimum overlapping observations at the peak lag for it to count.
        Default 30.
    domain : Optional[str]
        Accepted for API consistency.

    Returns
    -------
    List[Issue]
        One LEK002 Issue per flagged feature column.
    """
    issues: List[Issue] = []

    if target not in df.columns:
        raise ValueError(f"target '{target}' not found in DataFrame columns.")

    # The lag search below is positional (_align slices by integer offset),
    # so an out-of-order-but-valid DatetimeIndex silently produces wrong
    # lags rather than an error. See ensure_sorted_datetime_index's docstring.
    df = ensure_sorted_datetime_index(df, "audit_correlation_leakage")

    y = _encode_target(df[target], target)
    if y.dropna().nunique() < 2:
        return issues

    # Rank-transform the target once (Spearman == Pearson of ranks). Ranking the
    # full series a single time and correlating the shifted ranks across lags
    # avoids re-ranking on every lag: the previous hot path.
    ry = y.rank().to_numpy(dtype=float)

    # _lag_correlation_core is the single place that computes "correlation of
    # feature X with target at lag tau". lag_correlation_matrix (the heatmap
    # data) is built on the same call, so the two can never disagree about
    # the same underlying numbers.
    feature_cols, matrix = _lag_correlation_core(df, target, ry, max_lag, min_obs)
    lags = np.arange(-max_lag, max_lag + 1)

    for col, row in zip(feature_cols, matrix):
        valid = ~np.isnan(row)
        if not valid.any():
            continue

        # argmax over |r|, ties broken toward the smallest (most negative) tau:
        # same behavior as the previous strict `>` update inside an
        # ascending-tau loop, since row/lags are already ordered by ascending
        # tau and np.argmax returns the first occurrence on a tie.
        abs_row = np.where(valid, np.abs(row), -np.inf)
        idx = int(np.argmax(abs_row))
        best_abs = float(abs_row[idx])
        best_lag = int(lags[idx])
        best_signed = float(row[idx])

        if best_lag > 0 and best_abs >= min_correlation:
            issues.append(
                Issue(
                    module="leakage",
                    code="LEK002",
                    severity=WARNING,
                    description=(
                        f"Feature '{col}' has its peak cross-correlation with target "
                        f"'{target}' at lag +{best_lag} (Spearman={best_signed:.3f}); it "
                        f"aligns most strongly with future target values, suggesting "
                        f"lookahead leakage. Review how this feature is constructed."
                    ),
                    column=col,
                    evidence={
                        "peak_lag": int(best_lag),
                        "peak_correlation": round(best_signed, 4),
                        "min_correlation": min_correlation,
                        "max_lag": max_lag,
                        "metric": "spearman",
                    },
                )
            )

    return issues
