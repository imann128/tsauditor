"""
tsauditor.remediate
--------------------
The execution layer behind ``GuardReport.apply_fixes``. Where the report's
``suggestions()`` *say* what to do, this *does* it — but only for the columns
the audit actually flagged, and always on a copy.

Design guarantees
-----------------
- **Non-destructive.** The input DataFrame is never mutated; a fresh copy is
  returned. Users can diff the result against their raw source.
- **Report-driven.** Only columns flagged by the audit are touched; healthy,
  unflagged columns are returned byte-for-byte unchanged.
- **Time-series safe.** "Dropping" an outlier means setting it to NaN (so the
  imputation step can fill it), never deleting a row — deleting rows would
  break the index's uniform frequency and re-trigger the gap detectors.
- **Auditable.** A structured change log is attached to the report
  (``report.last_fixes``) recording every column touched and how many cells
  changed.

Outlier and stuck-value masks are recomputed here using the *same* formulas as
the detectors (``anomaly/point.py`` ANO002, ``anomaly/contextual.py`` ANO001).
They are intentionally kept in lockstep; ``tests/test_fix.py`` asserts the
outlier cell count matches the detector's own evidence so the two cannot drift.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

_MISSING_METHODS = {"interpolate", "ffill", "bfill", None}
_OUTLIER_METHODS = {"clip", "nan", "drop", None}
_STUCK_METHODS = {"nan", None}
_LEAKAGE_METHODS = {"drop", None}


# ── threshold resolvers (mirror the detectors' domain defaults) ───────────────


def _zscore_threshold(domain: Optional[str]) -> float:
    """Match anomaly/point.py ANO002."""
    if domain == "finance":
        return 5.0
    if domain == "sensor":
        return 3.5
    return 4.0


def _stuck_window(domain: Optional[str]) -> int:
    """Match anomaly/contextual.py ANO001."""
    if domain == "sensor":
        return 3
    return 5


def _spike_threshold(domain: Optional[str]) -> float:
    """Match anomaly/contextual.py ANO003."""
    if domain == "finance":
        return 4.0
    if domain == "sensor":
        return 3.0
    return 3.5


_SPIKE_WINDOW = 21  # contextual window, matches anomaly/contextual.py ANO003


# ── mask / bound helpers ──────────────────────────────────────────────────────


def _outlier_mask(values: pd.Series, z_thresh: float) -> pd.Series:
    """Combined z-score OR IQR outlier mask, identical to ANO002."""
    mean, std = values.mean(), values.std()
    if std == 0 or pd.isna(std):
        return pd.Series(False, index=values.index)
    z = (values - mean) / std
    z_mask = z.abs() > z_thresh
    q25, q75 = values.quantile([0.25, 0.75])
    iqr = q75 - q25
    iqr_mask = (values < q25 - 1.5 * iqr) | (values > q75 + 1.5 * iqr)
    return z_mask | iqr_mask


def _clip_bounds(values: pd.Series, z_thresh: float) -> tuple:
    """
    Winsorization bounds = the region a point must be in to be flagged by
    *neither* method: the intersection of the z-band and the IQR fence.
    Clipping to [L, U] pulls in exactly the flagged outliers and leaves every
    inlier untouched.
    """
    mean, std = values.mean(), values.std()
    q25, q75 = values.quantile([0.25, 0.75])
    iqr = q75 - q25
    lower = max(mean - z_thresh * std, q25 - 1.5 * iqr)
    upper = min(mean + z_thresh * std, q75 + 1.5 * iqr)
    return lower, upper


def _stuck_mask(series: pd.Series, window: int) -> pd.Series:
    """Run-length stuck-value mask, identical to ANO001."""
    diffs = series.diff().ne(0).cumsum()
    counts = series.groupby(diffs).transform("count")
    return (counts > window) & series.notna()


def _spike_info(values: pd.Series, window: int, threshold: float):
    """
    Contextual-spike mask plus the local clip band, identical to ANO003: each
    point is compared to the mean/std of its surrounding window *excluding
    itself*. Returns (mask, lower, upper) where [lower, upper] is the local
    acceptable band (local_mean ± threshold·local_std); clipping a flagged
    point to it pulls it back to the edge of its own neighbourhood.
    """
    sq = values.pow(2)
    mp = max(3, window // 2)
    roll = values.rolling(window=window, center=True, min_periods=mp)
    roll_sq = sq.rolling(window=window, center=True, min_periods=mp)
    n_excl = roll.count() - 1
    local_mean = (roll.sum() - values) / n_excl
    local_var = (roll_sq.sum() - sq) / n_excl - local_mean.pow(2)
    local_std = np.sqrt(local_var.clip(lower=0))
    deviation = (values - local_mean).abs()
    with np.errstate(divide="ignore", invalid="ignore"):
        z = deviation / local_std
    flat = (local_std == 0) & (deviation > 0) & (n_excl >= 2)
    mask = ((z > threshold) | flat).fillna(False)
    lower = local_mean - threshold * local_std
    upper = local_mean + threshold * local_std
    return mask, lower, upper


def _impute(series: pd.Series, method: str, datetime_index: bool) -> pd.Series:
    if method == "interpolate":
        how = "time" if datetime_index else "linear"
        return series.interpolate(method=how, limit_direction="both")
    if method == "ffill":
        return series.ffill()
    if method == "bfill":
        return series.bfill()
    return series


# ── main entry point ──────────────────────────────────────────────────────────


def apply_fixes(
    report,
    df: pd.DataFrame,
    missing: Optional[str] = "interpolate",
    outliers: Optional[str] = "clip",
    stuck: Optional[str] = "nan",
    leakage: Optional[str] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Return a repaired copy of ``df``, fixing only what the report flagged.

    Parameters
    ----------
    report : GuardReport
        The report produced by ``tsa.scan``. Its issues select which columns
        get fixed.
    df : pd.DataFrame
        The DataFrame to repair. Not modified; a copy is returned.
    missing : {"interpolate", "ffill", "bfill", None}
        How to impute NaNs (including cells newly NaN-ed by outlier/stuck
        handling). Default "interpolate".
    outliers : {"clip", "nan", "drop", None}
        Handles both global point outliers (ANO002) and contextual spikes
        (ANO003). "clip" winsorizes flagged points to the detection bounds —
        global IQR/z bounds for ANO002, the local rolling band for ANO003;
        "nan" sets them to NaN for the imputation step. "drop" is an alias for
        "nan" — rows are never deleted (that would break the time index).
        Default "clip".
    stuck : {"nan", None}
        "nan" replaces flagged stuck runs with NaN. Default "nan".
    leakage : {"drop", None}
        "drop" removes columns flagged by the leakage module. Off by default —
        dropping columns changes the feature matrix and must be explicit.
    verbose : bool
        If True, print a summary of the changes.

    Returns
    -------
    pd.DataFrame
        A new, repaired DataFrame. The original is untouched.
    """
    for name, value, allowed in (
        ("missing", missing, _MISSING_METHODS),
        ("outliers", outliers, _OUTLIER_METHODS),
        ("stuck", stuck, _STUCK_METHODS),
        ("leakage", leakage, _LEAKAGE_METHODS),
    ):
        if value not in allowed:
            raise ValueError(
                f"{name}={value!r} is invalid; choose one of {sorted(str(a) for a in allowed)}."
            )

    # Panel data must be repaired entity by entity. Interpolating an interleaved
    # frame carries one entity's values across into another's gaps: measured on a
    # two-entity panel, a gap in a series sitting near 10 was filled with ~1000
    # from the other entity. See _apply_fixes_by_group.
    group_col = report.metadata.get("group_col")
    if group_col is not None and group_col in df.columns:
        return _apply_fixes_by_group(
            report,
            df,
            group_col=group_col,
            missing=missing,
            outliers=outliers,
            stuck=stuck,
            leakage=leakage,
            verbose=verbose,
        )

    out = df.copy()
    domain = report.metadata.get("domain")
    # Never repair the target column (the label): binary targets trip ANO001,
    # and interpolating a 0/1 label into fractions is wrong.
    protected = report.metadata.get("target")
    datetime_index = isinstance(out.index, pd.DatetimeIndex)
    log: List[Dict[str, Any]] = []

    def _flagged(*codes: str) -> List[str]:
        seen = []
        for issue in report.all_issues:
            if (
                issue.code in codes
                and issue.column in out.columns
                and issue.column != protected
                and issue.column not in seen
            ):
                seen.append(issue.column)
        return seen

    outlier_cols = _flagged("ANO002")
    spike_cols = _flagged("ANO003")
    stuck_cols = _flagged("ANO001")
    missing_cols = _flagged("PRF002", "PRF006")
    nan_filled_cols: set = set()

    # 1. Leakage — drop flagged columns (opt-in only; never the target).
    if leakage == "drop":
        for col in report.leaky_columns():
            if col in out.columns and col != protected:
                out = out.drop(columns=col)
                log.append(
                    {"column": col, "action": "drop_column", "cells_changed": "—"}
                )

    # 2. Outliers — clip to bounds, or NaN-out for imputation.
    if outliers is not None:
        z_thresh = _zscore_threshold(domain)
        for col in outlier_cols:
            if col not in out.columns or not pd.api.types.is_numeric_dtype(out[col]):
                continue
            values = out[col].dropna()
            if outliers == "clip":
                lower, upper = _clip_bounds(values, z_thresh)
                clipped = out[col].clip(lower=lower, upper=upper)
                n = int(((out[col] != clipped) & out[col].notna()).sum())
                out[col] = clipped
                log.append(
                    {
                        "column": col,
                        "action": "clip_outliers",
                        "cells_changed": n,
                        "bounds": (float(lower), float(upper)),
                    }
                )
            else:  # "nan" / "drop"
                mask = _outlier_mask(values, z_thresh)
                idx = mask[mask].index
                out.loc[idx, col] = np.nan
                nan_filled_cols.add(col)
                log.append(
                    {
                        "column": col,
                        "action": "outliers_to_nan",
                        "cells_changed": int(len(idx)),
                    }
                )

        # Contextual spikes (ANO003): a local anomaly, so clip to the local
        # band rather than a global bound, or NaN it for imputation.
        spike_thresh = _spike_threshold(domain)
        for col in spike_cols:
            if col not in out.columns or not pd.api.types.is_numeric_dtype(out[col]):
                continue
            values = out[col].dropna()
            mask, lower, upper = _spike_info(values, _SPIKE_WINDOW, spike_thresh)
            idx = mask[mask].index
            if len(idx) == 0:
                continue
            if outliers == "clip":
                out.loc[idx, col] = out.loc[idx, col].clip(
                    lower=lower.loc[idx], upper=upper.loc[idx]
                )
                log.append(
                    {
                        "column": col,
                        "action": "clip_spikes",
                        "cells_changed": int(len(idx)),
                    }
                )
            else:  # "nan" / "drop"
                out.loc[idx, col] = np.nan
                nan_filled_cols.add(col)
                log.append(
                    {
                        "column": col,
                        "action": "spikes_to_nan",
                        "cells_changed": int(len(idx)),
                    }
                )

    # 3. Stuck values — replace flagged runs with NaN.
    if stuck == "nan":
        window = _stuck_window(domain)
        for col in stuck_cols:
            if col not in out.columns or not pd.api.types.is_numeric_dtype(out[col]):
                continue
            mask = _stuck_mask(out[col], window)
            if mask.any():
                out.loc[mask[mask].index, col] = np.nan
                nan_filled_cols.add(col)
                log.append(
                    {
                        "column": col,
                        "action": "stuck_to_nan",
                        "cells_changed": int(mask.sum()),
                    }
                )

    # 4. Imputation — fill flagged-missing columns plus anything we NaN-ed above.
    if missing is not None:
        impute_cols = set(missing_cols) | nan_filled_cols
        for col in impute_cols:
            if col not in out.columns or not pd.api.types.is_numeric_dtype(out[col]):
                continue
            before = out[col].isna().sum()
            out[col] = _impute(out[col], missing, datetime_index)
            filled = int(before - out[col].isna().sum())
            if filled:
                log.append(
                    {
                        "column": col,
                        "action": f"impute_{missing}",
                        "cells_changed": filled,
                    }
                )

    report.last_fixes = log
    if verbose:
        _print_log(log)
    return out


def _apply_fixes_by_group(
    report,
    df: pd.DataFrame,
    group_col: str,
    missing: Optional[str],
    outliers: Optional[str],
    stuck: Optional[str],
    leakage: Optional[str],
    verbose: bool,
) -> pd.DataFrame:
    """
    Repair a panel entity by entity.

    Each entity is repaired as its own independent time series using exactly the
    single-series path, with a report view narrowed to that entity's issues, then
    written back by **position**. Positional write-back matters: a panel index has
    the same timestamp once per entity, so a label-based ``.loc`` assignment would
    scatter one entity's repairs across all of them.

    Leaky-column drops are applied once to the whole frame rather than per entity,
    since a column either exists in the feature matrix or it does not.
    """
    from tsauditor.report.summary import GuardReport

    out = df.copy()
    log: List[Dict[str, Any]] = []
    protected = report.metadata.get("target")

    # 1. Leakage drops are frame-wide. Collect across every entity and the
    #    panel-level checks, then drop once.
    if leakage == "drop":
        for col in report.leaky_columns():
            if col in out.columns and col != protected and col != group_col:
                out = out.drop(columns=col)
                log.append(
                    {"column": col, "action": "drop_column", "cells_changed": "—"}
                )

    groups = out[group_col].to_numpy()
    payload_cols = [c for c in out.columns if c != group_col]

    for key in pd.unique(groups):
        positions = np.flatnonzero(groups == key)
        if positions.size == 0:
            continue

        sub = out.iloc[positions][payload_cols]

        # A report view containing only this entity's findings. group_col is
        # removed from the metadata so the recursive call takes the ordinary
        # single-series path.
        view_metadata = {
            k: v
            for k, v in report.metadata.items()
            if k not in ("group_col", "n_groups")
        }
        view = GuardReport(metadata=view_metadata)
        for issue in report.all_issues:
            if issue.group == str(key):
                view.critical.append(issue)  # bucket does not matter; all_issues merges

        repaired = apply_fixes(
            view,
            sub,
            missing=missing,
            outliers=outliers,
            stuck=stuck,
            leakage=None,  # already handled frame-wide above
            verbose=False,
        )

        for col in repaired.columns:
            out.iloc[positions, out.columns.get_loc(col)] = repaired[col].to_numpy()

        for entry in view.last_fixes:
            entry = dict(entry)
            entry["group"] = str(key)
            log.append(entry)

    report.last_fixes = log
    if verbose:
        _print_log(log)
    return out


def fix(
    df: pd.DataFrame,
    target: Optional[str] = None,
    time_col: Optional[str] = None,
    domain: Optional[str] = None,
    missing: Optional[str] = "interpolate",
    outliers: Optional[str] = "clip",
    stuck: Optional[str] = "nan",
    leakage: Optional[str] = None,
    verbose: bool = False,
):
    """
    One-shot scan-and-repair. Scans ``df`` and returns ``(clean_df, report)``.

    A convenience wrapper over ``scan()`` + ``GuardReport.apply_fixes()``. It
    always returns *both* the repaired copy and the report, so the audit trail
    (``report.last_fixes``, ``report.leaky_columns()``, the full issue list) is
    never silently discarded — you keep the record of what changed and why.

    The input ``df`` is never modified; ``clean_df`` is an independent copy.
    Pass ``target=`` so the label column is protected from every repair.

    Parameters
    ----------
    df, target, time_col, domain
        Passed through to ``scan``.
    missing, outliers, stuck, leakage, verbose
        Passed through to ``apply_fixes``.

    Returns
    -------
    (clean_df, report) : tuple[pd.DataFrame, GuardReport]

    Examples
    --------
    >>> clean, report = tsa.fix(df, target="Direction", domain="finance")
    >>> report.last_fixes          # exactly what changed
    >>> report.leaky_columns()     # what it flagged
    """
    from tsauditor.scanner import scan

    report = scan(df, target=target, time_col=time_col, domain=domain)
    clean = apply_fixes(
        report,
        df,
        missing=missing,
        outliers=outliers,
        stuck=stuck,
        leakage=leakage,
        verbose=verbose,
    )
    return clean, report


def _print_log(log: List[Dict[str, Any]]) -> None:
    try:
        from rich.console import Console

        console = Console()
        if not log:
            console.print("[green]apply_fixes: nothing to repair.[/green]")
            return
        console.print("[bold]apply_fixes — changes applied[/bold]")
        for entry in log:
            console.print(
                f"  • {entry['column']}: {entry['action']} "
                f"({entry['cells_changed']} cells)"
            )
    except Exception:
        pass


# ── Data Health Score ─────────────────────────────────────────────────────────
_QUALITY_CODES = ("PRF002", "PRF006", "ANO001", "ANO002", "ANO003")


def affected_cells(report, df: pd.DataFrame) -> int:
    """
    Count distinct data cells implicated by detected *quality* issues (missing,
    point outliers, contextual spikes, stuck runs). Leakage is excluded — a
    leaky column is a modeling risk, not a corrupt cell. Cells flagged by more
    than one detector in the same column are counted once.
    """
    domain = report.metadata.get("domain")
    z_thresh = _zscore_threshold(domain)
    window = _stuck_window(domain)
    spike_thresh = _spike_threshold(domain)

    by_col: Dict[str, set] = {}
    for issue in report.all_issues:
        if issue.code in _QUALITY_CODES and issue.column in df.columns:
            by_col.setdefault(issue.column, set()).add(issue.code)

    total = 0
    for col, codes in by_col.items():
        s = df[col]
        if not pd.api.types.is_numeric_dtype(s):
            continue
        mask = pd.Series(False, index=s.index)
        if {"PRF002", "PRF006"} & codes:
            mask |= s.isna()
        values = s.dropna()
        if "ANO002" in codes and len(values):
            om = _outlier_mask(values, z_thresh)
            mask.loc[om[om].index] = True
        if "ANO003" in codes and len(values):
            sm, _, _ = _spike_info(values, _SPIKE_WINDOW, spike_thresh)
            mask.loc[sm[sm].index] = True
        if "ANO001" in codes:
            km = _stuck_mask(s, window)
            mask |= km.fillna(False)
        total += int(mask.sum())
    return total


def health_score(report, df: pd.DataFrame) -> float:
    """
    Data Health Score: percentage of numeric data cells *not* implicated by any
    quality issue.  ``100 * (1 - affected_cells / total_cells)``, rounded to one
    decimal. Returns 100.0 when there are no numeric cells to assess.
    """
    numeric_cols = df.select_dtypes(include="number").shape[1]
    total = len(df) * numeric_cols
    if total == 0:
        return 100.0
    return round(100.0 * (1 - affected_cells(report, df) / total), 1)
