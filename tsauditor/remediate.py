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

Outlier, stuck-value, and spike masks are computed here via the *same*
functions the detectors use (``anomaly/point.py`` ANO002,
``anomaly/contextual.py`` ANO001/ANO003), imported from
``tsauditor.anomaly._common``. Detection and repair share one copy of every
threshold preset and every masking formula, so they cannot drift apart --
this used to be a set of hand-maintained duplicates connected only by
comments, which drifted out of sync at least once in practice; see
CHANGELOG [0.5.0] for the incident and ``tsauditor/anomaly/_common.py``
for the shared implementation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from tsauditor.anomaly._common import (
    zscore_preset,
    stuck_window_preset,
    spike_threshold_preset,
    zscore_iqr_masks,
    clip_bounds,
    stuck_run_mask,
    spike_bounds,
    SPIKE_WINDOW,
)
from tsauditor.utils.validation import _is_polars, _polars_to_pandas

_MISSING_METHODS = {"interpolate", "ffill", "bfill", None}
_OUTLIER_METHODS = {"clip", "nan", "drop", None}
_STUCK_METHODS = {"nan", None}
_LEAKAGE_METHODS = {"drop", None}


def _outlier_mask(values: pd.Series, z_thresh: float) -> pd.Series:
    """Combined z-score OR IQR outlier mask -- thin wrapper over the shared
    ANO002 mask, which also detects the degenerate (zero-variance) case."""
    z_mask, iqr_mask, _, _ = zscore_iqr_masks(values, z_thresh)
    return z_mask | iqr_mask


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

    time_col = report.metadata.get("time_col")

    # polars input was never actually reachable through apply_fixes()/fix():
    # polars.DataFrame has neither .copy() nor .index, so the very first
    # line below (`out = df.copy()`, or the isinstance(df.index, ...) check
    # further down) raised AttributeError immediately -- this predates every
    # other fix in this function; test_polars.py only ever exercised scan(),
    # never fix()/apply_fixes(), so the gap had no coverage on either side.
    # Converted here the same way validate_dataframe does at the scan()
    # boundary: internals stay pandas, and so does the return value --
    # apply_fixes has never returned anything polars-shaped, and scan()
    # itself only ever produces a plain GuardReport regardless of input
    # type, so a pandas return here is consistent with the existing
    # boundary, not a new inconsistency.
    if _is_polars(df):
        df = _polars_to_pandas(df, time_col)

    # Resolve time_col the same way scan()'s validate_dataframe does, so a
    # caller who used scan(df, time_col=...) and now calls
    # report.apply_fixes(df) (or tsa.fix(df, time_col=...)) gets a
    # correctly time-ordered repair -- not one silently computed on
    # whatever row order the raw time_col *column* happened to be in.
    # Without this, df.index is a meaningless RangeIndex (or whatever
    # index the caller's own df had), so the DatetimeIndex sort-safety
    # further below never engages at all for time_col callers: the exact
    # same "found the issue, repaired zero cells" failure as an
    # out-of-order DatetimeIndex, just reached through time_col instead of
    # a pre-set index. Restored to the caller's original shape (time_col
    # back as a plain column, not the index) before returning.
    restore_time_col = (
        time_col is not None
        and time_col in df.columns
        and not isinstance(df.index, pd.DatetimeIndex)
    )
    if restore_time_col:
        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.set_index(time_col)

    # Panel data must be repaired entity by entity. Interpolating an interleaved
    # frame carries one entity's values across into another's gaps: measured on a
    # two-entity panel, a gap in a series sitting near 10 was filled with ~1000
    # from the other entity. See _apply_fixes_by_group.
    group_col = report.metadata.get("group_col")
    if group_col is not None and group_col in df.columns:
        out = _apply_fixes_by_group(
            report,
            df,
            group_col=group_col,
            missing=missing,
            outliers=outliers,
            stuck=stuck,
            leakage=leakage,
            verbose=verbose,
        )
        return out.reset_index() if restore_time_col else out

    # scan() validates and sorts its own working copy before running any
    # detector (see utils.validation.ensure_sorted_datetime_index) -- but
    # `report` only carries the resulting Issues, not that sorted frame.
    # `df` here is whatever the caller passed to fix()/apply_fixes(),
    # completely independent of what scan() saw. If the caller's `df` has a
    # valid DatetimeIndex that is out of chronological order, every mask
    # this function computes below (stuck_run_mask's consecutive-run walk,
    # spike_bounds' rolling window) previously ran on that unsorted order
    # directly and could find nothing at all, even for a column the report
    # says was flagged -- repairing zero cells while report.last_fixes and
    # the caller both believe the data was cleaned. Restore chronological
    # order via position (not `.sort_index()` + relabel) specifically so a
    # duplicate timestamp -- already its own separate CRITICAL PRF004
    # finding -- does not turn the final reordering into an ambiguous
    # label-based reindex; and restore the caller's original row order
    # before returning, both so the "byte-for-byte unchanged" guarantee for
    # untouched columns holds in the caller's own row order, and because
    # `_apply_fixes_by_group` writes this function's per-entity result back
    # by raw position and requires the row order to match what it passed in.
    datetime_index = isinstance(df.index, pd.DatetimeIndex)
    restore_positions: Optional[np.ndarray] = None
    if datetime_index:
        sort_positions = np.argsort(df.index.values, kind="mergesort")
        if not np.array_equal(sort_positions, np.arange(len(df))):
            restore_positions = np.empty_like(sort_positions)
            restore_positions[sort_positions] = np.arange(len(sort_positions))
            df = df.iloc[sort_positions]

    out = df.copy()
    domain = report.metadata.get("domain")
    # Never repair the target column (the label): binary targets trip ANO001,
    # and interpolating a 0/1 label into fractions is wrong.
    protected = report.metadata.get("target")
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
    #
    # Every mask/bounds computation below reads from `df[col]` (the pristine,
    # pre-repair column), never `out[col]`. A column can carry more than one
    # finding (e.g. ANO002 and ANO003, or ANO002 and ANO001 -- a value stuck
    # at an extreme constant is both a global outlier and a stuck run), and
    # these steps run in a fixed order. Reading `out[col]` meant a later
    # step's detection ran on a column an earlier step had already clipped or
    # NaN-ed, which could silently change what that later step found: the
    # nan branches could rediscover nothing for cells an earlier step already
    # NaN-ed (so the change log wrongly credited only the first action, even
    # though the audit had raised both), and the local rolling stats behind
    # ANO003's spike detection could shift for *unrelated* nearby cells once
    # an earlier clip altered a value inside their rolling window. Detecting
    # against the same pristine input the original audit scored means a
    # later step here always finds exactly what its own Issue reported,
    # regardless of what an earlier step already touched.
    if outliers is not None:
        z_thresh = zscore_preset(domain)
        for col in outlier_cols:
            if col not in out.columns or not pd.api.types.is_numeric_dtype(out[col]):
                continue
            values = df[col].dropna()
            if outliers == "clip":
                lower, upper = clip_bounds(values, z_thresh)
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
                if len(idx) == 0:
                    continue
                # cells_changed counts only cells this step actually flips
                # to NaN for the first time, so summing cells_changed across
                # the whole log never double-counts a cell two detectors
                # both flagged. already_nan records the rest -- cells this
                # detector's own mask covers but an earlier step (e.g. a
                # value that is both a stuck run and a global outlier) had
                # already NaN-ed -- so the log still shows ANO002 fired on
                # them even though this specific action changed nothing.
                # Always logged when the mask fires, even at cells_changed=0,
                # so provenance for every contributing detector survives,
                # not just whichever ran first.
                already_nan = int(out.loc[idx, col].isna().sum())
                newly = int(len(idx) - already_nan)
                out.loc[idx, col] = np.nan
                nan_filled_cols.add(col)
                log.append(
                    {
                        "column": col,
                        "action": "outliers_to_nan",
                        "cells_changed": newly,
                        "already_nan": already_nan,
                    }
                )

        # Contextual spikes (ANO003): a local anomaly, so clip to the local
        # band rather than a global bound, or NaN it for imputation.
        spike_thresh = spike_threshold_preset(domain)
        for col in spike_cols:
            if col not in out.columns or not pd.api.types.is_numeric_dtype(out[col]):
                continue
            values = df[col].dropna()
            mask, lower, upper = spike_bounds(values, SPIKE_WINDOW, spike_thresh)
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
                already_nan = int(out.loc[idx, col].isna().sum())
                newly = int(len(idx) - already_nan)
                out.loc[idx, col] = np.nan
                nan_filled_cols.add(col)
                log.append(
                    {
                        "column": col,
                        "action": "spikes_to_nan",
                        "cells_changed": newly,
                        "already_nan": already_nan,
                    }
                )

    # 3. Stuck values — replace flagged runs with NaN.
    if stuck == "nan":
        window = stuck_window_preset(domain)
        for col in stuck_cols:
            if col not in out.columns or not pd.api.types.is_numeric_dtype(out[col]):
                continue
            mask, _ = stuck_run_mask(df[col], window)
            if mask.any():
                idx = mask[mask].index
                already_nan = int(out.loc[idx, col].isna().sum())
                newly = int(len(idx) - already_nan)
                out.loc[idx, col] = np.nan
                nan_filled_cols.add(col)
                log.append(
                    {
                        "column": col,
                        "action": "stuck_to_nan",
                        "cells_changed": newly,
                        "already_nan": already_nan,
                    }
                )

    # 3b. Infinite values — always converted to NaN, then imputed with everything
    #     else if `missing` is enabled.
    #
    #     Unconditional, unlike every other repair above, because there is no
    #     reading of an infinity under which keeping it is correct: it is the
    #     residue of a failed calculation upstream, not a measurement. Leaving it
    #     poisons the mean and standard deviation of the whole column and makes
    #     scikit-learn raise at fit time. If the caller disabled imputation
    #     (`missing=None`) the cell is left as NaN, which is honest about the
    #     value being unknown and is handled gracefully by pandas.
    for col in _flagged("PRF007"):
        if col not in out.columns or not pd.api.types.is_numeric_dtype(out[col]):
            continue
        mask = np.isinf(out[col].to_numpy(dtype=float, copy=False))
        if mask.any():
            out.loc[out.index[mask], col] = np.nan
            nan_filled_cols.add(col)
            log.append(
                {
                    "column": col,
                    "action": "non_finite_to_nan",
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

    if restore_positions is not None:
        out = out.iloc[restore_positions]

    report.last_fixes = log
    if verbose:
        _print_log(log)
    return out.reset_index() if restore_time_col else out


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

    # Rows with a null entity id are never scanned per-entity (see PNL004 in
    # tsauditor.panel), so there is no entity-specific report view or
    # distribution to repair them from. Leave them untouched, explicitly and
    # logged once, rather than have them silently vanish from the loop below
    # via `groups == key` never matching NaN.
    null_rows = pd.isna(groups)
    n_null = int(null_rows.sum())
    if n_null > 0:
        log.append(
            {
                "column": group_col,
                "action": "skip_null_group_rows",
                "cells_changed": n_null,
            }
        )

    for key in pd.unique(groups):
        if pd.isna(key):
            continue
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
    available_at: Optional[dict] = None,
    constraints: Optional[dict] = None,
    group_col: Optional[str] = None,
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
    df, target, time_col, domain, available_at, constraints, group_col
        Passed through to ``scan``. Without ``available_at=``, LEK004 (as-of
        leakage) never runs; without ``constraints=``, VAL001/VAL002 never
        run — both are opt-in because tsauditor cannot infer a release
        schedule or a validity bound on its own. Before this, the only way to
        exercise either check together with a one-shot repair was to call
        ``scan()`` and ``apply_fixes()`` separately; ``fix()`` silently
        skipped them with no error, which read as "nothing wrong" rather
        than "not checked." ``group_col`` was the same story for panel
        (long-format, multi-entity) data: every other panel-aware entry
        point (``scan()``, ``GuardReport.apply_fixes()``, ``health_score()``,
        ``to_json()``, ``to_pdf()``) accepted or threaded it, but ``fix()``
        itself had no parameter for it at all -- ``tsa.fix(panel_df,
        group_col=...)`` raised ``TypeError: unexpected keyword argument``,
        forcing panel users to always fall back to the two-call form this
        function exists to avoid. ``apply_fixes`` itself needs no separate
        argument for this: it reads ``group_col`` back off
        ``report.metadata``, which ``scan()`` populates.
    missing, outliers, stuck, leakage, verbose
        Passed through to ``apply_fixes``.

    Returns
    -------
    (clean_df, report) : tuple[pd.DataFrame, GuardReport]

    Examples
    --------
    >>> clean, report = tsa.fix(df, target="Direction", domain="finance")  # doctest: +SKIP
    >>> report.last_fixes          # exactly what changed  # doctest: +SKIP
    >>> report.leaky_columns()     # what it flagged  # doctest: +SKIP
    """
    from tsauditor.scanner import scan

    report = scan(
        df,
        target=target,
        time_col=time_col,
        domain=domain,
        available_at=available_at,
        constraints=constraints,
        group_col=group_col,
    )
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
_QUALITY_CODES = ("PRF002", "PRF006", "PRF007", "ANO001", "ANO002", "ANO003")


def _affected_cells_single(
    issues, df: pd.DataFrame, z_thresh: float, window: int, spike_thresh: float
) -> int:
    """Affected-cell count for one series (single entity or non-panel), given
    only *its own* Issues. Factored out of affected_cells so the panel path
    below can call it once per entity instead of once for the whole panel."""
    by_col: Dict[str, set] = {}
    for issue in issues:
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
        if "PRF007" in codes:
            # Counted as corrupt cells in their own right. isinf is False for
            # NaN, so a column flagged for both does not double-count: the
            # union below is over distinct positions.
            mask |= pd.Series(
                np.isinf(s.to_numpy(dtype=float, copy=False)), index=s.index
            )
        values = s.dropna()
        if "ANO002" in codes and len(values):
            om = _outlier_mask(values, z_thresh)
            mask.loc[om[om].index] = True
        if "ANO003" in codes and len(values):
            sm, _, _ = spike_bounds(values, SPIKE_WINDOW, spike_thresh)
            mask.loc[sm[sm].index] = True
        if "ANO001" in codes:
            km, _ = stuck_run_mask(s, window)
            mask |= km.fillna(False)
        total += int(mask.sum())
    return total


def affected_cells(report, df: pd.DataFrame) -> int:
    """
    Count distinct data cells implicated by detected *quality* issues (missing,
    point outliers, contextual spikes, stuck runs). Leakage is excluded — a
    leaky column is a modeling risk, not a corrupt cell. Cells flagged by more
    than one detector in the same column are counted once.

    Panel-aware. When the report came from ``scan(group_col=...)``, every
    quality detector ran on one entity at a time (see scanner.py's
    per-partition loop) -- so re-deriving each mask from the raw, interleaved
    ``df`` instead would compute one mean/std/rolling-window across every
    entity's mixed values at once: a real outlier in a small-scale entity can
    be diluted below a large-scale entity's ordinary range and vanish
    entirely, or the reverse -- an entity's normal values can look anomalous
    next to a very different one. Recomputing per entity, on only that
    entity's own Issues, mirrors what the original per-entity scan and
    ``apply_fixes``'s ``_apply_fixes_by_group`` both already do; this is what
    makes ``health_score()`` and ``to_json(df=...)``'s health block trustworthy
    for panel data instead of silently wrong.
    """
    domain = report.metadata.get("domain")
    z_thresh = zscore_preset(domain)
    window = stuck_window_preset(domain)
    spike_thresh = spike_threshold_preset(domain)

    group_col = report.metadata.get("group_col")
    if group_col is None or group_col not in df.columns:
        return _affected_cells_single(
            report.all_issues, df, z_thresh, window, spike_thresh
        )

    total = 0
    groups = df[group_col].to_numpy()
    for key in pd.unique(groups):
        if pd.isna(key):
            # PNL004: rows with a null entity id are never scanned per-entity,
            # so there is no entity-specific Issue list to recompute against.
            continue
        sub = df[groups == key]
        issues = [i for i in report.all_issues if i.group == str(key)]
        total += _affected_cells_single(issues, sub, z_thresh, window, spike_thresh)
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
