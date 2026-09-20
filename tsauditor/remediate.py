"""
tsauditor.remediate
--------------------
The execution layer behind ``GuardReport.apply_fixes``. Where the report's
``suggestions()`` say what to do, this does it, but only for the columns
the audit actually flagged, and always on a copy.

Design guarantees
-----------------
- **Non-destructive.** The input DataFrame is never mutated; a fresh copy is
  returned.
- **Report-driven.** Only columns flagged by the audit are touched; healthy,
  unflagged columns are returned byte-for-byte unchanged.
- **Time-series safe.** "Dropping" an outlier means setting it to NaN (so the
  imputation step can fill it), never deleting a row, since deleting rows
  would break the index's uniform frequency and re-trigger the gap detectors.
- **Auditable.** A structured change log is attached to the report
  (``report.last_fixes``) recording every column touched and how many cells
  changed.

Outlier, stuck-value, and spike masks are computed here via the same
functions the detectors use (``anomaly/point.py`` ANO002,
``anomaly/contextual.py`` ANO001/ANO003), imported from
``tsauditor.anomaly._common``, so detection and repair cannot drift apart.
See CHANGELOG [0.5.0] for the incident that motivated this.
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
    esd_masking_recovery,
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
    """Combined z-score OR IQR outlier mask, plus ESD-recovered points under
    masking. Must match what ``audit_point_anomalies`` (ANO002) flags, or
    ``apply_fixes(outliers="nan"/"drop")`` silently leaves some of what
    ``scan()`` reported untouched.

    Uses ``esd_masking_recovery`` (``tsauditor.anomaly._common``), the same
    function ``audit_point_anomalies`` uses for this decision, rather than an
    independent re-derivation. See ``anomaly/_common.py``'s module docstring
    and CHANGELOG [0.5.0].

    Also detects the degenerate (zero-variance) case, via
    ``zscore_iqr_masks``.

    Covers ``outliers="nan"`` and ``"drop"``. ``outliers="clip"`` also
    reaches ESD-recovered points, but not through this mask; see
    ``_esd_masking_repair``.
    """
    z_mask, iqr_mask, _, _ = zscore_iqr_masks(values, z_thresh)
    combined = z_mask | iqr_mask

    recovery = esd_masking_recovery(values, z_mask, iqr_mask)
    if recovery.esd_positions:
        recovered = np.zeros(len(values), dtype=bool)
        recovered[recovery.esd_positions] = True
        combined = combined | pd.Series(recovered, index=values.index)

    return combined


def _esd_masking_repair(
    out_col: pd.Series, values: pd.Series, z_thresh: float
) -> "tuple[pd.Series, pd.Index]":
    """
    Repair the specific rows ESD recovered under masking by setting them to
    NaN (for the ``missing`` imputation step to fill), on top of whatever
    the ordinary z-band/IQR-fence clip (``clip_bounds``) already did to
    ``out_col``. Returns ``(out_col, recovered_idx)``; ``recovered_idx`` is
    empty unless ``esd_masking_recovery`` reports masking.

    Why NaN and not a clip: an earlier version clipped these rows to a band
    derived from ``_generalized_esd``. That band does not actually exist in
    general: Rosner's test is a sequential procedure whose stopping point is
    a property of the whole removal sequence, not a per-point threshold, so
    no single band is guaranteed to lie outside every point the test flags.
    Adversarial simulation (n=200, sweeping contamination fraction and
    magnitude across 59 seeds) found 24 of 951 masking-suspected cases where
    the invented band left an ESD-recovered point unchanged. NaN has no such
    gap: it always changes the cell, matching what "nan"/"drop" repair (and
    ``scan()`` itself) already report.

    Why a separate step rather than folding into the ordinary clip mask: an
    ESD-recovered point is, by construction, already inside the z-band/IQR
    intersection ``clip_bounds`` computes, so nothing in the ordinary clip
    pass would touch it without naming it explicitly.

    Restricted to positions the IQR rule did not already flag (the only rule
    of z-score/IQR that can be non-empty here, since ``esd_masking_recovery``
    only runs when ``n_zscore == 0``). Points already flagged by IQR keep
    their ``clip_bounds`` target instead of being re-NaN-ed. Regression-tested:
    without this exclusion, a point flagged by both IQR and ESD ended up NaN
    instead of clipped.
    """
    z_mask, iqr_mask, _, _ = zscore_iqr_masks(values, z_thresh)
    recovery = esd_masking_recovery(values, z_mask, iqr_mask)
    if not recovery.esd_positions:
        return out_col, values.index[:0]

    already_flagged = (z_mask | iqr_mask).to_numpy()
    recovered_positions = [p for p in recovery.esd_positions if not already_flagged[p]]
    if not recovered_positions:
        return out_col, values.index[:0]

    recovered_idx = values.index[recovered_positions]
    # .loc restricts the write to exactly these rows; every other cell in
    # out_col is left exactly as it was.
    out_col = out_col.copy()
    out_col.loc[recovered_idx] = np.nan
    return out_col, recovered_idx


def _impute(series: pd.Series, method: str, datetime_index: bool) -> pd.Series:
    if method == "interpolate":
        how = "time" if datetime_index else "linear"
        return series.interpolate(method=how, limit_direction="both")
    if method == "ffill":
        return series.ffill()
    if method == "bfill":
        return series.bfill()
    return series


def _resolve_detector_settings(report):
    """
    Resolve the four detector thresholds/windows and the missing-data
    handling mode that ``apply_fixes``/``affected_cells`` need to recompute
    exactly the masks ``scan()`` used, from ``report.metadata``.

    An explicit value the caller passed to ``scan()`` always wins; only a
    genuinely unset (``None``) entry falls back to the domain-derived preset,
    the same "is None, not falsy" precedence the detectors themselves use, so
    a deliberate ``0`` is honoured rather than treated as unset.

    Falls back to the domain-only preset via ``.get(..., None)`` for a report
    built from an older or hand-constructed metadata dict missing these keys.
    """
    domain = report.metadata.get("domain")

    z_thresh = report.metadata.get("zscore_threshold")
    if z_thresh is None:
        z_thresh = zscore_preset(domain)

    window = report.metadata.get("stuck_window")
    if window is None:
        window = stuck_window_preset(domain)

    spike_thresh = report.metadata.get("spike_threshold")
    if spike_thresh is None:
        spike_thresh = spike_threshold_preset(domain)

    spike_window = report.metadata.get("spike_window")
    if spike_window is None:
        spike_window = SPIKE_WINDOW

    handle_missing = report.metadata.get("handle_missing") or "strict"

    return z_thresh, window, spike_thresh, spike_window, handle_missing


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
        (ANO003). "clip" winsorizes flagged points to the detection bounds:
        global IQR/z bounds for ANO002, the local rolling band for ANO003.
        Exception: a point ANO002 flagged only because ESD detected z-score
        masking sits inside the ordinary IQR/z bounds by construction, so no
        clip target exists for it; it is NaN-ed instead and left to the
        ``missing`` imputation step (see ``_esd_masking_repair``). "nan"
        sets flagged points to NaN for the imputation step. "drop" is an
        alias for "nan": rows are never deleted (that would break the time
        index). Default "clip".
    stuck : {"nan", None}
        "nan" replaces flagged stuck runs with NaN. Default "nan".
    leakage : {"drop", None}
        "drop" removes columns flagged by the leakage module. Off by default:
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

    # polars.DataFrame has neither .copy() nor .index, so convert here the
    # same way validate_dataframe does at the scan() boundary. Internals
    # stay pandas, and so does the return value.
    if _is_polars(df):
        df = _polars_to_pandas(df, time_col)

    # Resolve time_col the same way scan()'s validate_dataframe does, so a
    # caller who used scan(df, time_col=...) and now calls
    # report.apply_fixes(df) gets a correctly time-ordered repair, not one
    # computed on whatever row order the raw time_col column happened to be
    # in. Restored to the caller's original shape before returning.
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
    # detector, but `report` only carries the resulting Issues, not that
    # sorted frame. If the caller's `df` has a valid DatetimeIndex that is
    # out of chronological order, every mask computed below could find
    # nothing at all, repairing zero cells while report.last_fixes and the
    # caller both believe the data was cleaned. Restore chronological order
    # via position (not `.sort_index()`) so a duplicate timestamp, already
    # its own CRITICAL PRF004 finding, does not turn the reordering into an
    # ambiguous label-based reindex; restore the caller's original row order
    # before returning.
    datetime_index = isinstance(df.index, pd.DatetimeIndex)
    restore_positions: Optional[np.ndarray] = None
    if datetime_index:
        sort_positions = np.argsort(df.index.values, kind="mergesort")
        if not np.array_equal(sort_positions, np.arange(len(df))):
            restore_positions = np.empty_like(sort_positions)
            restore_positions[sort_positions] = np.arange(len(sort_positions))
            df = df.iloc[sort_positions]

    out = df.copy()
    z_thresh, stuck_window, spike_thresh, spike_window, handle_missing = (
        _resolve_detector_settings(report)
    )
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

    # 1. Leakage: drop flagged columns (opt-in only; never the target).
    if leakage == "drop":
        for col in report.leaky_columns():
            if col in out.columns and col != protected:
                out = out.drop(columns=col)
                log.append(
                    {"column": col, "action": "drop_column", "cells_changed": "-"}
                )

    # 2. Outliers: clip to bounds, or NaN-out for imputation.
    #
    # Every mask/bounds computation below reads from `df[col]` (the pristine,
    # pre-repair column), never `out[col]`, so a later step always detects
    # against the same input the original audit scored, regardless of what
    # an earlier step already clipped or NaN-ed. A column can carry more than
    # one finding (e.g. ANO002 and ANO003), and reading `out[col]` would let
    # an earlier step's edits silently change what a later step finds.
    if outliers is not None:
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

                # Second, targeted pass: ESD-recovered points are, by
                # construction, already inside [lower, upper], so the clip
                # above cannot move them. NaN-ed instead, restricted to
                # exactly those rows, and left for the `missing` imputation
                # step below. No-op when masking wasn't suspected.
                repaired, recovered_idx = _esd_masking_repair(
                    out[col], values, z_thresh
                )
                if len(recovered_idx) > 0:
                    already_nan = int(out.loc[recovered_idx, col].isna().sum())
                    newly = int(len(recovered_idx) - already_nan)
                    out[col] = repaired
                    nan_filled_cols.add(col)
                    log.append(
                        {
                            "column": col,
                            "action": "esd_masked_outliers_to_nan",
                            "cells_changed": newly,
                            "already_nan": already_nan,
                        }
                    )
            else:  # "nan" / "drop"
                mask = _outlier_mask(values, z_thresh)
                idx = mask[mask].index
                if len(idx) == 0:
                    continue
                # cells_changed counts only cells this step flips to NaN for
                # the first time; already_nan records cells this detector's
                # mask covers but an earlier step had already NaN-ed, so the
                # log still shows this detector fired even though nothing
                # changed. Always logged when the mask fires, even at 0
                # cells changed.
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
        for col in spike_cols:
            if col not in out.columns or not pd.api.types.is_numeric_dtype(out[col]):
                continue
            # Mirror audit_contextual_anomalies exactly: handle_missing
            # bridges a single-row gap before ANO003 looks at the series, so
            # the mask recomputed here must start from the same bridged
            # view, not a plain dropna().
            series = df[col]
            if handle_missing == "interpolate":
                series = series.interpolate(method="linear", limit=1)
            values = series.dropna()
            mask, lower, upper = spike_bounds(values, spike_window, spike_thresh)
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

    # 3. Stuck values: replace flagged runs with NaN.
    if stuck == "nan":
        for col in stuck_cols:
            if col not in out.columns or not pd.api.types.is_numeric_dtype(out[col]):
                continue
            mask, _ = stuck_run_mask(df[col], stuck_window)
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

    # 3b. Infinite values: always converted to NaN, then imputed with
    #     everything else if `missing` is enabled. Unconditional, since
    #     there is no reading of an infinity under which keeping it is
    #     correct: it poisons the mean/std of the column and makes
    #     scikit-learn raise at fit time. If `missing=None`, the cell is
    #     left as NaN.
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

    # 4. Imputation: fill flagged-missing columns plus anything we NaN-ed above.
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

    Each entity is repaired as its own independent time series using the
    single-series path, with a report view narrowed to that entity's issues,
    then written back by position. A panel index has the same timestamp once
    per entity, so a label-based ``.loc`` assignment would scatter one
    entity's repairs across all of them.

    Leaky-column drops are applied once to the whole frame rather than per
    entity, since a column either exists in the feature matrix or it does not.
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
                    {"column": col, "action": "drop_column", "cells_changed": "-"}
                )

    groups = out[group_col].to_numpy()
    payload_cols = [c for c in out.columns if c != group_col]

    # Rows with a null entity id are never scanned per-entity (see PNL004 in
    # tsauditor.panel), so there is no entity-specific report view to repair
    # them from. Left untouched, and logged once.
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
    zscore_threshold: Optional[float] = None,
    stuck_window: Optional[int] = None,
    spike_threshold: Optional[float] = None,
    spike_window: Optional[int] = None,
    handle_missing: str = "strict",
    missing: Optional[str] = "interpolate",
    outliers: Optional[str] = "clip",
    stuck: Optional[str] = "nan",
    leakage: Optional[str] = None,
    verbose: bool = False,
):
    """
    One-shot scan-and-repair. Scans ``df`` and returns ``(clean_df, report)``.

    A convenience wrapper over ``scan()`` + ``GuardReport.apply_fixes()``. It
    always returns both the repaired copy and the report, so the audit trail
    (``report.last_fixes``, ``report.leaky_columns()``, the full issue list)
    is never discarded.

    The input ``df`` is never modified; ``clean_df`` is an independent copy.
    Pass ``target=`` so the label column is protected from every repair.

    ``available_at=``, ``constraints=``, ``group_col=``, ``zscore_threshold=``,
    ``stuck_window=``, ``spike_threshold=``, ``spike_window=``, and
    ``handle_missing=`` are passed straight through to ``scan()``; each
    enables or tunes a check that would otherwise only be reachable by
    calling ``scan()`` and ``apply_fixes()`` separately. ``apply_fixes``
    itself needs no separate argument for these: it reads them back off
    ``report.metadata``, which ``scan()`` populates.

    Parameters
    ----------
    df : pd.DataFrame
        Passed through to ``scan``.
    target : str | None
        Passed through to ``scan``.
    time_col : str | None
        Passed through to ``scan``.
    domain : str | None
        Passed through to ``scan``.
    available_at : dict | None
        Passed through to ``scan``; required for LEK004 (as-of leakage) to run.
    constraints : dict | None
        Passed through to ``scan``; required for VAL001/VAL002.
    group_col : str | None
        Passed through to ``scan``; makes this a one-shot call for panel
        (long-format, multi-entity) data.
    zscore_threshold : float | None
        Passed through to ``scan``.
    stuck_window : int | None
        Passed through to ``scan``.
    spike_threshold : float | None
        Passed through to ``scan``.
    spike_window : int | None
        Passed through to ``scan``.
    handle_missing : str
        Passed through to ``scan``.
    missing : str | None
        Passed through to ``apply_fixes``.
    outliers : str | None
        Passed through to ``apply_fixes``.
    stuck : str | None
        Passed through to ``apply_fixes``.
    leakage : str | None
        Passed through to ``apply_fixes``.
    verbose : bool
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
        zscore_threshold=zscore_threshold,
        stuck_window=stuck_window,
        spike_threshold=spike_threshold,
        spike_window=spike_window,
        handle_missing=handle_missing,
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
        console.print("[bold]apply_fixes: changes applied[/bold]")
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
    issues,
    df: pd.DataFrame,
    z_thresh: float,
    window: int,
    spike_thresh: float,
    spike_window: int = SPIKE_WINDOW,
    handle_missing: str = "strict",
) -> int:
    """Affected-cell count for one series (single entity or non-panel), given
    only its own Issues. Factored out of affected_cells so the panel path
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
            # isinf is False for NaN, so a column flagged for both does not
            # double-count.
            mask |= pd.Series(
                np.isinf(s.to_numpy(dtype=float, copy=False)), index=s.index
            )
        values = s.dropna()
        if "ANO002" in codes and len(values):
            om = _outlier_mask(values, z_thresh)
            mask.loc[om[om].index] = True
        if "ANO003" in codes and len(values):
            # Same handle_missing-aware bridging as apply_fixes.
            spike_series = s
            if handle_missing == "interpolate":
                spike_series = spike_series.interpolate(method="linear", limit=1)
            spike_values = spike_series.dropna()
            if len(spike_values):
                sm, _, _ = spike_bounds(spike_values, spike_window, spike_thresh)
                mask.loc[sm[sm].index] = True
        if "ANO001" in codes:
            km, _ = stuck_run_mask(s, window)
            mask |= km.fillna(False)
        total += int(mask.sum())
    return total


def affected_cells(report, df: pd.DataFrame) -> int:
    """
    Count distinct data cells implicated by detected quality issues (missing,
    point outliers, contextual spikes, stuck runs). Leakage is excluded: a
    leaky column is a modeling risk, not a corrupt cell. Cells flagged by
    more than one detector in the same column are counted once.

    Panel-aware. When the report came from ``scan(group_col=...)``, every
    quality detector ran on one entity at a time, so this recomputes each
    mask per entity too, on only that entity's own Issues, rather than on the
    raw interleaved ``df`` where one entity's scale could dilute or exaggerate
    another's outliers.
    """
    z_thresh, window, spike_thresh, spike_window, handle_missing = (
        _resolve_detector_settings(report)
    )

    group_col = report.metadata.get("group_col")
    if group_col is None or group_col not in df.columns:
        return _affected_cells_single(
            report.all_issues,
            df,
            z_thresh,
            window,
            spike_thresh,
            spike_window,
            handle_missing,
        )

    total = 0
    groups = df[group_col].to_numpy()
    for key in pd.unique(groups):
        if pd.isna(key):
            # PNL004: rows with a null entity id are never scanned per-entity.
            continue
        sub = df[groups == key]
        issues = [i for i in report.all_issues if i.group == str(key)]
        total += _affected_cells_single(
            issues, sub, z_thresh, window, spike_thresh, spike_window, handle_missing
        )
    return total


def health_score(report, df: pd.DataFrame) -> float:
    """
    Data Health Score: percentage of numeric data cells not implicated by any
    quality issue. ``100 * (1 - affected_cells / total_cells)``, rounded to
    one decimal. Returns 100.0 when there are no numeric cells to assess.
    """
    numeric_cols = df.select_dtypes(include="number").shape[1]
    total = len(df) * numeric_cols
    if total == 0:
        return 100.0
    return round(100.0 * (1 - affected_cells(report, df) / total), 1)
