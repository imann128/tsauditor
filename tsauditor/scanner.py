"""
tsauditor.scanner
-----------------
The main entry point. scan() orchestrates all audit modules and
assembles a GuardReport.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from tsauditor.report.summary import GuardReport, Issue, CRITICAL, WARNING
from tsauditor.utils.validation import validate_dataframe, infer_frequency


def scan(
    df: pd.DataFrame,
    target: Optional[str] = None,
    time_col: Optional[str] = None,
    domain: Optional[str] = None,
    available_at: Optional[dict] = None,
    constraints: Optional[dict] = None,
    group_col: Optional[str] = None,
    # Anomaly detector tuning — all None/"strict" by default, matching the
    # underlying audit_point_anomalies / audit_contextual_anomalies defaults
    # exactly, so passing none of these changes nothing for existing callers.
    zscore_threshold: Optional[float] = None,
    stuck_window: Optional[int] = None,
    spike_threshold: Optional[float] = None,
    spike_window: Optional[int] = None,
    handle_missing: str = "strict",
    # Fine-grained toggles — all enabled by default
    run_profiler: bool = True,
    run_anomaly: bool = True,
    run_leakage: bool = True,
    run_stationarity: bool = True,
) -> GuardReport:
    """
    Audit a time-series DataFrame for data quality issues.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame. Must have a DatetimeIndex or a datetime column
        specified via time_col.
    target : Optional[str]
        Name of the target/label column. Required for leakage detection.
        If None, leakage checks are skipped.
    time_col : Optional[str]
        Name of a datetime column to use as the index. If None, the
        DataFrame index must already be a DatetimeIndex.
    domain : Optional[str]
        Domain hint for domain-specific heuristics.
        One of: "finance", "sensor", None.
    available_at : Optional[dict]
        Point-in-time availability metadata for the as-of leakage check (LEK004),
        mapping column name -> per-row publish timestamps (a pd.Series indexed by
        df.index) or a fixed publication lag (a pd.Timedelta). Only the columns
        provided are checked. Omitted (default) skips the as-of check, which
        cannot be inferred from values alone.
    constraints : Optional[dict]
        Domain-validity rules (VAL001/VAL002). A dict with optional keys
        ``"bounds"`` (per-column min/max, e.g.
        ``{"spread": {"min": 0, "min_exclusive": True}}``) and ``"relations"``
        (ordered ``(low, high)`` column pairs that must satisfy ``low <= high``,
        e.g. ``[("bid", "ask")]``). A flat ``{col: {...}}`` mapping is treated as
        bounds. Omitted (default) skips validity checks.

        The nested and flat forms are told apart by *shape*, not by whether
        the keys ``"bounds"``/``"relations"`` are present: a nested
        ``"bounds"`` value is always a dict mapping every column to its own
        spec dict, and a nested ``"relations"`` value is always a list/tuple
        of pairs -- a flat per-column spec can never take either shape. This
        means a column that happens to be named ``"bounds"`` or
        ``"relations"`` is handled correctly either way, e.g.
        ``constraints={"spread": {...}, "relations": {"min": 0}}`` is
        recognised as flat bounds for two columns (one of them named
        "relations"), not misread as the nested form.
    group_col : Optional[str]
        Entity column for panel (long-format) data, e.g. ``"ticker"``. When
        given, the frame is partitioned by this column and **each entity is
        audited as its own independent time series**; every resulting Issue is
        tagged with its entity via ``Issue.group``.

        Without it a panel is treated as one interleaved series, which makes the
        structural, anomaly and rolling checks meaningless — a rolling window
        would span several entities at once, and every timestamp would look
        duplicated.

        Panel-level structure checks (PNL001 ragged coverage, PNL003 too-short
        entities, PNL004 null-entity rows) also run unconditionally -- they
        are not gated by any of the run_* toggles below -- and
        ``report.prevalence()`` summarises how widely each finding occurs
        across entities.
    zscore_threshold : Optional[float]
        Absolute z-score above which a point is flagged (ANO002). When None
        (the default), derived from ``domain``.
    stuck_window : Optional[int]
        A run longer than this is flagged as stuck (ANO001). When None (the
        default), derived from ``domain``.
    spike_threshold : Optional[float]
        Local z-score above which a point is flagged as a contextual spike
        (ANO003). When None (the default), derived from ``domain``.
    spike_window : Optional[int]
        Width of the local context window for ANO003. Defaults to 21.
    handle_missing : str
        "interpolate" fills single-row gaps before running ANO003's spike
        check; anything else (the default, "strict") leaves NaNs in place.
        ANO001's stuck-run detection bridges a single-row gap either way,
        since a lone missing reading inside an otherwise-flat run is still a
        stuck run regardless of how the caller wants the rest of the series
        handled.
    run_profiler : bool
        Run structural profiling checks. Default True.
    run_anomaly : bool
        Run anomaly detection checks. Default True.
    run_leakage : bool
        Run leakage detection checks. Default True.
        The target-based checks (LEK001/002/003/005, and PNL002 in panel
        mode) are silently skipped if target is None -- they have no target
        to compare against. LEK004 (as-of leakage) is the exception: it is
        target-independent and still runs whenever ``available_at`` is
        supplied, target or not.
    run_stationarity : bool
        Run the ADF stationarity test (PRF003). Default True. This is the most
        expensive check by far (statsmodels ADF dominates runtime); set False to
        skip it when you only need structural, anomaly and leakage checks.

    Returns
    -------
    GuardReport
        Structured report with critical issues, warnings, and info.

    Examples
    --------
    >>> import tsauditor as tsa
    >>> report = tsa.scan(df, target="Direction", domain="finance")  # doctest: +SKIP
    >>> report.summary()  # doctest: +SKIP
    >>> report.to_json("report.json")  # doctest: +SKIP
    """
    # ── Validate domain argument ──────────────────────────────────────────────
    valid_domains = {"finance", "sensor", None}
    if domain not in valid_domains:
        raise ValueError(f"domain must be one of {valid_domains}, got '{domain}'.")

    # ── Validate and normalize input ──────────────────────────────────────────
    df = validate_dataframe(df, target=target, time_col=time_col)

    if group_col is not None and group_col not in df.columns:
        raise ValueError(
            f"group_col='{group_col}' not found in DataFrame columns: "
            f"{list(df.columns)}"
        )
    if group_col is not None and group_col == target:
        raise ValueError(
            f"group_col and target are both '{group_col}'; the entity column "
            f"cannot also be the prediction target."
        )

    # ── Build metadata ────────────────────────────────────────────────────────
    metadata = {
        "rows": len(df),
        "columns": len(df.columns),
        "time_start": str(df.index.min().date()),
        "time_end": str(df.index.max().date()),
        "frequency": infer_frequency(df.index),
        "target": target,
        "domain": domain,
        # Recorded so downstream consumers of the report -- chiefly
        # apply_fixes()/fix() -- can resolve time_col the same way this
        # function just did, given only the report and the caller's
        # original (not-yet-indexed) df. Without this, apply_fixes has no
        # way to know a time_col was ever used at all.
        "time_col": time_col,
    }

    report = GuardReport(metadata=metadata)

    options = _ScanOptions(
        target=target,
        domain=domain,
        available_at=available_at,
        constraints=constraints,
        zscore_threshold=zscore_threshold,
        stuck_window=stuck_window,
        spike_threshold=spike_threshold,
        spike_window=spike_window,
        handle_missing=handle_missing,
        run_profiler=run_profiler,
        run_anomaly=run_anomaly,
        run_leakage=run_leakage,
        run_stationarity=run_stationarity,
    )

    if group_col is None:
        for issue in _run_checks(df, options):
            _append_issue(report, issue)
        return report

    # ── Panel mode ────────────────────────────────────────────────────────────
    # Each entity is an independent time series, so the ordinary checks run
    # unchanged on each partition and every issue is tagged with its entity.
    # The detectors never learn what a panel is.
    from tsauditor.panel import audit_cross_sectional_leakage, audit_panel_structure

    groups = list(df.groupby(group_col, sort=True))

    metadata["group_col"] = group_col
    metadata["n_groups"] = len(groups)
    metadata["groups"] = sorted(str(key) for key, _ in groups)
    # The raw interleaved index is meaningless for frequency inference:
    # consecutive rows alternate between different entities reporting on
    # nearly the same date, so consecutive diffs are near-zero regardless of
    # each entity's own cadence. Re-infer from the deduplicated union of
    # every entity's own timestamps instead -- not a single entity's, which
    # is fragile to exactly which entity sorts first: a panel of 20 clean
    # daily entities plus one alphabetically-first entity with a sparse,
    # irregular history (e.g. a recent listing) would report the whole
    # panel's frequency as "irregular" if only that one entity were used,
    # even though every other entity, and the panel as a whole, is daily.
    # The union is dominated by whichever cadence the majority of entities
    # actually share, so one ragged entity can't skew it.
    if groups:
        metadata["frequency"] = infer_frequency(
            pd.DatetimeIndex(df.index.unique()).sort_values()
        )

    for issue in audit_panel_structure(df, group_col=group_col):
        _append_issue(report, issue)

    # Cross-sectional lookahead (PNL002) is panel-level and target-based. It
    # exists because the per-entity checks below degrade badly when a common
    # factor dominates; see tsauditor/panel.py.
    if run_leakage and target is not None:
        for issue in audit_cross_sectional_leakage(
            df, group_col=group_col, target=target
        ):
            _append_issue(report, issue)

    for key, sub in groups:
        sub = sub.drop(columns=[group_col])
        for issue in _run_checks(sub, options):
            issue.group = str(key)
            _append_issue(report, issue)

    return report


class _ScanOptions:
    """Plain container for the per-partition check settings."""

    __slots__ = (
        "target",
        "domain",
        "available_at",
        "constraints",
        "zscore_threshold",
        "stuck_window",
        "spike_threshold",
        "spike_window",
        "handle_missing",
        "run_profiler",
        "run_anomaly",
        "run_leakage",
        "run_stationarity",
    )

    def __init__(self, **kwargs):
        for name in self.__slots__:
            setattr(self, name, kwargs[name])


def _run_checks(df: pd.DataFrame, opts: "_ScanOptions"):
    """
    Run every enabled check against one already-validated time series and yield
    the resulting Issues.

    This is the single-series pipeline. ``scan`` calls it once for an ordinary
    frame, or once per entity for a panel — which is what keeps panel support
    from leaking into the detectors themselves.
    """
    # ── Profiler ──────────────────────────────────────────────────────────────
    if opts.run_profiler:
        from tsauditor.profiler import (
            audit_frequency,
            audit_stationarity,
            audit_missing,
            audit_non_finite,
        )

        # audit_frequency is run once and its issues routed by severity.
        # (Previously it was called three times — once per bucket.)
        yield from audit_frequency(df, domain=opts.domain)

        # ADF is the heaviest check; allow opting out.
        if opts.run_stationarity:
            yield from audit_stationarity(df, domain=opts.domain)

        yield from audit_missing(df, domain=opts.domain)

        # PRF007. Runs unconditionally and takes no threshold: an infinity is
        # never a valid measurement, and every other detector discards it
        # silently, so nothing else in the pipeline would report it.
        yield from audit_non_finite(df)

    # ── Anomaly ───────────────────────────────────────────────────────────────
    if opts.run_anomaly:
        from tsauditor.anomaly import (
            audit_point_anomalies,
            audit_contextual_anomalies,
        )

        yield from audit_point_anomalies(
            df, zscore_threshold=opts.zscore_threshold, domain=opts.domain
        )
        yield from audit_contextual_anomalies(
            df,
            stuck_window=opts.stuck_window,
            spike_threshold=opts.spike_threshold,
            spike_window=opts.spike_window,
            domain=opts.domain,
            handle_missing=opts.handle_missing,
        )

    # ── Leakage ───────────────────────────────────────────────────────────────
    if opts.run_leakage and opts.target is not None:
        from tsauditor.leakage import (
            audit_equivalence,
            audit_correlation_leakage,
            audit_temporal_leakage,
        )

        from tsauditor.leakage.combination import audit_combination_leakage

        yield from audit_equivalence(df, target=opts.target, domain=opts.domain)
        yield from audit_correlation_leakage(df, target=opts.target, domain=opts.domain)
        yield from audit_temporal_leakage(df, target=opts.target, domain=opts.domain)
        yield from audit_combination_leakage(df, target=opts.target, domain=opts.domain)

    # As-of leakage is target-independent and only runs when the caller supplies
    # availability metadata (it cannot be inferred from values alone).
    if opts.run_leakage and opts.available_at:
        from tsauditor.leakage import audit_asof_leakage

        yield from audit_asof_leakage(df, available_at=opts.available_at)

    # Domain-validity rules only run when the caller declares them.
    if opts.constraints:
        from tsauditor.validity import audit_validity

        raw = opts.constraints

        # Two accepted shapes: the nested {"bounds": {...}, "relations": [...]}
        # form, or a flat {col: spec} shorthand treated entirely as bounds.
        # These used to be told apart by key presence alone (`.get("bounds")`
        # / `.get("relations")` both None => flat), which broke the moment a
        # real column was named "bounds" or "relations" -- e.g.
        # {"spread": {...}, "relations": {"min": 0}} (a flat dict bounding a
        # column literally called "relations") got misread as the nested
        # form with relations={"min": 0}, and audit_validity crashed trying
        # to unpack a dict key as a (low, high) pair.
        #
        # Distinguish structurally instead, since the two nested values have
        # shapes a flat per-column spec can never take: a nested "bounds"
        # value maps every column to its own spec dict (dict-of-dicts), and
        # spec values (min/max/min_exclusive/max_exclusive) are always
        # scalars/bools, never dicts themselves. A nested "relations" value
        # is always a list/tuple of pairs, never a dict. A flat spec can
        # collide with neither shape.
        has_bounds_key = (
            "bounds" in raw
            and isinstance(raw["bounds"], dict)
            and all(isinstance(v, dict) for v in raw["bounds"].values())
        )
        has_relations_key = "relations" in raw and isinstance(
            raw["relations"], (list, tuple)
        )

        if has_bounds_key or has_relations_key:
            bounds = raw.get("bounds") if has_bounds_key else None
            relations = raw.get("relations") if has_relations_key else None
        else:
            bounds = raw
            relations = None

        yield from audit_validity(df, bounds=bounds, relations=relations)


def _append_issue(report: GuardReport, issue: Issue) -> None:
    """Route an Issue to the correct severity bucket in the report."""
    if issue.severity == CRITICAL:
        report.critical.append(issue)
    elif issue.severity == WARNING:
        report.warnings.append(issue)
    else:
        report.info.append(issue)
