"""
tsauditor.scanner
-----------------
The main entry point. scan() orchestrates all audit modules and
assembles a GuardReport.
"""

from __future__ import annotations

import warnings
from typing import Optional

import pandas as pd

from tsauditor.report.summary import GuardReport, Issue, CRITICAL, WARNING
from tsauditor.utils.validation import validate_dataframe, infer_frequency

# Rough heuristic, not a benchmarked cutoff: above this many groups, a
# sequential group_col scan is slow enough that most callers would rather
# know n_jobs=-1 exists than find out by waiting.
_MANY_GROUPS_WARNING_THRESHOLD = 100


def scan(
    df: pd.DataFrame,
    target: Optional[str] = None,
    time_col: Optional[str] = None,
    domain: Optional[str] = None,
    available_at: Optional[dict] = None,
    constraints: Optional[dict] = None,
    group_col: Optional[str] = None,
    n_jobs: int = 1,
    chunk_size: Optional[int] = None,
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
    group_col : Optional[str]
        Entity column for panel (long-format) data, e.g. ``"ticker"``. When
        given, the frame is partitioned by this column and **each entity is
        audited as its own independent time series**; every resulting Issue is
        tagged with its entity via ``Issue.group``.

        Without it a panel is treated as one interleaved series, which makes the
        structural, anomaly and rolling checks meaningless — a rolling window
        would span several entities at once, and every timestamp would look
        duplicated.

        Panel-level structure checks (PNL001, PNL003) also run, and
        ``report.prevalence()`` summarises how widely each finding occurs across
        entities.
    n_jobs : int
        Number of parallel workers used to audit groups when ``group_col`` is
        given. Default ``1`` (sequential, matches prior behaviour exactly).
        Set to ``-1`` to use all available cores. Ignored when ``group_col`` is
        None, a single (non-panel) scan is always run in-process.
        Parallelism is dispatched in chunks (see ``chunk_size``), not one task
        per group, because per-task overhead (pickling, worker dispatch) can
        dominate for datasets with many small groups, a real, measured
        failure mode with short per-entity series (tens of rows), not a
        hypothetical one.
    chunk_size : Optional[int]
        Number of groups audited per dispatched parallel task. Default None,
        which auto-selects a chunk size from ``len(groups)`` and the actual
        worker count (``joblib.cpu_count()`` when ``n_jobs=-1``) so each
        worker gets a handful of tasks rather than one group each. Only
        relevant when ``n_jobs != 1`` and ``group_col`` is given.
    run_profiler : bool
        Run structural profiling checks. Default True.
    run_anomaly : bool
        Run anomaly detection checks. Default True.
    run_leakage : bool
        Run leakage detection checks. Default True.
        Silently skipped if target is None.
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
    >>> # Panel data, parallelized across all cores:
    >>> report = tsa.scan(df, group_col="ticker", domain="finance", n_jobs=-1)  # doctest: +SKIP
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
    }

    report = GuardReport(metadata=metadata)

    options = _ScanOptions(
        target=target,
        domain=domain,
        available_at=available_at,
        constraints=constraints,
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
    metadata["n_jobs"] = n_jobs
    # frequency was inferred from the interleaved index, which is meaningless
    # for a panel; re-infer it from a single entity instead.
    if groups:
        metadata["frequency"] = infer_frequency(groups[0][1].index)

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

    # ── Per-entity checks: sequential (n_jobs=1, default) or parallel ───────
    if n_jobs == 1:
        if len(groups) > _MANY_GROUPS_WARNING_THRESHOLD:
            warnings.warn(
                f"Scanning {len(groups)} groups sequentially (n_jobs=1, the "
                f"default). For a panel this large, pass n_jobs=-1 to audit "
                f"entities in parallel across all available cores. "
                f"({_MANY_GROUPS_WARNING_THRESHOLD} groups is a rough "
                f"heuristic, not a hard cutoff, most machines will see a "
                f"real speedup well before that point.)",
                UserWarning,
                stacklevel=2,
            )
        for key, sub in groups:
            for issue in _audit_one_group(key, sub, group_col, options):
                _append_issue(report, issue)
    else:
        from joblib import Parallel, delayed

        effective_chunk_size = chunk_size or _auto_chunk_size(len(groups), n_jobs)
        chunks = [
            groups[i : i + effective_chunk_size]
            for i in range(0, len(groups), effective_chunk_size)
        ]
        chunk_results = Parallel(n_jobs=n_jobs)(
            delayed(_audit_group_chunk)(chunk, group_col, options) for chunk in chunks
        )
        for chunk_issues in chunk_results:
            for issue in chunk_issues:
                _append_issue(report, issue)

    return report


def _auto_chunk_size(n_groups: int, n_jobs: int) -> int:
    """
    Pick a chunk size so each worker gets a handful of dispatched tasks
    rather than exactly one, amortizes per-task overhead (pickling, worker
    dispatch) which otherwise dominates for datasets with many small groups.
    """
    if n_jobs > 0:
        effective_jobs = n_jobs
    else:
        # n_jobs=-1 (or any other negative joblib convention): resolve to the
        # actual worker count joblib itself would use, not a guess.
        from joblib import cpu_count

        effective_jobs = cpu_count()
    # aim for ~4 chunks per worker
    target_n_chunks = max(effective_jobs * 4, 1)
    return max(1, n_groups // target_n_chunks)


def _audit_one_group(
    key, sub: pd.DataFrame, group_col: str, options: "_ScanOptions"
) -> list:
    """Audit a single entity's partition, tagging every resulting Issue with
    its group key. Pure function of its arguments, safe to call from a
    worker process."""
    sub = sub.drop(columns=[group_col])
    issues = list(_run_checks(sub, options))
    for issue in issues:
        issue.group = str(key)
    return issues


def _audit_group_chunk(chunk, group_col: str, options: "_ScanOptions") -> list:
    """Audit several groups within one dispatched task, this is the unit
    that actually gets sent to a joblib worker, not a single group, to keep
    per-task overhead from dominating on many-small-groups panels."""
    all_issues = []
    for key, sub in chunk:
        all_issues.extend(_audit_one_group(key, sub, group_col, options))
    return all_issues


class _ScanOptions:
    """Plain container for the per-partition check settings."""

    __slots__ = (
        "target",
        "domain",
        "available_at",
        "constraints",
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

        yield from audit_point_anomalies(df, domain=opts.domain)
        yield from audit_contextual_anomalies(df, domain=opts.domain)

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

        bounds = opts.constraints.get("bounds")
        relations = opts.constraints.get("relations")
        if bounds is None and relations is None:
            # flat {col: spec} mapping treated as bounds
            bounds = opts.constraints
        yield from audit_validity(df, bounds=bounds, relations=relations)


def _append_issue(report: GuardReport, issue: Issue) -> None:
    """Route an Issue to the correct severity bucket in the report."""
    if issue.severity == CRITICAL:
        report.critical.append(issue)
    elif issue.severity == WARNING:
        report.warnings.append(issue)
    else:
        report.info.append(issue)
