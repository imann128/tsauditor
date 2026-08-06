# API Reference

Complete reference for everything `tsauditor` exposes publicly.

**Looking for how a check decides?** This page documents signatures and return types. The reasoning behind each detector lives on its own page: [Profiler](Detectors-Profiler), [Anomaly](Detectors-Anomaly), [Leakage](Detectors-Leakage), [Validity](Detectors-Validity).

---

## The public surface

```python
import tsauditor as tsa

tsa.scan                  # audit a DataFrame           -> GuardReport
tsa.fix                   # scan and repair in one call -> (DataFrame, GuardReport)
tsa.adapters.to_timesfm   # audit, repair, format       -> np.ndarray
tsa.GuardReport           # the report type
tsa.Issue                 # a single finding
tsa.__version__           # "0.5.0"
```

The detector functions are not re-exported at the top level. Import them from their modules when you want to run one in isolation:

```python
from tsauditor.profiler import (
    audit_frequency, audit_missing, audit_non_finite, audit_stationarity,
)
from tsauditor.anomaly  import audit_point_anomalies, audit_contextual_anomalies
from tsauditor.leakage  import (
    audit_equivalence, audit_correlation_leakage, audit_temporal_leakage,
    audit_asof_leakage, audit_combination_leakage,
)
from tsauditor.validity import audit_validity
from tsauditor.panel    import audit_panel_structure, audit_cross_sectional_leakage
```

---

## `tsauditor.scan()`

The single entry point for all audits.

```python
tsauditor.scan(
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
    run_profiler: bool = True,
    run_anomaly: bool = True,
    run_leakage: bool = True,
    run_stationarity: bool = True,
) -> GuardReport
```

### Parameters

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `df` | `pd.DataFrame` | required | Input data. Must have a `DatetimeIndex`, or pass `time_col`. A polars DataFrame is accepted **with** `time_col` (needs the `[polars]` extra). |
| `target` | `str` or `None` | `None` | Target/label column. **Required for LEK001–LEK003, LEK005 and PNL002**; they are silently skipped without it. |
| `time_col` | `str` or `None` | `None` | Datetime column to set as the index. |
| `domain` | `str` or `None` | `None` | Threshold preset: `"finance"`, `"sensor"`, or `None`. See [Domain Presets](Domain-Presets). |
| `available_at` | `dict` or `None` | `None` | Point-in-time availability for LEK004. Maps column → per-row publish timestamps (`pd.Series` on `df.index`) or a fixed `pd.Timedelta` lag. Only listed columns are checked. |
| `constraints` | `dict` or `None` | `None` | Domain-validity rules (VAL001/VAL002). `{"bounds": {...}, "relations": [...]}`. A flat `{col: {...}}` mapping is treated as bounds, disambiguated from the nested form by *shape*, not by key name, so a column literally named `"bounds"` or `"relations"` is handled correctly. See [Validity Detectors](Detectors-Validity#the-flat-shorthand). |
| `group_col` | `str` or `None` | `None` | Entity column for **panel (long-format) data**. Each entity is audited as its own time series and every issue is tagged via `Issue.group`. Also enables PNL001/PNL003/PNL004. See [Panel Data](Panel-Data). |
| `zscore_threshold` | `float` or `None` | `None` | Absolute z-score above which a point is flagged (ANO002). `None` derives it from `domain`. |
| `stuck_window` | `int` or `None` | `None` | A run longer than this is flagged as stuck (ANO001). `None` derives it from `domain`. |
| `spike_threshold` | `float` or `None` | `None` | Local z-score above which a point is flagged as a contextual spike (ANO003). `None` derives it from `domain`. |
| `spike_window` | `int` or `None` | `None` | Width of ANO003's local context window. `None` defaults to 21. |
| `handle_missing` | `str` | `"strict"` | `"interpolate"` fills single-row gaps before ANO003's spike check runs; anything else leaves NaNs in place. ANO001's stuck-run detection bridges a single-row gap either way: a lone missing reading inside an otherwise-flat run is still a stuck run regardless of this setting. |
| `run_profiler` | `bool` | `True` | Run structural checks (PRF). |
| `run_anomaly` | `bool` | `True` | Run anomaly checks (ANO). |
| `run_leakage` | `bool` | `True` | Run leakage checks (LEK). Target-based checks still need `target`; LEK004 runs whenever `available_at` is given. |
| `run_stationarity` | `bool` | `True` | Run the ADF test (PRF003), **the runtime hot spot**. Set `False` for a much faster sweep. |

`zscore_threshold`, `stuck_window`, `spike_threshold`, `spike_window`, and `handle_missing` all default to values that reproduce the previous, always-domain-derived behaviour exactly, so passing none of them changes nothing for existing callers. Before these were added, tuning an individual anomaly parameter meant bypassing `scan()` and calling `audit_point_anomalies`/`audit_contextual_anomalies` directly.

### Returns

A [`GuardReport`](#guardreport).

### Raises

| Exception | When |
| --------- | ---- |
| `TypeError` | `df` is not a `pd.DataFrame` |
| `ValueError` | invalid `domain`; `target` or `time_col` not found; empty `df`; numeric index that cannot be safely coerced; a declared `available_at` / `constraints` column is missing or non-numeric; an `available_at` Series whose timezone awareness does not match `df.index`'s |

### Notes on behaviour

**A numeric index is refused, not coerced.** `pd.to_datetime([0, 1, 2])` would silently produce three timestamps a few nanoseconds apart in 1970, corrupting every gap and frequency result without any visible failure. `scan()` raises instead. Pass `time_col=` or set a `DatetimeIndex` yourself.

**Your DataFrame is copied at validation.** Nothing downstream can mutate your input.

**Rows are sorted ascending by time** before any check runs.

**Toggle order.** `run_stationarity` only has an effect when `run_profiler` is also `True`.

### Examples

```python
import pandas as pd
import tsauditor as tsa

report = tsa.scan(df, target="Direction", domain="finance")

# Faster sweep: skip the expensive ADF test
report = tsa.scan(df, target="Direction", run_stationarity=False)

# As-of leakage: CPI is published ~30 days after its reference date
report = tsa.scan(df, available_at={"cpi": pd.Timedelta(days=30)})

# Validity: strictly-positive spread and an uncrossed book
report = tsa.scan(df, constraints={
    "bounds":    {"spread": {"min": 0, "min_exclusive": True}},
    "relations": [("bid", "ask")],
})

# polars input: time_col is mandatory
report = tsa.scan(polars_df, time_col="date", target="Direction")
```

---

## `tsauditor.fix()`

One-shot scan-and-repair. Returns **both** the cleaned copy and the report, so the audit trail is never discarded. The input is never modified.

```python
clean_df, report = tsauditor.fix(
    df,
    target=None,
    time_col=None,
    domain=None,
    available_at=None,
    constraints=None,
    group_col=None,
    missing="interpolate",
    outliers="clip",
    stuck="nan",
    leakage=None,
    verbose=False,
)
```

Equivalent to `scan()` then `report.apply_fixes()`. `df`, `target`, `time_col`, `domain`, `available_at`, `constraints`, and `group_col` pass through to `scan`; the rest pass through to `apply_fixes`.

**The target label is never repaired.** Pass `target=` so it is protected.

`available_at=` and `constraints=` let LEK004 (as-of leakage) and VAL001/VAL002 (validity) run as part of a one-shot repair, not just through a separate `scan()` + `apply_fixes()` call. `group_col=` does the same for panel data, one call instead of `scan(group_col=...)` + `apply_fixes()`.

### Returns

`(clean_df, report)`: a `tuple[pd.DataFrame, GuardReport]`.

```python
clean, report = tsa.fix(df, target="Direction", domain="finance")

report.last_fixes        # exactly what changed
report.leaky_columns()   # what it flagged
```

→ See [Remediation](Remediation) for what each repair option does.

---

## `tsauditor.adapters.to_timesfm()`

Audit, repair, and format a single series into a finite `float32` array for Google TimesFM. Adds no `timesfm` dependency.

```python
array = tsauditor.adapters.to_timesfm(
    df,
    target_col,
    *,
    domain=None,
    context_len=1024,
    min_context=32,
    return_report=False,
)
```

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `df` | `pd.DataFrame` | required | Raw input containing the series |
| `target_col` | `str` | required | Column to extract and forecast |
| `domain` | `str` or `None` | `None` | Passed to the audit |
| `context_len` | `int` | `1024` | Keep only the most recent N points |
| `min_context` | `int` | `32` | Raise below this length |
| `return_report` | `bool` | `False` | If `True`, return `(array, report)` |

Everything after `*` is keyword-only.

### Returns

A 1-D `np.float32` array, or `(array, report)` when `return_report=True`.

### Raises

| Exception | When |
| --------- | ---- |
| `KeyError` | `target_col` is not in `df` |
| `TypeError` | `target_col` is not numeric (categorical or string) |
| `ValueError` | The repaired series still contains non-finite values, or has fewer than `min_context` points |

### Four things to know

**`target_col` must be numeric.** A non-numeric column raises `TypeError` naming the column and its dtype, rather than a confusing raw `ValueError` from the numpy conversion further downstream.

**`target_col` is cleaned, not protected.** `fix()` normally shields the target from repair; you never clean a label. But here the series *is* what you want cleaned, so it is repaired as an ordinary column. Internally the adapter calls `fix(df, domain=domain)` with no `target=`.

**The result is verified finite.** Repair does not always eliminate every NaN (a lone unflagged missing value, for instance). Rather than letting a NaN reach the model and fail tokenization confusingly, the adapter raises.

**`context_len` and `min_context` are your knobs, not TimesFM constants.** TimesFM 2.5 accepts contexts up to 16k and needs no frequency indicator. The default 1024 is deliberately conservative; raise it if you want more history.

```python
array, report = tsa.adapters.to_timesfm(
    df, target_col="price", domain="finance", return_report=True,
)
print(report.last_fixes)   # what was imputed before you trust the forecast
```

Imputed values are estimates. The model treats them as real history.

---

## `GuardReport`

The structured output of `scan()`.

```python
from tsauditor import GuardReport
```

### Attributes

| Attribute | Type | Description |
| --------- | ---- | ----------- |
| `critical` | `List[Issue]` | Issues that block modeling |
| `warnings` | `List[Issue]` | Issues worth reviewing |
| `info` | `List[Issue]` | Informational findings |
| `metadata` | `Dict[str, Any]` | rows, columns, time range, frequency, target, domain |
| `last_fixes` | `List[Dict]` | Change log from the most recent `apply_fixes` / `fix`. Empty until one runs; **overwritten**, not appended. |

`metadata` looks like:

```python
{
    'rows': 1537,
    'columns': 24,
    'time_start': '2020-01-22',
    'time_end': '2026-04-03',
    'frequency': 'daily',
    'target': 'Direction',
    'domain': 'finance',
}
```

`frequency` is one of `"sub-daily"`, `"daily"`, `"weekly"`, `"monthly"`, `"irregular"`, `"unknown"`.

### Properties

**`all_issues`** → `List[Issue]`: every issue, sorted by severity then module.

### Methods

**`filter(code=None, module=None, severity=None, column=None, group=None)`** → `List[Issue]`

Filters combine with AND. Omitted filters match everything.

```python
report.filter(code="LEK001")
report.filter(module="leakage", severity="critical")
report.filter(column="ret")
report.filter(group="AAPL")            # panel scans only
```

Valid `module` values: `"profiler"`, `"anomaly"`, `"leakage"`, `"validity"`, `"panel"`.
Valid `severity` values: `"critical"`, `"warning"`, `"info"`.

### Panel accessors

Only meaningful after `scan(..., group_col=...)`. → [Panel Data](Panel-Data)

**`is_panel`** (property) → `bool`: whether this report came from a panel scan.

**`groups()`** → `List[str]`: every entity scanned, sorted, including entities with zero issues. Empty for single-series scans.

**`groups_affected(code=None, column=None, severity=None)`** → `List[str]`

Which entities a given finding hit.

```python
report.groups_affected(code="LEK001", column="ret")
# ['AAA', 'BBB', 'CCC', 'DDD', 'EEE']
```

**`prevalence()`** → `List[Dict]`

How widely each finding occurs across entities: the headline output of a panel scan. One row per `(code, column)`, sorted by severity then reach.

Keys: `code`, `module`, `severity`, `column`, `n_groups`, `total_groups`, `pct`, `n_issues`, `example_groups`.

```python
import pandas as pd
pd.DataFrame(report.prevalence())
```

A finding at 100% is systemic; suspect the pipeline, not the entities.

Works on single-series reports too, with `n_groups` and `total_groups` set to `None`.

**`leaky_columns()`** → `List[str]`

Sorted, deduplicated list of columns flagged by the **leakage** module, plus PNL002 (cross-sectional lookahead). PNL002 is tagged `module="panel"`, not `"leakage"`, since it's emitted alongside the panel-structure checks (PNL001/PNL003/PNL004); it's carved in by code so those structural, columnless findings stay excluded. Validity issues never appear here, even though VAL002 is CRITICAL: a data error is not a leak.

**`suggestions()`** → `List[Dict]`

One dict per issue with keys `code`, `column`, `severity`, `suggestion`, ordered by severity.

**`apply_fixes(df, missing="interpolate", outliers="clip", stuck="nan", leakage=None, verbose=False)`** → `pd.DataFrame`

Repaired **copy** of `df`, fixing only flagged columns. Never touches the target. Records `last_fixes`.

- `missing`: `"interpolate"` / `"ffill"` / `"bfill"` / `None`
- `outliers`: `"clip"` / `"nan"` / `"drop"` (alias for `"nan"`) / `None`: covers ANO002 **and** ANO003
- `stuck`: `"nan"` / `None`
- `leakage`: `"drop"` / `None`

→ [Remediation](Remediation)

**`health_score(df)`** → `float`

Percentage of numeric cells not implicated by a quality issue (PRF002, PRF006, PRF007, ANO001, ANO002, ANO003). Leakage, stationarity, index problems, and validity are excluded.

**Re-scans `df`**, so calling it on a `fix()` output gives a true "after" score. This costs a full scan; do not call it in a loop. On a panel scan, affected cells are recomputed per entity, not pooled across the whole interleaved frame → [Panel Data](Panel-Data#health-score-is-per-entity-not-pooled).

**`summary()`** → `None`: prints a rich CLI table plus suggested actions.

**`to_json(path, df=None, fixed_df=None)`** → `None`

JSON export. Passing `df` adds a `health` block; additionally passing `fixed_df` adds `score_after`. For a panel scan, gains a `panel` block (`group_col`, `n_groups`, the full prevalence table).

**`to_pdf(path, df=None, fixed_df=None, title=None)`**

Formal, text-selectable PDF. Requires the `[pdf]` extra, else `ImportError`. For a panel scan, the issues table is replaced by a prevalence table (one row per finding, with the fraction of entities it hit) instead of a raw per-issue dump.

**`to_dict()`** → `Dict[str, Any]`: metadata, issues, and counts (`critical`, `warnings`, and `info`). `to_json()` is built from this same dict, so the two cannot drift out of sync with each other.

---

## `Issue`

A single quality finding.

| Attribute | Type | Description |
| --------- | ---- | ----------- |
| `module` | `str` | `"profiler"`, `"anomaly"`, `"leakage"`, or `"validity"` |
| `code` | `str` | e.g. `"LEK001"`; see [Issue Code Reference](Issue-code-reference) |
| `severity` | `str` | `"critical"`, `"warning"`, `"info"` |
| `description` | `str` | Human-readable explanation |
| `column` | `str` or `None` | Affected column, or `None` for dataset-level findings |
| `evidence` | `Dict[str, Any]` | Supporting statistics; **keys vary by code** |
| `group` | `str` or `None` | Entity, for panel scans. `None` otherwise, and omitted entirely from `to_dict()` when `None`, so single-series JSON is unchanged. |

**`suggestion`** (property) → `str`: recommended action, derived from the code and filled from `evidence`.

**`to_dict()`** → `Dict`: all attributes plus the rendered `suggestion`.

```python
issue = report.filter(code="LEK001")[0]

issue.column       # 'ChangeP'
issue.evidence     # {'metric': 'auc', 'auc': 1.0, 'separation': 1.0, ...}
issue.suggestion   # "Remove or reconstruct column 'ChangeP': it near-..."
```

The `evidence` dict is where the reasoning lives. Its keys differ per code and are documented in full on each detector page.

`column` is `None` for the dataset-level index checks: PRF001, PRF004, PRF005.

---

## Severity constants

```python
from tsauditor.report.summary import CRITICAL, WARNING, INFO
```

```python
CRITICAL = "critical"
WARNING  = "warning"
INFO     = "info"
```

They are plain strings, so `issue.severity == "critical"` works too. Use the constants for readability.

---

## Detector functions

Each can be called directly on a DataFrame, bypassing `scan()`. All return `List[Issue]`. All except `audit_validity` require a `DatetimeIndex`.

| Function | Import from | Key parameters |
| -------- | ----------- | -------------- |
| `audit_frequency` | `tsauditor.profiler` | `domain` |
| `audit_missing` | `tsauditor.profiler` | `cluster_threshold`, `missing_rate_threshold`, `domain` |
| `audit_non_finite` | `tsauditor.profiler` | *(none; see [Profiler Detectors](Detectors-Profiler#audit_non_finite))* |
| `audit_stationarity` | `tsauditor.profiler` | `alpha`, `min_obs`, `max_lag` |
| `audit_point_anomalies` | `tsauditor.anomaly` | `zscore_threshold`, `domain` |
| `audit_contextual_anomalies` | `tsauditor.anomaly` | `stuck_window`, `spike_threshold`, `spike_window`, `domain`, `handle_missing` |
| `audit_equivalence` | `tsauditor.leakage` | `target`, `continuous_threshold`, `binary_threshold`, `min_obs` |
| `audit_correlation_leakage` | `tsauditor.leakage` | `target`, `max_lag`, `min_correlation`, `min_obs` |
| `audit_temporal_leakage` | `tsauditor.leakage` | `target`, `max_lag`, `excess_threshold`, `min_correlation`, `min_obs` |
| `audit_asof_leakage` | `tsauditor.leakage` | `available_at`, `min_violations` |
| `audit_combination_leakage` | `tsauditor.leakage` | `target`, `threshold`, `min_obs`, `max_features`, `max_reported`, `gate` |
| `audit_validity` | `tsauditor.validity` | `bounds`, `relations` |
| `audit_panel_structure` | `tsauditor.panel` | `group_col`, `min_rows` |
| `audit_cross_sectional_leakage` | `tsauditor.panel` | `group_col`, `target`, `max_lag`, `excess_threshold`, `min_entities` |

Two things `scan()` does that direct calls do not: it validates and sorts your DataFrame, and it never forwards custom thresholds, only `domain`. If you call a detector directly on an unsorted or unvalidated frame, results are undefined.

Also note that `audit_validity` takes `bounds` and `relations` as separate arguments; the combined `constraints=` dict is a `scan()`-level convenience.

→ Full behavioural documentation: [Profiler](Detectors-Profiler), [Anomaly](Detectors-Anomaly), [Leakage](Detectors-Leakage), [Validity](Detectors-Validity)
