# Panel Data

**Panel data**, also called long-format or multi-entity data, is many time series stacked in one frame, distinguished by an entity column:

```
             ticker   price      ret  direction
2024-01-01   AAA     100.13   -0.002        0.0
2024-01-01   BBB     149.34    0.004        1.0
2024-01-01   CCC     199.64   -0.001        0.0
2024-01-02   AAA      99.99    0.001        1.0
2024-01-02   BBB     150.02    0.005        1.0
```

500 stocks × 3 years. 200 sensors × 6 months. 50 stores × 2 years of sales. This is one of the most common shapes real time-series data arrives in.

Pass `group_col=` and each entity is audited as its own independent time series:

```python
report = tsa.scan(panel, target="direction", group_col="ticker")
```

---

## Why you must not skip `group_col`

A panel scanned without `group_col` is not merely less accurate. It is meaningless, and in one case it used to crash.

Consider the frame above. Every timestamp appears once per entity, so tsauditor sees:

- **Duplicate timestamps everywhere.** PRF004 fires (CRITICAL) on data that is perfectly well formed.
- **A rolling window spanning several companies.** ANO003's 21-point local window covers AAA at 100, BBB at 150, CCC at 200, then AAA again. Every point looks like a spike relative to that "neighbourhood", because the neighbourhood is four different companies.
- **Nonsense gap analysis.** The median gap between consecutive rows is zero, so the frequency is inferred as `"sub-daily"` for daily data.
- **Interleaved NaN runs.** Each entity's first `ret` is NaN; stacked, these look like a scattered missingness pattern that belongs to no one.

Only the leakage checks partly survive, and only by luck, pooled AUC happens to stay 1.0 when the leak holds within every entity. Change the target definition per entity and that collapses too.

Here is the same panel scanned both ways:

| | without `group_col` | with `group_col="ticker"` |
|---|---|---|
| PRF004 duplicate timestamps | **fires** (spurious) | silent |
| Inferred frequency | `"sub-daily"` (wrong) | `"daily"` |
| LEK001 on `ret` | 1 finding | **5 findings, one per entity** |
| Rolling-window checks | span 5 companies | correct |

---

## What tsauditor tells you if you forget

PRF004 detects the *shape* of panel duplication, every timestamp repeating a uniform number of times, and says so:

```
Duplicate timestamps detected in the index. Chronological alignment broken.
Every timestamp repeats 5-5 times, which is the shape of panel (long-format)
data rather than a duplication bug. If these rows are separate entities, pass
group_col='<your entity column>' so each entity is audited as its own time
series.
```

The evidence carries this as a flag you can test programmatically:

```python
report.filter(code="PRF004")[0].evidence["looks_like_panel"]   # True
```

A single repeated timestamp among otherwise unique ones is still reported as an ordinary duplication bug, `looks_like_panel` is `False` and no hint is added.

---

## How it works

Partition, scan, merge. That is the whole design:

```python
for entity, sub in df.groupby(group_col):
    for issue in run_all_the_normal_checks(sub.drop(columns=group_col)):
        issue.group = entity
```

**No detector knows what a panel is.** `audit_equivalence` sees exactly what it always sees: one DataFrame, one time series. This is deliberate, panel support adds no branching to the detection logic, so it cannot introduce subtle inconsistencies between panel and non-panel results.

Three consequences worth knowing:

- The entity column is **dropped** before the detectors run, so it is never audited as a feature.
- `frequency` in the metadata is re-inferred from a **single entity**, since the interleaved index is meaningless.
- Every issue gains a `group` attribute. It is `None` for single-series scans and for panel-level findings.

---

## Reading the output: prevalence

A 500-entity panel with 20 columns can generate tens of thousands of issues. Listing them is useless. So `summary()` switches to a **prevalence** view for panels:

```python
report.summary()
```

```
Dataset
  Rows       : 1000
  Columns    : 4
  Time range : 2024-01-01 → 2024-10-04
  Frequency  : daily
  Entities   : 5 (grouped by 'ticker')

Critical: 5  Warnings: 14  Info: 0

  Severity      Code       Column               Entities               %   Examples
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  CRITICAL      LEK001     ret                  5/5               100.0%   AAA, BBB, CCC, DDD, EEE
  WARNING       ANO001     direction            4/5                80.0%   AAA, CCC, DDD, EEE
  WARNING       ANO002     ret                  3/5                60.0%   BBB, DDD, EEE
  WARNING       ANO003     ret                  2/5                40.0%   DDD, EEE
  WARNING       LEK003     price                2/5                40.0%   AAA, CCC
  WARNING       ANO002     price                1/5                20.0%   DDD
  WARNING       ANO003     price                1/5                20.0%   CCC
  WARNING       LEK002     price                1/5                20.0%   CCC
```

**The percentage is the diagnosis.**

| Reach | What it usually means |
| ----- | --------------------- |
| **100%** | A **pipeline bug**. Every entity cannot independently develop the same defect. Something in your feature construction is wrong for everyone. |
| **20–80%** | A **subset problem**. Some entities share a property: a data vendor, a listing venue, a sensor model: that the others do not. |
| **A few percent** | **Isolated**. Individual entities worth inspecting. Often genuine, often benign. |

In the table above, `LEK001` on `ret` at 100% is exactly the first case: `ret` defines `direction` by construction, for every ticker. No amount of per-entity investigation would help; the target definition is the problem.

`ANO001` on `direction` at 80% is the second case, and it is a *false positive*, a binary label consists of long runs of identical values, which is what stuck-value detection looks for. See [Anomaly Detectors](Detectors-Anomaly#limitations-and-false-positives).

### Programmatic access

```python
report.prevalence()          # list[dict], the table above
report.groups()              # every entity scanned
report.is_panel              # True

report.groups_affected(code="LEK001", column="ret")
# ['AAA', 'BBB', 'CCC', 'DDD', 'EEE']

report.filter(group="AAA")   # everything wrong with one entity
report.filter(code="LEK001", column="ret")
```

For display, hand `prevalence()` to pandas:

```python
import pandas as pd
pd.DataFrame(report.prevalence())
```

Each row has: `code`, `module`, `severity`, `column`, `n_groups`, `total_groups`, `pct`, `n_issues`, `example_groups`.

`to_json()` gains a `panel` block containing `group_col`, `n_groups`, and the full prevalence table; individual issues carry their `group`.

---

## Panel-only checks

Four checks exist that are meaningless for a single series.

### PNL001: Ragged panel
**WARNING**, dataset-level.

Entities do not share a common time index. AAA has 200 days, BBB has 150, CCC has 100.

*Evidence:* `n_groups`, `n_timestamps`, `min_coverage`, `max_coverage`, `n_complete_groups`, `worst_groups`, `group_col`

This matters because it is invisible per-entity, each series looks fine on its own, but it silently breaks every cross-sectional operation. A market-wide average computed at time *t* covers a different set of entities than the one at *t+1*. A pivot to wide format produces NaNs whose meaning ("not listed yet" vs "data missing") is unrecoverable.

Raggedness is often legitimate: companies IPO and delist, sensors are installed and retired. PNL001 is a WARNING, not an error. It tells you the shape of the problem so you can decide between reindexing onto the full timestamp set, restricting to complete entities, or handling the imbalance explicitly.

### PNL003: Entity too short to audit
**INFO**, dataset-level.

Some entities have fewer than 30 rows, below the `min_obs` floor the leakage checks and the ADF test need.

*Evidence:* `n_short_groups`, `n_groups`, `min_rows`, `shortest_groups`, `group_col`

The point of this check is to stop you misreading silence as health. A 12-row entity produces no LEK001 finding, but that is because the check *declined to score it*, not because it is clean. Without PNL003 those entities look identical to genuinely healthy ones in the prevalence table.

### PNL004: Rows with a null entity id
**WARNING**, dataset-level.

Some rows have a null (`NaN`/`None`) value in `group_col`. `groupby(group_col)` drops null keys by default, there is no entity identity to assign coverage, short-history, or leakage findings to, so these rows are excluded from *every* check `scan()` runs, panel-level and per-entity alike. `apply_fixes()` leaves them untouched for the same reason: there is no single entity's distribution to repair them from.

*Evidence:* `n_null_rows`, `n_total_rows`, `pct_null`, `group_col`

This is the sharpest version of the "silence is not health" problem in this module: a null-id row produces exactly zero findings, in either direction, from anything. Before PNL004 existed this happened invisibly. If you see it, either backfill the entity id for those rows or drop them explicitly, do not treat their clean-looking report as a clean result.

### PNL002: Cross-sectional lookahead
**WARNING**, dataset-level. Needs both `group_col=` and `target=`.

This is the panel-native leak. A cross-sectional feature, a rank, z-score, decile, or sector-neutralised value computed *across entities at one timestamp*, is legitimate:

```python
panel["xs_rank"] = panel.groupby("date")["ret"].rank(pct=True)          # fine
```

Shift it by one period and it is a leak:

```python
panel["xs_rank"] = panel.groupby("ticker")["xs_rank"].shift(-1)        # LEAK
```

Every row now carries tomorrow's cross-sectional rank.

*Evidence:* `metric`, `lag`, `observed_cs_corr`, `expected_from_cs_persistence`, `excess`, `excess_threshold`, `contemporaneous_cs_corr`, `n_entities`, `group_col`

#### Why the per-entity checks aren't enough

They do detect this, but only when idiosyncratic variation is large relative to any common factor. A cross-sectional rank measures *relative* position. Once a market factor dominates absolute outcomes, relative position decouples from each entity's own result, and the within-entity signal collapses:

| common/idio ratio | LEK002 | LEK003 | PNL002 |
| ----------------- | ------ | ------ | ------ |
| 0 | 100% of entities | 100% | detected |
| 5 | 95% | 95% | detected |
| **25** | **32.5%** | **22.5%** | **detected** |
| **100** | **22.5%** | **12.5%** | **detected** |

Real equity markets sit in the region where this matters, market factors routinely explain most of individual daily return variance.

**And the failure is worse than a plain miss.** At a 25:1 ratio, LEK002 flagged the *legitimate* `xs_rank` in 11 of 40 entities and the *leak* in 13 of 40. That is essentially no discriminating power. Worse, the prevalence table then reports the leak at 32%, which reads as "isolated, a few odd tickers" when it is in fact present in every one of them.

PNL002 flags the leak and leaves `xs_rank` alone, at every ratio.

#### Calibration

A genuine cross-sectional alpha factor produces the same *shape* of signal as a leak, separated only by magnitude, so this is a WARNING, not a CRITICAL. Realistic factor rank-ICs are 0.02–0.08. Tested factors up to IC 0.15 are not flagged; 0.30 and above are.

#### Gates

Returns nothing rather than a noisy answer when there are fewer than **20 co-present entities** per timestamp or fewer than **30 scored timestamps**. A cross-sectional correlation over a handful of entities is mostly noise.

Reproduce the numbers above with `python docs/proposals/pnl002_evidence.py`.

---

## Complete example

```python
import pandas as pd, numpy as np, tsauditor as tsa

# Build a 5-ticker panel where `ret` defines `direction`: a planted leakdates = pd.date_range("2024-01-01", periods=200, freq="B")
parts = []
for i, ticker in enumerate(["AAA", "BBB", "CCC", "DDD", "EEE"]):
    rng = np.random.default_rng(i)
    price = 100 + 50 * i + np.cumsum(rng.normal(0, 1, 200))
    ret = pd.Series(price).pct_change().to_numpy()
    parts.append(pd.DataFrame({
        "ticker": ticker,
        "price": price,
        "ret": ret,
        "direction": (ret > 0).astype(float),
    }, index=dates))

panel = pd.concat(parts).sort_index()

report = tsa.scan(
    panel,
    target="direction",
    group_col="ticker",
    domain="finance",
    run_stationarity=False,      # see the performance note below
)

report.summary()
print(report.groups_affected(code="LEK001", column="ret"))
```

```
['AAA', 'BBB', 'CCC', 'DDD', 'EEE']
```

---

## Repairing a panel

`apply_fixes()` and `fix()` are panel-aware: when the report came from a `group_col=` scan, each entity is repaired as its own independent time series.

```python
report = tsa.scan(panel, group_col="ticker", domain="finance", run_stationarity=False)
clean = report.apply_fixes(panel)
```

**This matters more than it sounds.** Repairing an interleaved panel as one series carries values across entity boundaries. Measured on a two-entity panel where one series sits near 10 and the other near 1000, a gap in the low series was filled with **~1000**, the other entity's values, silently. Partitioning first fills it with ~10, correctly.

Three guarantees on top of the usual ones:

- **Row order, index and shape are preserved exactly.** Write-back is positional, not label-based, a panel index repeats each timestamp once per entity, so a `.loc` assignment would scatter one entity's repairs across all of them.
- **The change log is tagged per entity.** Each `report.last_fixes` entry gains a `group` key.
- **Leaky-column drops are frame-wide.** A column either exists in the feature matrix or it does not, so `leakage="drop"` removes it once rather than once per entity.

```python
for entry in report.last_fixes:
    print(entry)
```

```
{'column': 'price', 'action': 'impute_interpolate', 'cells_changed': 6, 'group': 'AAA'}
```

---

## Performance

Panel scanning runs the full check suite **once per entity**. Cost scales linearly with entity count, and the ADF stationarity test (PRF003) dominates.

**Set `run_stationarity=False` for panels** unless you specifically need it:

```python
report = tsa.scan(panel, group_col="ticker", run_stationarity=False)
```

Non-stationarity is a per-column modeling note rated INFO. Across 500 entities it produces 500× that note while contributing nothing to the health score. Run it once on a representative entity instead.

---

## Limitations

**Health score is computed on the pooled frame.** It counts affected cells across the whole panel and does not break down per entity.

**No parallelism.** Entities are scanned sequentially.

**Nested grouping is unsupported.** `group_col` takes a single column. For a two-level entity key, combine them first:

```python
panel["entity"] = panel["region"] + "|" + panel["store"]
report = tsa.scan(panel, group_col="entity")
```
