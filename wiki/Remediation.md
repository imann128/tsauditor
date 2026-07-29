# Remediation

`tsauditor` is advisory by default: it detects and suggests, but changes nothing. This page covers what happens when you ask it to act.

Three entry points, all in `tsauditor/remediate.py`:

| Function | What it does |
| -------- | ------------ |
| `report.apply_fixes(df, ...)` | Repair a frame using an existing report |
| `tsa.fix(df, ...)` | Scan and repair in one call |
| `report.health_score(df)` | Score the fraction of cells not implicated by a quality issue |

---

## Panel data

If the report came from a `scan(..., group_col=...)` call, repairs are applied **per entity** automatically. Nothing extra to pass:

```python
report = tsa.scan(panel, group_col="ticker", run_stationarity=False)
clean = report.apply_fixes(panel)
```

This is not cosmetic. Repairing an interleaved panel as a single series carries values across entity boundaries — measured on a two-entity panel, a gap in a series near 10 was filled with **~1000** from the other entity. Row order, index and shape are preserved exactly (write-back is positional, since a panel index repeats each timestamp per entity), each `last_fixes` entry gains a `group` key, and `leakage="drop"` removes a column frame-wide rather than once per entity.

See [Panel Data](Panel-Data#repairing-a-panel).

---

## Four guarantees

These hold for every repair path and are worth relying on.

**1. Nothing is mutated.** `apply_fixes` calls `df.copy()` and returns the copy. The frame you passed in is byte-for-byte unchanged. You can always diff the result against your source.

**2. Only flagged columns are touched.** A column that produced no issue is returned exactly as it arrived. Repairs are driven by the report, not applied blindly.

**3. Rows are never deleted.** "Dropping" an outlier means setting it to NaN so the imputation step can fill it. Deleting a row would break the index's uniform frequency and re-trigger the very gap detectors that just passed.

**4. The target label is never repaired.** `apply_fixes` reads `report.metadata["target"]` and excludes that column from every operation. This matters more than it sounds: a binary 0/1 label trips ANO001 constantly, since it consists of long runs of identical values. Interpolating it would turn your labels into fractions.

---

## What gets repaired, and what does not

| Code | Repaired? | Action |
| ---- | --------- | ------ |
| ANO002 point outliers | Yes | clip / NaN |
| ANO003 contextual spikes | Yes | clip to local band / NaN |
| ANO001 stuck values | Yes | NaN, then impute |
| PRF002, PRF006 missing | Yes | impute |
| PRF007 infinite values | **Yes, always** | NaN, then impute |
| LEK001–LEK004 leakage | **Opt-in only** | drop the whole column |
| PRF001, PRF004, PRF005 index problems | **No** | — |
| PRF003 non-stationarity | **No** | — |
| VAL001, VAL002 validity | **No** | — |

The gaps are deliberate.

**Index problems are not repaired** because there is no safe default. Deduplicating timestamps requires knowing whether to keep the first, keep the last, or aggregate — and aggregating requires knowing whether to sum, average, or take the last. Resampling to fill gaps invents rows that never existed. These are decisions only you can make.

**Non-stationarity is not repaired** because it is not a defect. Differencing a price series changes what the column *means*; `tsauditor` will not silently redefine your features.

**Validity issues are not repaired** because the correct remedy depends entirely on the cause. A sentiment score of 1.8 on a [−1, 1] scale might warrant clipping to 1.0, or it might mean someone forgot to normalize — in which case clipping destroys evidence of a bug that affects every row.

---

## `apply_fixes`

```python
clean = report.apply_fixes(
    df,
    missing="interpolate",
    outliers="clip",
    stuck="nan",
    leakage=None,
    verbose=False,
)
```

| Parameter | Accepts | Default | Effect |
| --------- | ------- | ------- | ------ |
| `missing` | `"interpolate"`, `"ffill"`, `"bfill"`, `None` | `"interpolate"` | How to fill NaNs |
| `outliers` | `"clip"`, `"nan"`, `"drop"`, `None` | `"clip"` | Handles **both** ANO002 and ANO003 |
| `stuck` | `"nan"`, `None` | `"nan"` | Whether to void stuck runs |
| `leakage` | `"drop"`, `None` | `None` | Whether to remove leaky columns |
| `verbose` | `bool` | `False` | Print a change summary |

An invalid value raises `ValueError` immediately, before anything is modified.

Two naming quirks to be aware of. `"drop"` for outliers is an **alias for `"nan"`** — it does not drop anything, because rows are never deleted. And `outliers=` controls contextual spikes (ANO003) as well as point outliers (ANO002), despite the name.

### Execution order

The order is not arbitrary — steps 2 and 3 feed step 4.

**1. Leakage** (only if `leakage="drop"`). Every column in `report.leaky_columns()` is removed, except the target.

**2. Outliers and spikes** (only if `outliers` is not `None`).

For ANO002 columns, `"clip"` winsorizes to a band computed as the intersection of the z-score band and the IQR fence:

```
lower = max(mean - z_thresh × std,  Q1 - 1.5 × IQR)
upper = min(mean + z_thresh × std,  Q3 + 1.5 × IQR)
```

Taking the intersection means clipping pulls in exactly the points that either rule flagged, and leaves every point that neither rule flagged untouched.

For ANO003 columns, clipping targets the **local** band `local_mean ± threshold × local_std`, because a contextual spike is a local anomaly. Clipping it to a global bound would be meaningless — the whole point of ANO003 is that the value is globally ordinary.

With `"nan"`, flagged cells are set to NaN and their columns are queued for imputation.

**3. Stuck values** (only if `stuck="nan"`). Flagged runs become NaN and are queued for imputation.

**3b. Infinite values** (PRF007). Flagged infinities become NaN and are queued for imputation.

This step is the one exception to "only if you asked for it": it runs unconditionally, with no parameter to disable it. Every other repair is opt-in because there is a legitimate reading under which the original value should be kept — an outlier may be a real market event, a stuck run may be a genuine trading halt. There is no such reading for an infinity. It is the residue of a division by zero, an overflow, or a log of a non-positive number, and keeping it makes the column's mean `inf`, its standard deviation `NaN`, and `scikit-learn` raise at `fit` time.

If `missing=None`, the cells are left as NaN rather than imputed. NaN is honest about the value being unknown; `inf` is a false claim about its size.

**Why it runs before imputation, not after.** `interpolate` filling a NaN that neighbours an infinity carries the infinity into the gap. On real data — five features built from the OGDC series by ordinary feature engineering — 19 infinities across 3 columns became **35** after `fix()` in 0.3.0, because `log_ret` had a NaN run sitting against an infinity. Converting first makes this impossible.

**4. Imputation** (only if `missing` is not `None`). Fills columns flagged PRF002/PRF006 **plus** every column NaN-ed in steps 2, 3 and 3b.

`"interpolate"` uses `method="time"` on a `DatetimeIndex` and `method="linear"` otherwise, with `limit_direction="both"` so leading and trailing NaNs are also filled.

Note the interaction: with `outliers="nan"`, an outlier is voided in step 2 and then filled in step 4 with a value interpolated from its neighbours. With `outliers="clip"`, it is pulled to the boundary instead. Clipping preserves the information that the point was extreme; NaN-then-interpolate discards it.

### The change log

Every operation appends to `report.last_fixes`:

```python
for entry in report.last_fixes:
    print(entry)
```

```
{'column': 'price', 'action': 'clip_outliers', 'cells_changed': 18, 'bounds': (89.25536216923865, 108.1799349119554)}
{'column': 'price', 'action': 'clip_spikes', 'cells_changed': 2}
{'column': 'price', 'action': 'stuck_to_nan', 'cells_changed': 21}
{'column': 'price', 'action': 'impute_interpolate', 'cells_changed': 27}
```

Possible `action` values: `drop_column`, `clip_outliers`, `outliers_to_nan`, `clip_spikes`, `spikes_to_nan`, `stuck_to_nan`, `impute_ffill`, `impute_bfill`, `impute_interpolate`.

`last_fixes` is overwritten on each call, not appended to.

---

## `fix` — the one-shot wrapper

```python
clean, report = tsa.fix(df, target="Direction", domain="finance")
```

Exactly equivalent to `scan()` followed by `apply_fixes()`. It always returns **both** values, so the audit trail cannot be silently discarded — you keep the record of what changed and why.

`missing`, `outliers`, `stuck`, `leakage`, and `verbose` pass straight through.

One limitation: `fix()` does not accept `available_at` or `constraints`, so LEK004 and the validity checks never run through this path. If you need them, call `scan()` and `apply_fixes()` separately.

---

## Health score

```python
report.health_score(df)
```

```
92.2
```

**Definition:** the percentage of numeric cells *not* implicated by a quality issue.

```
100 × (1 − affected_cells / (n_rows × n_numeric_columns))
```

rounded to one decimal place. Returns `100.0` if there are no numeric columns.

**Which codes count.** Only these six:

```
PRF002, PRF006  (missing)
PRF007          (infinite values)
ANO001          (stuck)
ANO002          (point outliers)
ANO003          (contextual spikes)
```

In practice PRF007 rarely moves the score, because ANO003 already marks those positions: an infinity's deviation from its local mean is infinite, so it always clears the spike threshold. Measured on the OGDC-derived features above, the score is 85.2 with or without PRF007 counted. It is included for correct attribution rather than to change the number.

**Leakage is deliberately excluded.** A leaky column is a modeling risk, not a corrupt cell. Its values are perfectly valid data. Including leakage would conflate two different kinds of problem and make the score meaningless — you would not know whether a low score meant "dirty data" or "one bad feature."

Non-stationarity (PRF003), index problems (PRF001/004/005), and validity (VAL001/002) are also excluded.

**Cells are counted once.** A cell flagged by both ANO002 and ANO003 counts as one affected cell, not two. The masks are unioned per column before counting.

**It re-scans.** `health_score` runs a fresh scan of whatever frame you pass, rather than reusing the report's existing issues. This is what makes a genuine before/after comparison possible:

```python
report = tsa.scan(df, domain="finance", run_stationarity=False)
clean  = report.apply_fixes(df)

print("before:", report.health_score(df))
print("after :", report.health_score(clean))
```

```
before: 92.2
after : 99.0
```

Reusing the original report's issues would have given a stale answer for `clean`. The re-scan skips leakage and ADF checks, since neither affects the score.

This does mean `health_score` costs a full scan. Do not call it in a loop.

---

## Complete worked example

```python
import pandas as pd, numpy as np, tsauditor as tsa

rng = np.random.default_rng(3)
n   = 200
idx = pd.date_range("2024-01-01", periods=n, freq="D")

price = 100 + np.cumsum(rng.normal(0, 1, n))
price[40]      = 400.0            # outlier
price[80:86]   = price[79]        # stuck run
price[120:126] = np.nan           # outage

df = pd.DataFrame(
    {"price": price, "volume": rng.integers(1000, 5000, n).astype(float)},
    index=idx,
)

report = tsa.scan(df, domain="finance", run_stationarity=False)
print("health before:", report.health_score(df))
for i in report.all_issues:
    print(" ", i.severity.upper(), i.code, i.column)

clean = report.apply_fixes(df)
print()
for f in report.last_fixes:
    print(" ", f["column"], f["action"], f["cells_changed"])

print()
print("health after      :", report.health_score(clean))
print("original NaNs     :", int(df["price"].isna().sum()))
print("repaired NaNs     :", int(clean["price"].isna().sum()))
```

```
health before: 92.2
  WARNING ANO002 price
  WARNING ANO001 price
  WARNING ANO003 price
  WARNING PRF002 price

  price clip_outliers 18
  price clip_spikes 2
  price stuck_to_nan 21
  price impute_interpolate 27

health after      : 99.0
original NaNs     : 6
repaired NaNs     : 0
```

---

## Read this before trusting the defaults

**`clip_outliers` changed 18 cells, not 1.** Only one outlier was planted. The other 17 come from the IQR fence, which on this data flags a chunk of the legitimate tail. This is not a bug — it is the documented behaviour of the 1.5×IQR rule on real distributions — but it means **`apply_fixes` modifies substantially more data than the number of faults you are aware of.**

Always read `last_fixes` before accepting a repair. If `cells_changed` is far larger than you expected, inspect before proceeding. See the [Anomaly Detectors](Detectors-Anomaly#limitations-and-false-positives) page for why the IQR rule behaves this way.

**`stuck_to_nan` changed 21 cells, not 6.** The planted stuck run was 6 long, but a random walk produces other short repeats, and the run-length rule catches them all.

**Imputed values are estimates, not data.** After `apply_fixes`, some of your values are interpolations. Downstream models treat them as real observations and cannot tell the difference. For a forecasting model this can matter a great deal.

**The health score improves partly by construction.** Clipping outliers to the detection boundary guarantees they no longer exceed the boundary. A score of 99.0 means "few cells now trip the detectors," not "this data is now correct."

**A quieter, more conservative starting point:**

```python
clean = report.apply_fixes(
    df,
    outliers=None,           # leave extreme values alone; review them yourself
    stuck="nan",
    missing="interpolate",
)
```

This repairs only the failures that are unambiguous — frozen sensors and gaps — and leaves the judgement calls to you.

---

## Interaction with domain presets

`apply_fixes` resolves its thresholds from `report.metadata["domain"]`, so the domain you scanned with automatically governs repair:

| `domain` | z-score | stuck window | spike threshold |
| -------- | ------- | ------------ | --------------- |
| `"finance"` | 5.0 | 5 | 4.0 |
| `"sensor"` | 3.5 | 3 | 3.0 |
| `None` | 4.0 | 5 | 3.5 |

These duplicate the detector modules' values. That duplication is a real maintenance hazard — if one side changed and the other did not, `apply_fixes` would silently modify cells the report never flagged. `tests/test_fix.py` asserts the repaired cell count matches the detector's own evidence, which pins them together. Keep that test passing if you touch either side.

See [Domain Presets](Domain-Presets) for the full picture.
