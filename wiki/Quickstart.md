# Quickstart

This page takes you from a raw DataFrame to a clean one in five steps. Every example below is runnable and every printed output on this page was produced by actually running the code.

---

## Step 0. What tsauditor needs from you

Two requirements, and one optional-but-recommended argument.

**Required: your rows must be ordered in time.** `tsauditor` reasons about *before* and *after*, so it needs a `DatetimeIndex`. Either set one yourself:

```python
df = df.set_index("date")           # where "date" holds datetimes
```

or let `scan()` do it by naming the column:

```python
report = tsa.scan(df, time_col="date")
```

`tsauditor` deliberately refuses to guess. If your index is a plain `RangeIndex` (`0, 1, 2, ...`), it raises an error rather than silently reinterpreting those integers as nanosecond timestamps near 1970, which would quietly corrupt every gap and frequency result.

**Recommended: name your target column.** The leakage checks compare each feature *against the target*, so without `target=` they cannot run at all and are silently skipped:

```python
report = tsa.scan(df, target="Direction")
```

Since leakage detection is the main reason to use this library, omitting `target=` throws away most of its value.

**Optional: set a domain.** `domain="finance"` or `domain="sensor"` adjusts the thresholds to match what "normal" means in that field. See [Domain Presets](Domain-Presets).

---

## Step 1. Scan

```python
import tsauditor as tsa

report = tsa.scan(df, target="Direction", domain="finance")
report.summary()
```

`scan()` returns a `GuardReport`, a structured Python object, not just printed text. Issues are bucketed into three severity levels:

```python
report.critical    # list[Issue], will corrupt training or evaluation
report.warnings    # list[Issue], worth reviewing, may be fine in context
report.info        # list[Issue], informational only
```

The single most useful line for a modeler:

```python
report.leaky_columns()
```

```
['ChangeP', 'Returns']
```

That is the shortlist of features to review or remove before training. `tsauditor` will not remove them for you, that is a modeling decision only you can make.

You can also filter programmatically:

```python
report.filter(code="LEK001")                       # one specific check
report.filter(module="leakage")                    # everything from one module
report.filter(severity="critical")                 # everything blocking
report.filter(module="leakage", severity="critical")  # filters combine (AND)
```

---

## Step 2. Read a finding

Every `Issue` carries the reasoning behind it, not just a verdict.

```python
issue = report.filter(code="LEK001")[0]

print(issue.column)       # which column
print(issue.description)  # plain-language explanation
print(issue.evidence)     # the numbers behind the decision
print(issue.suggestion)   # what to do about it
```

```
ChangeP

Feature 'ChangeP' near-deterministically reproduces target 'Direction'
(auc score=1.0000 >= 0.95 for binary target). Likely data leakage,
review before modeling.

{'metric': 'auc', 'auc': 1.0, 'separation': 1.0, 'threshold': 0.95,
 'target_type': 'binary', 'n_obs': 1537}

Remove or reconstruct column 'ChangeP': it near-deterministically reproduces
the target variable and will leak. Keep it only if you can confirm it is
genuinely available at prediction time.
```

The `evidence` dict is the important part. It tells you *why*: the check used AUC (because the target is binary), scored 1.0 against a threshold of 0.95, and had 1,537 observations to work with. An AUC of exactly 1.0 means the feature separates the two classes perfectly, there is no ambiguity here.

The keys inside `evidence` differ per issue code. Each is documented on the relevant detector page.

---

## Step 3. Repair (optional, always on a copy)

`tsauditor` is advisory by default. When you want it to act, `fix()` scans and repairs in one call:

```python
clean, report = tsa.fix(df, target="Direction", domain="finance")
```

Three guarantees:

1. **Your original `df` is never modified.** `clean` is an independent copy.
2. **The target label is never repaired.** Interpolating a 0/1 label into fractions would be wrong.
3. **Every change is logged.** `report.last_fixes` records exactly what happened.

Here is a complete, runnable example with deliberately broken data:

```python
import pandas as pd, numpy as np, tsauditor as tsa

rng = np.random.default_rng(3)
n = 200
idx = pd.date_range("2024-01-01", periods=n, freq="D")

price = 100 + np.cumsum(rng.normal(0, 1, n))
price[40]      = 400.0            # an outlier
price[80:86]   = price[79]        # a stuck run (sensor froze)
price[120:126] = np.nan           # an outage

df = pd.DataFrame(
    {"price": price, "volume": rng.integers(1000, 5000, n).astype(float)},
    index=idx,
)

report = tsa.scan(df, domain="finance", run_stationarity=False)
print("health before:", report.health_score(df))

for i in report.all_issues:
    print(" ", i.severity.upper(), i.code, i.column)
```

```
health before: 92.2
  WARNING ANO002 price
  WARNING ANO001 price
  WARNING ANO003 price
  WARNING PRF002 price
```

All three planted faults were found: the outlier (ANO002 and ANO003), the stuck run (ANO001), and the outage (PRF002). Now repair:

```python
clean = report.apply_fixes(df)

for f in report.last_fixes:
    print(f)

print("health after :", report.health_score(clean))
print("NaNs in original:", int(df["price"].isna().sum()))
print("NaNs in clean   :", int(clean["price"].isna().sum()))
```

```
{'column': 'price', 'action': 'clip_outliers', 'cells_changed': 18, 'bounds': (89.25536216923865, 108.1799349119554)}
{'column': 'price', 'action': 'clip_spikes', 'cells_changed': 2}
{'column': 'price', 'action': 'stuck_to_nan', 'cells_changed': 21}
{'column': 'price', 'action': 'impute_interpolate', 'cells_changed': 27}

health after : 99.0
NaNs in original: 6
NaNs in clean   : 0
```

Health rose from 92.2 to 99.0, and the original still has its six NaNs, untouched, as promised.

Note that `clip_outliers` changed 18 cells, not 1. The z-score and IQR rules together flag more points than just the one you planted; the IQR fence in particular is strict. This is worth understanding before trusting the defaults, see [Remediation](Remediation) for what each repair action does and how to tune it.

For fine-grained control:

```python
clean = report.apply_fixes(
    df,
    missing="interpolate",   # or "ffill", "bfill", None
    outliers="clip",         # or "nan", "drop", None
    stuck="nan",             # or None
    leakage=None,            # or "drop" to remove leaky columns entirely
)
```

---

## Step 4. Catch the subtler leaks

Two checks cannot be inferred from your data's values alone, so you must declare the rules. They are off by default.

**As-of leakage (LEK004)**, many series describe a period but are *published later*. CPI for January is not knowable on January 31st; it comes out in February. If the value sits at its reference date, every row before the real release date uses information that did not yet exist.

```python
import pandas as pd

# CPI is published roughly 30 days after the period it describes
report = tsa.scan(df, available_at={"cpi": pd.Timedelta(days=30)})
```

For real, ragged release schedules, pass a `pd.Series` of actual publication timestamps indexed by `df.index` instead of a fixed lag.

**Validity rules (VAL001, VAL002)**, values that are *definitionally* impossible, not merely surprising:

```python
report = tsa.scan(df, constraints={
    "bounds":    {"spread": {"min": 0, "min_exclusive": True}},  # spread must be > 0
    "relations": [("bid", "ask")],                               # bid must never exceed ask
})
```

`tsauditor` cannot guess that a column named `spread` must be positive, so you tell it once and it verifies every row.

---

## Step 5. Export, or feed a model

```python
# Machine-readable, with health score and before/after delta
report.to_json("report.json", df=df, fixed_df=clean)

# Formal, text-selectable PDF (needs the [pdf] extra)
report.to_pdf("report.pdf", df=df, fixed_df=clean)

# A forecast-ready, finite-checked float32 array for Google TimesFM
array = tsa.adapters.to_timesfm(df, target_col="price", domain="finance")
```

The TimesFM adapter never imports `timesfm`, it audits, repairs, verifies the result contains no NaN or infinity, and hands you a plain NumPy array. You own the model.

---

## A note on speed

The ADF stationarity test (PRF003) dominates the runtime of a scan, it fits many regressions per column. If you only need structural, anomaly, and leakage checks, turn it off:

```python
report = tsa.scan(df, target="Direction", run_stationarity=False)
```

Other toggles: `run_profiler`, `run_anomaly`, `run_leakage`, all `True` by default.

---

## Where to go next

- [How It Works](How-it-works), what `scan()` does internally, in order
- [Leakage Detectors](Detectors-Leakage), the four leakage checks, explained properly
- [OGDC Case Study](OGDC-case-study), the real-world leak this library was built for
- [API Reference](API-Reference), every parameter of every public function
