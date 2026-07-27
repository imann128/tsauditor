# Profiler Detectors

The profiler asks whether your data is **structurally sound** — before anyone looks at whether the values make sense. Three functions, six issue codes.

| Function | Codes | Module |
| -------- | ----- | ------ |
| [`audit_frequency`](#audit_frequency) | PRF001, PRF004, PRF005 | `tsauditor/profiler/frequency.py` |
| [`audit_missing`](#audit_missing) | PRF002, PRF006 | `tsauditor/profiler/missing.py` |
| [`audit_stationarity`](#audit_stationarity) | PRF003 | `tsauditor/profiler/stationarity.py` |

All three raise `ValueError` if the DataFrame index is not a `DatetimeIndex`, and all three return an empty list for an empty DataFrame. All three examine **numeric columns only** — text columns are ignored entirely.

---

## `audit_frequency`

### What it detects

Problems with the time index itself: repeated timestamps, unexpectedly long gaps between rows, and gaps that arrive in clusters.

This is the only profiler check that raises a CRITICAL issue, and it runs first in the pipeline, because a broken index invalidates everything computed afterward.

### Signature

```python
audit_frequency(df: pd.DataFrame, domain: str = None) -> list
```

| Parameter | Type | Default | What it does |
| --------- | ---- | ------- | ------------ |
| `df` | `pd.DataFrame` | required | Must have a `DatetimeIndex` |
| `domain` | `str` or `None` | `None` | `"finance"` fixes the gap threshold at 5.0 days. Anything else derives it from your data. |

Returns a `list` of `Issue` objects (possibly empty).

### How it works

**Step 1 — duplicates (PRF004).** If any timestamp appears more than once, raise CRITICAL immediately. Then drop the duplicates internally, keeping the first, so the gap arithmetic that follows is valid.

Why CRITICAL: a duplicate timestamp does not announce itself. `df.rolling("5D")` silently includes both rows. `df.shift(1)` silently returns the wrong neighbour. `df.resample()` silently aggregates them together. Nothing errors; the numbers are simply wrong. It must be resolved before modeling.

**Step 2 — gap sizes.** Compute the difference between each pair of consecutive timestamps, converted to days.

**Step 3 — pick the threshold.** This is where `domain` matters:

```
domain == "finance"  ->  threshold = 5.0 days              (fixed)
otherwise            ->  threshold = 3 × median gap        (adaptive)
                         (falls back to 1.0 if the median gap is 0)
```

The fixed 5.0 for finance exists because markets close. A daily equity series has a 3-day gap over every weekend and longer over holidays; that is normal, not a defect. Five days lets ordinary weekends and single holidays pass while still catching a genuine multi-day outage.

For everything else the threshold adapts to your data. A sensor sampling every 10 minutes gets a threshold of 30 minutes; a monthly series gets roughly 90 days. You do not have to know your sampling rate in advance.

**Step 4 — large gaps (PRF001).** Any gap at or above the threshold is flagged, as a single WARNING for the column-less dataset level, listing up to five example locations.

**Step 5 — gap clusters (PRF005).** Run-length encoding over the "is this gap large?" boolean sequence. Two or more *consecutive* large gaps form a cluster.

The distinction matters. One large gap is usually a holiday. Several large gaps in a row means your feed was down for a stretch, or the source changed its sampling regime — a structural event, not a calendar quirk.

### Issue codes and evidence

**PRF004 — Duplicate timestamps.** CRITICAL. `column` is `None` (dataset-level).

| Evidence key | Meaning |
| ------------ | ------- |
| `duplicate_count` | Number of rows involved (counts *all* members of each duplicate set, not the excess) |
| `examples` | Up to 5 duplicated timestamps, as strings |

**PRF001 — Large gaps.** WARNING. `column` is `None`.

| Evidence key | Meaning |
| ------------ | ------- |
| `gap_count` | How many gaps exceeded the threshold |
| `maximum_gap_days` | The single largest gap, in days |
| `locations` | Up to 5 timestamps *following* a large gap |

**PRF005 — Clustered gaps.** WARNING. `column` is `None`.

| Evidence key | Meaning |
| ------------ | ------- |
| `cluster_count` | Number of runs of 2+ consecutive large gaps |
| `max_consecutive_gaps` | Longest such run |
| `cluster_start_locations` | Up to 5 cluster start timestamps |

### When it does not fire

- The DataFrame is empty
- There are fewer than 2 rows (no gaps exist to measure)
- All gaps fall below the threshold

### Worked example

```python
import pandas as pd
from tsauditor.profiler.frequency import audit_frequency

idx = pd.DatetimeIndex([
    "2024-01-01",
    "2024-01-02", "2024-01-02",   # duplicate
    "2024-01-03",
    "2024-01-20",                  # 17-day gap
    "2024-01-21",
])
df = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6]}, index=idx)

for issue in audit_frequency(df):
    print(issue.code, issue.severity, issue.evidence)
```

```
PRF004 critical {'duplicate_count': 2, 'examples': ['2024-01-02 00:00:00']}
PRF001 warning {'gap_count': 1, 'maximum_gap_days': 17.0, 'locations': ['2024-01-20 00:00:00']}
```

Read `duplicate_count: 2` carefully — it means two *rows* participate in the duplication, not that there are two extra rows. Only one timestamp is repeated, which is why `examples` has a single entry.

PRF005 did not fire: there is one large gap, and a cluster requires at least two consecutive ones.

### Limitations and false positives

**Adaptive thresholds need enough data.** With only a handful of rows the median gap is unstable, so `3 × median` can be far too tight or too loose. Prefer `domain="finance"` (or a longer series) for short frames.

**The finance preset assumes daily data.** The 5.0-day threshold is calibrated for a daily equity series. Intraday financial data will produce a flood of false negatives — every overnight gap passes unnoticed. For intraday data, leave `domain=None` and let the adaptive rule do its job.

**Regular gaps are not distinguished from irregular ones.** A perfectly regular weekly series has a 7-day gap between every pair of rows; with `domain=None` the adaptive threshold handles that correctly (21 days), but with `domain="finance"` every single gap would be flagged.

---

## `audit_missing`

### What it detects

Two separate problems, per column: an overall missing rate that is too high, and NaNs that arrive in consecutive runs.

### Signature

```python
audit_missing(
    df: pd.DataFrame,
    cluster_threshold: int = None,
    missing_rate_threshold: float = 0.30,
    domain: str = None,
) -> list
```

| Parameter | Type | Default | What it does |
| --------- | ---- | ------- | ------------ |
| `df` | `pd.DataFrame` | required | Must have a `DatetimeIndex` |
| `cluster_threshold` | `int` or `None` | `None` | Minimum consecutive NaNs to count as a cluster. `None` derives it from `domain`. |
| `missing_rate_threshold` | `float` | `0.30` | Flag a column whose missing proportion reaches this. `0.30` = 30%. |
| `domain` | `str` or `None` | `None` | `"finance"` sets `cluster_threshold` to 5; anything else sets it to 3. |

### How it works

**Why two checks and not one.** A column that is 20% missing at random is annoying but usually survivable — interpolation works fine. A column that is 20% missing *in one continuous block* is a different problem entirely: something was broken for a stretch of time, interpolation across the whole block is fabrication, and the missingness itself may carry information. Splitting these into PRF006 (rate) and PRF002 (structure) lets you respond appropriately to each.

**PRF006 — high missing rate.** Simply `missing_count / total_rows >= missing_rate_threshold`.

**PRF002 — clustered missing.** Run-length encoding over the NaN mask, vectorized with NumPy rather than looped. Any run of length `>= cluster_threshold` counts as a cluster.

**Why the domain thresholds differ.** `finance = 5`, everything else `= 3`. A daily price series legitimately has runs of missing values across long weekends and holidays, so a tolerance of 5 avoids flagging the calendar. A sensor sampling continuously has no such excuse — three consecutive missing readings already suggests a dropout, so the tolerance is tighter.

### Issue codes and evidence

**PRF006 — High missing rate.** WARNING. `column` is set.

| Evidence key | Meaning |
| ------------ | ------- |
| `missing_count` | Absolute number of NaNs |
| `missing_percentage` | As a percentage, rounded to 2dp |
| `threshold_percentage` | The threshold it crossed |

**PRF002 — Clustered missing values.** WARNING. `column` is set.

| Evidence key | Meaning |
| ------------ | ------- |
| `missing_percentage` | Overall missing rate for the column |
| `longest_consecutive_run` | Length of the longest NaN run |
| `cluster_count` | How many runs met or exceeded the threshold |
| `first_occurrence` | Timestamp where the first qualifying cluster starts |
| `cluster_threshold` | The threshold actually used |

Note that `longest_consecutive_run` is the longest run in the column *overall*, not the longest qualifying one. In practice these coincide, since the longest run is necessarily at or above the threshold whenever any run is.

### When it does not fire

- The column has zero missing values (skipped immediately)
- The column is non-numeric
- Missing rate is below `missing_rate_threshold` **and** no run reaches `cluster_threshold`
- The DataFrame is empty

### Worked example

```python
import pandas as pd, numpy as np
from tsauditor.profiler.missing import audit_missing

idx = pd.date_range("2024-01-01", periods=20, freq="D")
values = [1.0, 2, 3, np.nan, np.nan, np.nan, np.nan, 8, 9, 10,
          11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
df = pd.DataFrame({"sensor": values}, index=idx)

for issue in audit_missing(df):
    print(issue.code, issue.column, issue.evidence)
```

```
PRF002 sensor {'missing_percentage': 20.0, 'longest_consecutive_run': 4, 'cluster_count': 1, 'first_occurrence': '2024-01-04 00:00:00', 'cluster_threshold': 3}
```

Only PRF002 fired. The column is 20% missing, which is below the 30% rate threshold, so PRF006 stayed quiet — but those four NaNs are consecutive, which exceeds the default cluster threshold of 3. This is exactly the case the two-check split exists for: the *rate* is acceptable, the *shape* is not.

### Limitations and false positives

**The 30% rate threshold is arbitrary** and not domain-adjusted. A column that is 29% missing gets no PRF006. If your tolerance differs, pass `missing_rate_threshold=` explicitly.

**Non-numeric columns are invisible.** A completely empty string column produces no issue at all.

**Cluster length is measured in rows, not elapsed time.** Three consecutive missing rows in an irregular series might span three minutes or three months. The check cannot tell the difference.

---

## `audit_stationarity`

### What it detects

Columns whose statistical properties drift over time — specifically, columns that fail to reject the null hypothesis of a unit root under the Augmented Dickey-Fuller (ADF) test.

**This is the most expensive check in the library.** Skip it with `scan(..., run_stationarity=False)` when you do not need it.

### Plain-language background

A series is **stationary** when its mean and variance do not systematically change over time. Daily temperature is roughly stationary: it fluctuates, but around a stable level. A stock price is not: it wanders, and the level it wanders around today has nothing to do with the level five years ago.

This matters because many time-series models — ARIMA, linear regression on lagged values, most classical forecasting — assume stationarity. Fed a non-stationary series they can produce a *spurious regression*: an impressive R² that reflects nothing but two variables both drifting upward over time.

The usual remedy is **differencing**: model the change from one step to the next rather than the level. Prices are non-stationary; returns usually are not.

The **ADF test** formalises this. Its null hypothesis is "this series has a unit root", which loosely means "this series wanders". A p-value above 0.05 means you cannot reject that null, so the series is treated as non-stationary.

### Signature

```python
audit_stationarity(
    df: pd.DataFrame,
    alpha: float = 0.05,
    min_obs: int = 25,
    max_lag: int = None,
    domain: str = None,
) -> list
```

| Parameter | Type | Default | What it does |
| --------- | ---- | ------- | ------------ |
| `df` | `pd.DataFrame` | required | Must have a `DatetimeIndex` |
| `alpha` | `float` | `0.05` | Significance level. Flag when `p_value > alpha`. |
| `min_obs` | `int` | `25` | Columns with fewer non-null observations are skipped. |
| `max_lag` | `int` or `None` | `None` | Caps the ADF lag search. **This is your speed knob.** |
| `domain` | `str` or `None` | `None` | Accepted for API consistency; does not change behaviour. |

### How it works

Per numeric column:

1. Drop NaNs, then replace `inf` / `-inf` with NaN and drop those too.
2. Skip if fewer than `min_obs` observations remain.
3. Skip if the column is constant — `adfuller` raises `"Invalid input, x is constant"` on such input, and a constant series is trivially (if degenerately) stationary anyway.
4. Run `adfuller(series, maxlag=max_lag, autolag="AIC")`.
5. If it raises `ValueError` or `LinAlgError` on a near-singular input, skip that column rather than aborting the whole scan.
6. Flag INFO if `p_value > alpha`.

**Why `max_lag` is the speed knob.** With `max_lag=None`, statsmodels picks a maximum lag from the sample size and then fits an OLS regression at *every* lag from 0 up to that maximum, selecting the best by AIC. On a long series that is a great many regressions, per column. Passing `max_lag=4` caps the search at five fits, which is dramatically faster at a modest cost in test precision.

**Why the severity is only INFO.** Non-stationarity is not a data *defect*. A price series is supposed to be non-stationary; that is what prices do. This is a modeling advisory, not a bug report — hence INFO, and hence why it never counts against your health score.

### Issue code and evidence

**PRF003 — Non-stationary column.** INFO. `column` is set.

| Evidence key | Meaning |
| ------------ | ------- |
| `adf_statistic` | The ADF test statistic (more negative = stronger evidence of stationarity) |
| `p_value` | The p-value, rounded to 4dp |
| `n_observations` | Observations used by the test after lag adjustment |
| `alpha` | The significance level applied |

### When it does not fire

- Fewer than `min_obs` (default 25) non-null, finite observations
- The column is constant
- `adfuller` raised on degenerate input
- `p_value <= alpha` — the column is stationary
- `run_stationarity=False` was passed to `scan()`

### Worked example

```python
import pandas as pd, numpy as np
from tsauditor.profiler.stationarity import audit_stationarity

rng = np.random.default_rng(0)
idx = pd.date_range("2024-01-01", periods=200, freq="D")

df = pd.DataFrame({
    "random_walk": np.cumsum(rng.normal(size=200)),  # non-stationary by construction
    "white_noise": rng.normal(size=200),             # stationary by construction
}, index=idx)

for issue in audit_stationarity(df):
    print(issue.code, issue.column, issue.evidence)
```

```
PRF003 random_walk {'adf_statistic': -1.6696, 'p_value': 0.4468, 'n_observations': 199, 'alpha': 0.05}
```

Exactly one column flagged, and it is the right one. The random walk's p-value of 0.4468 is far above 0.05 — no evidence against a unit root. `white_noise` produced a p-value below 0.05 and was correctly left alone.

### Limitations and false positives

**The ADF test has low power on short series.** With a few dozen points it frequently fails to reject the null even for genuinely stationary data, producing a false PRF003. Treat results on short series with suspicion.

**A single trend break can look like a unit root.** ADF is known to mistake a structural break in the mean for non-stationarity. If your series has a regime change, expect PRF003 whether or not it is warranted.

**Flagging is expected and often correct.** Price columns *should* be flagged. PRF003 on a price series is the tool working, not a problem to fix.

**`domain` does nothing here.** It is accepted so every detector has the same signature shape, but it has no effect on the test.
