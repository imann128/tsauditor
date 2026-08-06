# Anomaly Detectors

The anomaly module asks whether individual **values** are plausible. Two functions, three issue codes.

| Function | Codes | Module |
| -------- | ----- | ------ |
| [`audit_point_anomalies`](#audit_point_anomalies) | ANO002 | `tsauditor/anomaly/point.py` |
| [`audit_contextual_anomalies`](#audit_contextual_anomalies) | ANO001, ANO003 | `tsauditor/anomaly/contextual.py` |

Both raise `ValueError` without a `DatetimeIndex`, return `[]` for an empty DataFrame, and examine numeric columns only.

---

## The distinction that matters most

`tsauditor` looks for outliers in two different ways, and understanding the difference will save you confusion.

**ANO002 is global.** It asks: *is this value extreme compared to the entire column?* It uses the column's overall mean, standard deviation, and quartiles.

**ANO003 is local.** It asks: *is this value extreme compared to its immediate neighbours?* It uses a rolling window around each point.

Neither subsumes the other.

A value can be globally extreme but locally ordinary. In a series that trends steadily upward, the final values are the largest in the column (globally extreme) but each sits comfortably among its neighbours. ANO002 may flag them; ANO003 will not, and ANO003 is right.

A value can be locally extreme but globally ordinary. This is the more dangerous case, because a global check misses it entirely. Here is a runnable demonstration:

```python
import pandas as pd, numpy as np
from tsauditor.anomaly.point import audit_point_anomalies
from tsauditor.anomaly.contextual import audit_contextual_anomalies

idx  = pd.date_range("2024-01-01", periods=120, freq="D")
vals = np.linspace(0.0, 240.0, 120)   # a slow, clean ramp from 0 to 240
vals[30] = 200.0                      # globally ordinary, locally absurd

df = pd.DataFrame({"level": vals}, index=idx)

print("the planted value :", vals[30])
print("global range      :", vals.min(), "->", vals.max())
print("its neighbours    :", np.round(vals[27:34], 1))
print()
print("audit_point_anomalies      ->", audit_point_anomalies(df))
for i in audit_contextual_anomalies(df):
    print("audit_contextual_anomalies ->", i.code, i.column, i.evidence)
```

```
the planted value : 200.0
global range      : 0.0 -> 240.0
its neighbours    : [ 54.5  56.5  58.5 200.   62.5  64.5  66.6]

audit_point_anomalies      -> []
audit_contextual_anomalies -> ANO003 level {'n_spikes': 1, 'max_spike_zscore': 11.1472, 'zero_variance_context': False}
```

A value of 200 in a column that spans 0 to 240 is unremarkable by any global measure: the z-score does not come close to the threshold, and 200 sits well inside the interquartile fence. ANO002 finds nothing. But its neighbours are all near 58, and against that local backdrop it scores z = 11.1. This is why both checks exist.

---

## `audit_point_anomalies`

### What it detects

Values that are extreme relative to the whole column, using two independent statistical rules.

### Signature

```python
audit_point_anomalies(
    df: pd.DataFrame,
    zscore_threshold: float = None,
    domain: str = None,
) -> list
```

| Parameter | Type | Default | What it does |
| --------- | ---- | ------- | ------------ |
| `df` | `pd.DataFrame` | required | Must have a `DatetimeIndex` |
| `zscore_threshold` | `float` or `None` | `None` | Flag when \|z\| exceeds this. An explicit value always wins over `domain`. |
| `domain` | `str` or `None` | `None` | `"finance"` → 5.0, `"sensor"` → 3.5, `None` → 4.0. Consulted only when `zscore_threshold` is `None`. |

`domain` supplies a default, it does not override you:

```python
audit_point_anomalies(df, zscore_threshold=2.0, domain="finance")   # uses 2.0
audit_point_anomalies(df, domain="finance")                         # uses 5.0
audit_point_anomalies(df, zscore_threshold=0.0)                     # uses 0.0, not the default
```

### How it works

Per numeric column, NaNs dropped:

**Rule 1: Z-score.** How many standard deviations from the mean is this point?

```
z = (value - mean) / std          flagged when |z| > threshold
```

**Rule 2: IQR fence.** The classic Tukey rule. Q1 and Q3 are the 25th and 75th percentiles; IQR is their difference.

```
flagged when  value < Q1 - 1.5 × IQR   or   value > Q3 + 1.5 × IQR
```

**Combination: `z_mask OR iqr_mask`.** A point flagged by *either* rule is reported.

**Why two rules.** They fail in opposite directions, so running both covers each one's blind spot.

The z-score uses the mean and standard deviation, and both are themselves distorted by outliers. Enough extreme values inflate the standard deviation so much that they push their own z-scores back under the threshold: the outliers mask themselves. The IQR rule uses percentiles, which barely move when a minority of points are extreme, so it keeps working long after the z-score has given up.

**This is not theoretical. It is measurable, and it is the reason the combined rule matters.** Planting outliers at 10σ into 1,000 clean points and reading the evidence back from the detector:

| outliers planted | `zscore_outlier_count` | `iqr_outlier_count` | `agreement_count` |
| ---------------- | ---------------------- | ------------------- | ----------------- |
| 1 | 1 | 11 | 1 |
| 5 | 5 | 15 | 5 |
| 20 | 20 | 29 | 20 |
| **50** | **0** | **57** | **0** |
| **150** | **0** | **151** | **0** |
| **300** | *no issue raised at all* | | |

At 5% contamination the z-score half is already blind. The IQR half still finds all of them, so ANO002 still fires, which is exactly why the detector uses `z OR IQR` rather than the z-score alone.

Conversely, the IQR rule is fixed at 1.5× and cannot be tuned by domain, and on a heavily skewed distribution it flags a large share of the legitimate tail. The z-score, tuned by domain, handles that better.

**Why the domain thresholds differ.** Financial returns have famously fat tails: a five-sigma daily move is unusual but genuinely happens, so `finance` uses 5.0 to avoid flagging real market events. Physical sensor readings are much better behaved, so `sensor` tightens to 3.5 to catch smaller glitches.

### Issue code and evidence

**ANO002: Point anomalies.** WARNING. One issue per affected column (not per value).

| Evidence key | Meaning |
| ------------ | ------- |
| `zscore_outlier_count` | Points flagged by the z-score rule |
| `iqr_outlier_count` | Points flagged by the IQR rule |
| `agreement_count` | Points flagged by **both** rules. High confidence when non-zero, but see the warning below before reading a **zero** as reassurance. |
| `esd_outlier_count` | Generalized ESD estimate of the true outlier count. **Computed only when `agreement_count` is ambiguous** (z-score 0, IQR above 0); `None` otherwise. |
| `masking_suspected` | `True` when the z-score rule is blind but ESD finds substantial contamination. |
| `max_zscore` | Largest \|z\| in the column |
| `worst_value` | The value with that largest \|z\| |
| `worst_timestamp` | When it occurred |

When both independent rules agree, the point is very likely a genuine error. A **non-zero** `agreement_count` is therefore strong evidence.

**A zero `agreement_count` is not evidence of the opposite, and this is easy to get wrong.** See [Reading a zero agreement_count](#reading-a-zero-agreement_count) immediately below.

### When it does not fire

- The column is entirely NaN
- The column has zero standard deviation (constant)
- No point crosses either rule
- The column is non-numeric

### Worked example

```python
import pandas as pd, numpy as np
from tsauditor.anomaly.point import audit_point_anomalies

rng  = np.random.default_rng(42)
idx  = pd.date_range("2024-01-01", periods=100, freq="D")
vals = rng.normal(100, 5, size=100)
vals[50] = 500.0                       # one blatant outlier

df = pd.DataFrame({"temperature": vals}, index=idx)

for issue in audit_point_anomalies(df):
    print(issue.code, issue.column, issue.evidence)
```

```
ANO002 temperature {'zscore_outlier_count': 1, 'iqr_outlier_count': 1, 'agreement_count': 1, 'max_zscore': 9.8538, 'worst_value': 500.0, 'worst_timestamp': '2024-02-20 00:00:00'}
```

Both rules agree (`agreement_count: 1`), the worst value is the 500.0 that was planted, and its z-score is 9.85 against a default threshold of 4.0.

### Reading a zero `agreement_count`

A high IQR count with a zero z-score count has **two possible causes, and they call for opposite responses**:

1. **A skewed distribution.** Volume, market capitalisation and most count-like columns are right-skewed, and the 1.5×IQR fence flags a chunk of the legitimate upper tail. Nothing is wrong. This is common and usually benign.
2. **Heavy contamination masking the z-score.** As the table above shows, 50 genuine planted outliers produce `zscore_outlier_count: 0` and `agreement_count: 0`, a signature *identical* to case 1.

**Do not read a zero `agreement_count` as "probably just skew."** Earlier versions of this page said exactly that, and it was wrong: it would lead you to dismiss the case where contamination is worst.

**Since 0.3.0 the evidence answers this for you.** `esd_outlier_count` comes from Rosner's Generalized ESD test, which removes the most extreme point and *recomputes* the mean and standard deviation before testing the next, so masking cannot occur by construction:

| scenario | `z` | `iqr` | `agreement` | `esd` | `masking_suspected` |
| -------- | --- | ----- | ----------- | ----- | ------------------- |
| clean Gaussian | 0 | 10 | 0 | **0** | False |
| 50 planted outliers | 0 | 57 | 0 | **50** | **True** |
| 150 planted outliers | 0 | 151 | 0 | **150** | **True** |
| lognormal (skewed, clean) | 3 | 71 | 3 | N/A | False |

The first two rows are indistinguishable from the counts alone. ESD separates them exactly.

It is computed only for the ambiguous case and reported as `None` otherwise, since it is O(k·n): about 27ms on 1,000 points. **It never changes what gets flagged**; flagging remains the z-score OR IQR rule.

If you want to look yourself, or you are on an older version:

Compute a robust z-score (one that uses the median and MAD, so the suspect points do not inflate the scale that judges them) and sort it descending:

```python
s = df["your_col"].dropna()
if trending:                       # a price level, a cumulative count, etc.
    s = s.diff().dropna()          # see the caveat below: this matters

med = s.median()
mad = (s - med).abs().median() * 1.4826
robust_z = ((s - med) / mad).abs().sort_values(ascending=False)

print(robust_z.head(60).to_string())
```

What you are looking for is **where the values stop being extreme**, not how extreme the largest one is:

- **Skew**: values decay smoothly from the single largest. There is no natural boundary; each point is slightly less extreme than the one before. This is one distribution with a long tail.
- **Contamination**: a group of comparably-extreme points, then a distinct drop to the rest. The suspect points form their own cluster.

**Do not use the maximum to decide.** Measured on real examples, clean lognormal data reached a robust z of **52.1**, while a series with 50 genuinely planted outliers only reached **9.4**. The largest value tells you about tail thickness, not about contamination.

**This is a judgement call, not a formula.** There is no threshold that reliably separates the two cases: testing across clean Gaussian, lognormal, exponential and several contamination levels, the boundary between "smooth tail" and "separate cluster" is visible by eye but does not reduce to a single reliable number. Treat it as evidence to weigh alongside what you know about how the column was produced.

**Caveat: difference a trending column first.** On a trending series (a price level, a cumulative count) the robust z-score flags the trend itself, because values late in the trend are legitimately far from the global median. On the OGDC `Price` column it flags 85 points, all in the final uptrend, all contiguous, none of them errors. After differencing, 33. Skipping this step makes the diagnostic mislead you in the opposite direction.

### Limitations and false positives

**A zero `agreement_count` is ambiguous.** See the section directly above.

**The IQR rule fires readily on skewed data.** The 1.5×IQR fence will flag a substantial portion of a right-skewed upper tail when nothing is wrong.

**Beyond roughly 30% contamination, ANO002 stops reporting entirely.** At that point the extreme values have shifted the quartiles far enough that the IQR fence no longer excludes them, and the standard deviation is far too large for the z-score. Both rules return nothing and **no issue is raised at all**. This is an inherent limit of both methods rather than an implementation flaw (at 30% the "outliers" are arguably a second population rather than anomalies), but it means **silence from ANO002 is not proof of clean data.**

**One issue per column, not per point.** If a column has 200 outliers you get one `Issue`. The counts are in the evidence; to recover the actual positions you must recompute the mask yourself (see [Internals](Internals#_outlier_maskvalues-z_thresh)).

**Non-stationary columns will be flagged.** A strongly trending series has legitimately extreme values at both ends. Consider auditing the differenced series instead of the level.

---

## `audit_contextual_anomalies`

### What it detects

Two locally-defined problems: values that are **frozen** (ANO001) and values that **jump away from their neighbours** (ANO003).

### Signature

```python
audit_contextual_anomalies(
    df: pd.DataFrame,
    stuck_window: int = None,
    spike_threshold: float = None,
    spike_window: int = None,
    domain: str = None,
    handle_missing: str = "strict",
) -> list
```

| Parameter | Type | Default | What it does |
| --------- | ---- | ------- | ------------ |
| `df` | `pd.DataFrame` | required | Must have a `DatetimeIndex` |
| `stuck_window` | `int` or `None` | `None` | A run **longer than** this is flagged. Derived from `domain` if `None`. |
| `spike_threshold` | `float` or `None` | `None` | Local z-score threshold. Derived from `domain` if `None`. |
| `spike_window` | `int` or `None` | `None` | Size of the local context window. Falls back to **21**. |
| `domain` | `str` or `None` | `None` | `"finance"` or `"sensor"`: see table below |
| `handle_missing` | `str` | `"strict"` | `"interpolate"` fills single-gap NaNs first; anything else leaves them |

Domain defaults:

| `domain` | `stuck_window` | `spike_threshold` |
| -------- | -------------- | ----------------- |
| `"finance"` | 5 | 4.0 |
| `"sensor"` | 3 | 3.0 |
| `None` | 5 | 3.5 |

As with `audit_point_anomalies`, an explicitly passed value always wins; `domain` fills in only what you left as `None`. A deliberate `0` is honoured: resolution uses `is None`, not the `x or default` idiom that would treat `0` as unset.

### How ANO001 works: stuck values

A "stuck" value is one that repeats unchanged for an implausibly long run. In sensor data this is the classic signature of a frozen instrument: the hardware failed but is still reporting its last reading. In financial data it usually means a forward-fill was applied somewhere upstream and nobody wrote it down.

The implementation is a run-length grouping over a **bridged** view of the series, not the raw one:

```python
bridge_series = series.interpolate(method="linear", limit=1)
diffs  = bridge_series.diff().ne(0).cumsum()   # a new group ID whenever the value changes
counts = bridge_series.groupby(diffs).transform("count")
stuck  = (counts > stuck_window) & series.notna()
```

Two details worth noting. First, the comparison is **strictly greater than**, so `stuck_window=5` flags runs of 6 or more. Second — and this is the opposite of what grouping on the raw series would do — a single missing reading inside an otherwise-flat run still counts as one continuous stuck run, not two shorter ones. Grouping on the raw series would split it: `diff()` against a NaN yields NaN, which is `!= 0`, so the gap itself and the row right after it would both look like "changed," and neither resulting half might cross `stuck_window` even though the true, uninterrupted run does. Interpolating a single NaN (`limit=1`) only produces a zero diff when both neighbours already agree, so a genuine transition — a gap between two *different* values — still breaks the group correctly; this bridges a real stuck run without masking a real change. Concretely: `[1, 1, 1, NaN, 1, 1, 1]` bridges to one run of length 7, but `[1, 1, 1, NaN, 2, 2, 2]` still correctly splits into two runs of 3.

### How ANO003 works: contextual spikes

This is the most subtle detector in the library, and the reason for its design is worth understanding.

The obvious implementation is a centered rolling z-score: for each point, take the mean and standard deviation of its surrounding window and see how far the point sits from that mean.

**That implementation does not work**, because the point is inside its own window. A large spike inflates both the window mean and the window standard deviation, and the two effects cancel. In a 5-point centered window, a 50× spike scored only z ≈ 1.8, nowhere near any sensible threshold. The spike masked itself. This was a real bug in an earlier version of this module.

The fix is to compute the local mean and standard deviation from the neighbours **excluding the point itself**. Doing that efficiently requires a trick: rather than looping, compute the rolling sum and rolling sum-of-squares, then subtract the point's own contribution.

```
n_excl     = rolling_count - 1
sum_excl   = rolling_sum   - x
sumsq_excl = rolling_sumsq - x²

local_mean = sum_excl / n_excl
local_var  = sumsq_excl / n_excl - local_mean²
local_std  = sqrt(max(local_var, 0))        # the max() kills tiny negative float error

z = |x - local_mean| / local_std
```

**Why the window is 21 and not 5.** The window must be wide enough for the standard deviation of the neighbours to be a stable estimate. A 4-or-5-point window leaves 3-4 neighbours after excluding the point, and the standard deviation of 3 numbers is extremely noisy: it collapses toward zero by chance often enough to flood the output with false positives. Twenty-one is wide enough to be stable while still being local. `min_periods` is `max(3, window // 2)`, so points near the series edges are still evaluated with a partial window.

**The flat-context case.** If every neighbour has the identical value, `local_std` is 0 and `z = deviation / 0` is undefined. A point that differs from a perfectly flat neighbourhood is unambiguously a spike, so it is flagged explicitly via a separate condition rather than being silently dropped as NaN:

```
flat_context_spike = (local_std == 0) & (deviation > 0) & (n_excl >= 2)
```

That case is reported in the evidence as `zero_variance_context: True`.

### Issue codes and evidence

**ANO001: Stuck values.** WARNING. One issue per column.

| Evidence key | Meaning |
| ------------ | ------- |
| `max_stuck_duration` | Length of the longest flagged run |

This evidence is deliberately minimal: no timestamps, no run count.

**ANO003: Contextual spikes.** WARNING. One issue per column.

| Evidence key | Meaning |
| ------------ | ------- |
| `n_spikes` | Number of flagged points |
| `max_spike_zscore` | Largest finite local z-score, or `None` if all were infinite |
| `zero_variance_context` | `True` if at least one spike was found against a perfectly flat window |

### When it does not fire

- The column is entirely NaN after any interpolation
- No run exceeds `stuck_window` (ANO001)
- No point exceeds `spike_threshold` and no flat-context spike exists (ANO003)
- Fewer than 3 usable points in a window (`min_periods` not met)

### Worked example: ANO001

```python
import pandas as pd, numpy as np
from tsauditor.anomaly.contextual import audit_contextual_anomalies

rng = np.random.default_rng(42)
vals = list(rng.normal(20, 1, size=40))
vals[10:18] = [20.0] * 8                     # the sensor froze for 8 readings

df = pd.DataFrame({"reading": vals},
                  index=pd.date_range("2024-01-01", periods=40, freq="D"))

for issue in audit_contextual_anomalies(df, domain="sensor"):
    print(issue.code, issue.column, issue.evidence)
```

```
ANO001 reading {'max_stuck_duration': 8}
ANO003 reading {'n_spikes': 2, 'max_spike_zscore': 3.3685, 'zero_variance_context': False}
```

With `domain="sensor"` the stuck window is 3, so a run of 8 is comfortably flagged. Under `domain="finance"` (window 5) this same run would still be flagged, since 8 > 5, but a run of 4 would be caught only by the sensor preset.

**Note the second line.** `audit_contextual_anomalies` runs both checks in one call, so ANO003 also reported two spikes: the points where the series jumps out of the frozen run and back into normal noise. Those transitions genuinely are locally extreme, since the flat run gives them a near-zero local standard deviation to be measured against.

This is worth remembering generally: **a stuck run tends to produce ANO003 findings at its edges.** They are not independent evidence of a second problem; they are a consequence of the first one. Fix the stuck run and they disappear.

### Limitations and false positives

**Legitimately constant data will be flagged.** A binary flag column, a categorical encoding, a market that was genuinely closed, a `Regime` indicator: all produce long identical runs, and ANO001 flags all of them. This is the single most common false positive in the library. If a column is *supposed* to be piecewise-constant, ignore ANO001 for it.

**A binary target trips ANO001 constantly.** This is precisely why `apply_fixes` refuses to repair the target column: a 0/1 label has runs of repeated values everywhere, and "fixing" them would destroy the label.

**`spike_window=21` is not adaptive.** It is a fixed number of rows regardless of your sampling rate. For high-frequency data 21 points may be a fraction of a second; for monthly data it is nearly two years. Set `spike_window` explicitly if 21 rows is the wrong notion of "local" for your series.

**Short series get unstable local estimates.** With fewer than about 40 points, `min_periods` admits windows with few neighbours and the local standard deviation becomes noisy.

**Consecutive spikes hide each other.** Two adjacent extreme values each appear in the other's context window, inflating the local standard deviation and lowering both z-scores. A sustained level shift is generally *not* caught by ANO003; it will look like a legitimate new local mean after a few points.
