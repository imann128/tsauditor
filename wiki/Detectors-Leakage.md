# Leakage Detectors

This is the core of the library. Five checks, one question: **does this feature know something it could not have known at prediction time?**

| Function | Code | Severity | Module |
| -------- | ---- | -------- | ------ |
| [`audit_equivalence`](#lek001-target-equivalence) | LEK001 | CRITICAL | `leakage/equivalence.py` |
| [`audit_correlation_leakage`](#lek002-positive-lag-correlation-peak) | LEK002 | WARNING | `leakage/correlation.py` |
| [`audit_temporal_leakage`](#lek003-lookahead-beyond-persistence) | LEK003 | WARNING | `leakage/temporal.py` |
| [`audit_asof_leakage`](#lek004-as-of-availability-leakage) | LEK004 | CRITICAL | `leakage/asof.py` |
| [`audit_combination_leakage`](#lek005-combination-leakage) | LEK005 | CRITICAL | `leakage/combination.py` |

LEK001–LEK003 and LEK005 require `target=`. Without it, `scan()` skips them silently. LEK004 is target-independent but requires you to supply `available_at=`.

For panel data there is a sixth leakage check, [PNL002](Panel-Data#pnl002-cross-sectional-lookahead), which catches leaks that live in the relationship *between* entities.

---

## What data leakage is

Data leakage is when information that would not be available at prediction time leaks into your training features. The model learns from it, scores brilliantly in backtesting, and fails completely in production, because in production that information genuinely is not there.

It is dangerous specifically because it produces *good* results. A bug that makes your model worse gets found immediately. A bug that makes your model look excellent gets shipped.

`tsauditor` detects five flavours:

1. **The feature is the target.** Same information, different column name. (LEK001)
2. **The feature aligns with the target's future.** Its correlation peaks at a positive lag. (LEK002)
3. **The feature knows the future better than persistence explains.** The signature of a centered window. (LEK003)
4. **The feature was published after the timestamp it sits at.** A point-in-time violation. (LEK004)
5. **No single feature leaks, but a group of them rebuilds the target.** (LEK005)

For panel data, [PNL002](Panel-Data#pnl002-cross-sectional-lookahead) adds a sixth: a leak living in the relationship *between* entities.

---

## Why every check uses ranks, not Pearson

Every leakage detector uses **Spearman** correlation or **AUC**, never Pearson. This is a deliberate decision and the single most important design choice in the library.

Pearson correlation measures **linear** association. But the question these checks ask is not "is this relationship linear?" It is "does this feature *determine* the target?", a question about determinism, not linearity.

The gap between those two questions is exactly where the OGDC bug lived.

### The point-biserial ceiling

When you compute Pearson correlation between a continuous feature and a **binary** 0/1 target, you get the *point-biserial* correlation. This statistic has a hard mathematical ceiling of roughly:

```
√(2/π) ≈ 0.798
```

That ceiling applies **even when the feature perfectly determines the target**. A feature whose sign *defines* the label, `Direction = 1 if ChangeP > 0 else 0`, cannot score above about 0.8 in Pearson, no matter how perfect the relationship is.

So a leakage detector with a sensible-looking Pearson threshold of 0.95, or even 0.85, would score the OGDC leak at ~0.77 and wave it through. Here is that failure, demonstrated:

```python
import numpy as np
from scipy.stats import pearsonr

rng       = np.random.default_rng(7)
change    = rng.normal(0, 1, 300)
direction = (change > 0).astype(int)   # the target is literally defined by the feature

r, _ = pearsonr(change, direction)
print("Pearson correlation:", round(r, 4))
```

```
Pearson correlation: 0.7733
```

A perfect, definitional leak scores 0.77. Under any reasonable Pearson threshold it passes as clean.

**AUC scores the same relationship at exactly 1.0.** That is why LEK001 uses AUC for binary targets. Spearman is used for continuous targets and for LEK002/LEK003, because it captures any *monotonic* relationship, including non-linear ones a log or square transform would hide from Pearson, and is robust to outliers.

---

## LEK001: Target equivalence
### What it detects

A feature that near-deterministically reproduces the target at lag 0. The feature *is* the target, arrived at by a different route.

This is the flagship check, rated CRITICAL, and the one the library was written for.

### Signature

```python
audit_equivalence(
    df: pd.DataFrame,
    target: str,
    continuous_threshold: float = 0.95,
    binary_threshold: float = 0.95,
    min_obs: int = 30,
    domain: Optional[str] = None,
) -> List[Issue]
```

| Parameter | Type | Default | What it does |
| --------- | ---- | ------- | ------------ |
| `df` | `pd.DataFrame` | required | Input data |
| `target` | `str` | required | Target column name. Raises `ValueError` if absent. |
| `continuous_threshold` | `float` | `0.95` | \|Spearman ρ\| threshold, used when the target is continuous |
| `binary_threshold` | `float` | `0.95` | AUC-separation threshold, used when the target is binary. Loosen toward 0.90 to tolerate label noise. |
| `min_obs` | `int` | `30` | Minimum pairwise-complete observations required to trust a score |
| `domain` | `str` or `None` | `None` | Accepted for API consistency; **has no effect** |

### How it works

**Step 1, classify the target.** Count distinct non-null values.

- Fewer than 2 → return immediately. A constant target has nothing to reproduce.
- Exactly 2 → **binary**. The two categories are sorted by their string form and mapped to 0.0 and 1.0, so this works for numeric `0/1` and categorical `"up"/"down"` alike.
- More than 2 → **continuous**. Must be numeric, or `ValueError`.

**Step 2, pick the metric.**

| Target type | Metric | Threshold |
| ----------- | ------ | --------- |
| Binary | AUC separation, `max(AUC, 1 - AUC)` | `binary_threshold` |
| Continuous | \|Spearman ρ\| | `continuous_threshold` |

**Step 3, score each numeric feature** against the target, using only pairwise-complete rows with `inf` treated as missing.

**Understanding AUC here.** AUC is computed via the Mann-Whitney rank statistic, using average ranks so ties are handled correctly. It answers: *if I pick one random row where the target is 1 and one where it is 0, how often does the feature rank the class-1 row higher?*

- **0.5**, no separation at all. The feature is useless, which is what an unrelated column looks like.
- **1.0**, perfect separation. Every class-1 row outranks every class-0 row.
- **0.0**, also perfect separation, just inverted.

Because 0.0 is as damning as 1.0, the check uses `max(AUC, 1 - AUC)`, making it direction-agnostic. A perfectly inverted leak scores 1.0 and is caught.

Both metrics live on a comparable [0, 1] scale, which is why one threshold of 0.95 is meaningful for either target type.

### Evidence

**Binary target:**

| Key | Meaning |
| --- | ------- |
| `metric` | `"auc"` |
| `auc` | The raw AUC, before the `max()`: tells you the direction |
| `separation` | `max(auc, 1 - auc)`, the value compared against the threshold |
| `threshold` | Threshold applied |
| `target_type` | `"binary"` |
| `n_obs` | Pairwise-complete observations used |

**Continuous target:**

| Key | Meaning |
| --- | ------- |
| `metric` | `"spearman"` |
| `spearman_rho` | Signed ρ |
| `threshold` | Threshold applied |
| `target_type` | `"continuous"` |
| `n_obs` | Pairwise-complete observations used |

### When it does not fire

- The target has fewer than 2 distinct non-null values
- The feature has fewer than `min_obs` (default 30) pairwise-complete observations with the target, a high score from a handful of points is spurious
- The feature is constant (zero variance)
- The feature is non-numeric
- For binary targets, only one class is present in the overlapping rows
- The score falls below the threshold

### Worked example

```python
import pandas as pd, numpy as np
from tsauditor.leakage.equivalence import audit_equivalence

rng    = np.random.default_rng(7)
n      = 300
idx    = pd.date_range("2024-01-01", periods=n, freq="D")
change = rng.normal(0, 1, n)

df = pd.DataFrame({
    "Direction": (change > 0).astype(int),   # target, defined by ChangeP
    "ChangeP":   change,                     # the leak
    "noise":     rng.normal(0, 1, n),        # an innocent column
}, index=idx)

for issue in audit_equivalence(df, target="Direction"):
    print(issue.column, issue.evidence)
```

```
ChangeP {'metric': 'auc', 'auc': 1.0, 'separation': 1.0, 'threshold': 0.95, 'target_type': 'binary', 'n_obs': 300}
```

`ChangeP` is caught with a perfect AUC of 1.0. `noise` is correctly left alone. Recall from the section above that Pearson scored this same relationship at 0.7733.

### Limitations and false positives

**A legitimately excellent feature will be flagged.** LEK001 cannot distinguish "this feature is the target in disguise" from "this feature is a genuinely superb predictor that happens to be available in advance." It flags; you judge. The suggestion text says so explicitly: *keep it only if you can confirm it is genuinely available at prediction time.*

**Only lag-0 equivalence is checked.** A feature equal to the target shifted by one step is invisible to LEK001. That is LEK002's job.

**Only monotonic relationships are found.** A feature equal to `target²` on a symmetric range is perfectly deterministic but not monotonic, and Spearman will not see it.

**Multi-feature leakage is not LEK001's job.** If no single column reproduces the target but a group of them together does, LEK001 stays silent by design, that is [LEK005](#lek005-combination-leakage).

**`domain` does nothing.** It is accepted for signature consistency only.

---

## LEK002: Positive-lag correlation peak
### What it detects

A feature whose correlation with the target is strongest at a **positive lag**, meaning it aligns better with the target's *future* than with its present.

### Signature

```python
audit_correlation_leakage(
    df: pd.DataFrame,
    target: str,
    max_lag: int = 10,
    min_correlation: float = 0.5,
    min_obs: int = 30,
    domain: Optional[str] = None,
) -> List[Issue]
```

| Parameter | Type | Default | What it does |
| --------- | ---- | ------- | ------------ |
| `max_lag` | `int` | `10` | Lags tested in each direction, so the search covers −10 to +10 |
| `min_correlation` | `float` | `0.5` | The peak must reach this magnitude. Not a formality: see the limitation below |
| `min_obs` | `int` | `30` | Minimum overlapping observations at a given lag for it to count |
| `domain` | `str` or `None` | `None` | No effect |

### How it works

Define the cross-correlation at lag τ:

```
r(τ) = corr( feature_t , target_{t+τ} )
```

So **τ > 0 compares the feature against the target's future**. The check sweeps τ from `−max_lag` to `+max_lag`, records where `|r(τ)|` is largest, and flags the feature if that peak occurs at a positive τ *and* reaches `min_correlation`.

The logic is straightforward: a legitimate feature carries information from the past or present, so its association with the target should peak at τ ≤ 0. If it peaks in the future, the feature is somehow carrying future information.

**Implementation note.** The target is rank-transformed **once** before the loop, and shifted ranks are correlated at each lag. Ranking inside the loop was the previous hot path. Since Spearman is Pearson-on-ranks, this is equivalent and much faster.

### Evidence

| Key | Meaning |
| --- | ------- |
| `peak_lag` | The positive lag where `\|r\|` was maximal |
| `peak_correlation` | The **signed** correlation at that lag |
| `min_correlation` | Threshold applied |
| `max_lag` | Search range used |
| `metric` | `"spearman"` |

### When it does not fire

- The peak occurs at lag 0 or a negative lag, the normal, healthy case
- The peak magnitude is below `min_correlation`
- Fewer than `min_obs` overlapping observations at every lag
- The feature or target is constant over the overlapping subset
- The target has fewer than 2 distinct values

### Worked example

```python
import pandas as pd, numpy as np
from tsauditor.leakage.correlation import audit_correlation_leakage

rng = np.random.default_rng(7)
n   = 300
idx = pd.date_range("2024-01-01", periods=n, freq="D")
y   = pd.Series(rng.normal(0, 1, n), index=idx)

df = pd.DataFrame({
    "y":           y,
    "tomorrow_y":  y.shift(-1),   # tomorrow's target, available today, a leak
    "yesterday_y": y.shift(1),    # yesterday's target, perfectly legitimate
}, index=idx)

for issue in audit_correlation_leakage(df, target="y"):
    print(issue.column, issue.evidence)
```

```
tomorrow_y {'peak_lag': 1, 'peak_correlation': 1.0, 'min_correlation': 0.5, 'max_lag': 10, 'metric': 'spearman'}
```

`tomorrow_y` is caught at exactly lag +1 with correlation 1.0. `yesterday_y`, which peaks at lag −1, is correctly ignored, using yesterday's value today is not leakage, it is just a lag feature.

### Limitations and false positives

**This is a suspicion flag, not a proof, and the module says so in its own docstring.** In pure cross-correlation, a genuine strong predictor and a lookahead leak produce the same signature: a positive-lag peak. The only separator is magnitude. Real one-step-ahead predictive power in most domains is weak (|r| perhaps 0.05–0.2); leakage is strong (|r| above 0.5). Read `peak_correlation` and judge accordingly. This is why LEK002 is WARNING and not CRITICAL.

**Autocorrelated targets create false peaks, and the effect is large.** If the target is strongly persistent, a feature correlated with `target_t` is automatically correlated with `target_{t+1}` too, and noise can push the peak a step into the future. LEK003 exists precisely to control for this.

Measured over 100 trials per cell on 400-point series. Both columns are generated from *separate* draws, so every flag below is a false positive:

| `min_correlation` | Two random walks | Two AR(0.98) | Genuine leak, i.i.d. target | Genuine leak, random-walk target |
| ----------------- | ---------------- | ------------ | --------------------------- | -------------------------------- |
| 0.1 (before 0.3.1) | 37% | 51% | 100% | 100% |
| **0.5 (current default)** | **13%** | **8%** | **100%** | **100%** |

Raising the gate cost no true positive in 200 trials, which is why the default changed. But 13% and 8% are still not small. **If your columns are price levels or other near-random-walk series, treat a lone LEK002 flag as weak evidence and check `peak_correlation` yourself.** LEK003 on the same data false-positives at 3% and 15%, so where the two disagree, trust LEK003.

Why not require the positive-lag peak to beat the lag-0 correlation by a margin, as LEK003 effectively does? Because it does not work here. On a persistent target a genuine lookahead correlates with the target at lag 0 almost as strongly as at lag 1, so a flat margin suppresses real leaks along with false ones: a 0.10 margin cut false positives to 3% but dropped detection on a random-walk target from 100% to **0%**. LEK003 escapes this by dividing by the target's *measured* autocorrelation rather than subtracting a constant.

**Only ±10 lags by default.** A leak at lag +15 is invisible unless you raise `max_lag`.

**A feature that leaks at lag 0 *and* lag +1 may not be flagged.** If the lag-0 correlation is even marginally higher, the peak is at 0 and LEK002 stays quiet. LEK001 should catch that case.

---

## LEK003: Lookahead beyond persistence
### What it detects

A feature that correlates with the target's future **more strongly than the target's own persistence can explain**, the signature of a centered or forward-looking rolling window.

### The problem this solves

Here is why a naive "does it correlate with the future?" test is useless on time-series data.

Time-series targets are usually strongly autocorrelated. Today's price predicts tomorrow's price with correlation near 0.99, for no interesting reason, that is just what prices do.

Now take a perfectly honest trailing feature, say a 5-day trailing average. It tracks `target_t`. And `target_t` predicts `target_{t+1}` through sheer persistence. Therefore the honest feature correlates with `target_{t+1}` too, transitively, innocently.

A naive detector would flag every well-behaved feature you have.

### The solution

Control for persistence explicitly. The correlation a feature can reach with the future *legitimately* is bounded by its present association with the target, multiplied by the target's own autocorrelation:

```
expected(k) = |corr(feature_t, target_t)| × |corr(target_t, target_{t+k})|
observed(k) = |corr(feature_t, target_{t+k})|

excess(k)   = observed(k) − expected(k)
```

If `excess(k)` exceeds `excess_threshold` at any lag k in 1..`max_lag`, the feature knows the future better than persistence alone permits. Something other than persistence is carrying that information, most commonly a window that reaches forward.

All correlations are Spearman.

### Signature

```python
audit_temporal_leakage(
    df: pd.DataFrame,
    target: str,
    max_lag: int = 5,
    excess_threshold: float = 0.1,
    min_correlation: float = 0.1,
    min_obs: int = 30,
    domain: Optional[str] = None,
) -> List[Issue]
```

| Parameter | Type | Default | What it does |
| --------- | ---- | ------- | ------------ |
| `max_lag` | `int` | `5` | Forward lags examined (1..max_lag) |
| `excess_threshold` | `float` | `0.1` | How far above the persistence baseline counts as leakage |
| `min_correlation` | `float` | `0.1` | The observed future correlation must itself reach this |
| `min_obs` | `int` | `30` | Minimum overlapping observations |
| `domain` | `str` or `None` | `None` | No effect |

Both `excess_threshold` and `min_correlation` must be satisfied, so a large excess over a tiny baseline is not enough on its own.

**Implementation note.** The target's autocorrelations do not depend on any feature, so the shifted target series and their persistence correlations are computed once before the per-feature loop rather than inside it.

### Evidence

| Key | Meaning |
| --- | ------- |
| `lag` | The lag with the largest excess |
| `observed_future_corr` | Signed observed correlation at that lag |
| `excess_over_persistence` | `observed − expected` at that lag |
| `excess_threshold` | Threshold applied |
| `metric` | `"spearman"` |

### When it does not fire

- No lag produces an excess reaching `excess_threshold`
- The observed correlation is below `min_correlation`
- The present feature-target correlation cannot be computed
- Fewer than `min_obs` overlapping observations
- The feature or target is constant

### Worked example

```python
import pandas as pd, numpy as np
from tsauditor.leakage.temporal import audit_temporal_leakage

rng = np.random.default_rng(11)
n   = 400
idx = pd.date_range("2024-01-01", periods=n, freq="D")
y   = pd.Series(rng.normal(0, 1, n), index=idx)

df = pd.DataFrame({
    "y":             y,
    "roll_trailing": y.rolling(5).mean(),                # honest: past only
    "roll_centered": y.rolling(5, center=True).mean(),   # peeks 2 steps ahead
}, index=idx)

for issue in audit_temporal_leakage(df, target="y"):
    print(issue.column, issue.evidence)
```

```
roll_centered {'lag': 1, 'observed_future_corr': 0.4128, 'excess_over_persistence': 0.4094, 'excess_threshold': 0.1, 'metric': 'spearman'}
```

The centered window is caught; the trailing window is correctly ignored. The two features are otherwise identical, same input, same width, same aggregation. The only difference is `center=True`, and that is the entire bug.

### Why LEK002 misses this case

Run `audit_correlation_leakage` on that exact same DataFrame and it reports **nothing**. Here is why:

```
lag -3: -0.0098
lag -2: +0.4407   <- the peak is here, at a NEGATIVE lag
lag -1: +0.4164
lag +0: +0.4020
lag +1: +0.4128
lag +2: +0.4204
lag +3: -0.0249
```

A 5-wide centered window spans two steps back and two steps forward, so its correlation profile is roughly symmetric. The peak lands at lag −2, which looks perfectly innocent to LEK002, it only fires on a *positive*-lag peak.

But the correlation of +0.4128 at lag **+1** is far above what persistence explains, because white noise has essentially zero autocorrelation. LEK003 sees that excess and flags it.

This is the clearest possible demonstration of why both checks exist. They are not redundant; they catch different failures.

### Limitations and false positives

**It produces false positives, and you should expect them.** In testing, a 11-wide *trailing* window on a white-noise target was flagged with an excess of 0.10, right at the threshold. The persistence baseline is itself an estimate, and estimation noise can push an honest feature over the line. Treat a small excess (near 0.1) as weak evidence and a large one (above 0.3) as strong.

**Highly autocorrelated targets weaken the check.** When the target's persistence is near 1.0, `expected` is nearly as large as `observed` can be, so the excess is squeezed toward zero and real leaks can hide. On the random-walk version of the example above, a 5-wide centered window was **not** flagged, only wider ones were. Financial price levels are exactly this case; consider running the check on returns rather than levels.

**Only 5 forward lags by default.** A window reaching 10 steps ahead may not be caught at `max_lag=5`.

**The multiplicative baseline is a heuristic, not a theorem.** `|r0| × |persistence|` is a reasonable bound on transitive correlation, not a proven one. It can be too generous or too strict depending on the joint distribution.

---

## LEK004: As-of availability leakage
### What it detects

A value sitting at a timestamp **earlier than the moment it was actually published**. Every row before the real release date consumes information that did not yet exist.

### The problem

Many real-world series describe a *reference period* but become *available* later:

- **CPI** for January is published in mid-February
- **Company earnings** for Q1 arrive weeks after Q1 ends
- **Unemployment**, policy rates, GDP, all released on a delay
- **News sentiment scores** are often computed and backfilled retroactively

If such a value is aligned to its reference date and consumed at that timestamp, your backtest is trading on numbers nobody had yet.

### Why this check is opt-in

**This cannot be detected from the values alone.** Whether a CPI figure was knowable on January 31st depends entirely on when the statistics office published it, a fact that exists nowhere in your DataFrame. No amount of statistical cleverness can recover it.

So the check is explicit: you supply the availability information, and `tsauditor` verifies your data respects it. It never guesses release dates. If your `available_at` metadata is wrong, the check's conclusion is wrong. That responsibility is yours.

### Signature

```python
audit_asof_leakage(
    df: pd.DataFrame,
    available_at: Dict[str, Union[pd.Series, pd.Timedelta]],
    min_violations: int = 1,
) -> List[Issue]
```

| Parameter | Type | Default | What it does |
| --------- | ---- | ------- | ------------ |
| `df` | `pd.DataFrame` | required | Must have a `DatetimeIndex`: the decision times |
| `available_at` | `dict` | required | Per-column availability. Unlisted columns are not checked. |
| `min_violations` | `int` | `1` | Rows required before raising. Default 1: one confirmed lookahead is real leakage. |

### Two ways to declare availability

**A `pd.Timedelta`, a fixed publication lag.**

```python
available_at={"cpi": pd.Timedelta(days=30)}
```

Availability becomes `row_timestamp + lag` for every row. Because the lag is positive and uniform, **every** row is a violation, which is the point. It catches the classic "I forgot to shift the macro series" bug in one line.

**A `pd.Series`, real per-row publish timestamps.**

```python
available_at={"cpi": actual_release_dates}   # must be indexed by df.index
```

This is the general and correct form. Real release schedules are ragged, CPI is not published exactly 30 days later every month, and a Series captures that. Rows where the value was genuinely available produce no violation.

If the Series does not align with `df.index` at all, you get an explicit error rather than a silently empty result.

### How it works

```
violation  =  value is not NaN
          AND availability timestamp is known
          AND available_at > row timestamp
```

Then it reports the count, the maximum lookahead in days, and the earliest offending row timestamp.

### Evidence

| Key | Meaning |
| --- | ------- |
| `n_violations` | Rows using a value before it existed |
| `max_lookahead_days` | Largest gap between availability and row timestamp, in days (3dp) |
| `first_violation` | Earliest offending row timestamp |
| `check` | `"as-of"` |

### When it does not fire

- The column is not listed in `available_at`
- Every value was available at or before its row timestamp
- Violations number fewer than `min_violations`
- All values in the column are NaN

Raises `ValueError` if the index is not a `DatetimeIndex`, if a listed column does not exist, if a Series fails to align, or if the spec is neither a Series nor a Timedelta.

### Worked example

```python
import pandas as pd, numpy as np
from tsauditor.leakage.asof import audit_asof_leakage

rng = np.random.default_rng(7)
idx = pd.date_range("2024-01-01", periods=300, freq="D")
df  = pd.DataFrame({"cpi": rng.normal(3, 0.2, 300)}, index=idx)

# CPI is published about 30 days after the period it describes
for issue in audit_asof_leakage(df, available_at={"cpi": pd.Timedelta(days=30)}):
    print(issue.column, issue.evidence)
```

```
cpi {'n_violations': 300, 'max_lookahead_days': 30.0, 'first_violation': '2024-01-01 00:00:00', 'check': 'as-of'}
```

All 300 rows are violations, because the column sits at reference dates while the values were not available until 30 days later. The fix is to shift the column forward to its release schedule, not to drop it. The data is fine; the alignment is wrong.

### Limitations and false positives

**It is only as correct as your metadata.** Wrong `available_at`, wrong answer. There is no way around this.

**A fixed `Timedelta` is a blunt instrument.** It flags every row identically and cannot represent a real, irregular release calendar. Use it as a quick smoke test; use a Series for anything serious.

**Columns you do not list are never checked.** Silence from LEK004 means nothing unless you declared the columns.

**Revisions are not modeled.** Many macro series are published, then revised weeks later. This check verifies first availability only; it has no concept of a value changing after release.

---

## LEK005: Combination leakage
### What it detects

A **group of two or three** features that together reconstruct the target, while none does alone, additively (sums, differences) or multiplicatively (products, ratios).

### The problem

Every check above is univariate: each feature is scored against the target on its own. That misses a whole class of real bug:

```python
target = (high + low) / 2
target = revenue - costs
target = price * quantity
target = numerator / denominator
target = a + b + c
```

No input is near-deterministic by itself, so LEK001 stays silent, but the group rebuilds the target exactly.

The canonical shape is a **difference**. If `x1` and `x2` are independent with similar variance and `target = x1 - x2`, each correlates with the target at only about 0.7, far below LEK001's 0.95 threshold, while the pair explains it perfectly.

**This is not hypothetical.** In the OGDC file, `MACD_hist` is exactly `MACD - MACD_signal` (verified to 3e-15). Point `target="MACD_hist"` at tsauditor and LEK005 reports an adjusted R² of 1.0000 for that pair, where the best *single* column reaches only 0.12.

### Signature

```python
audit_combination_leakage(
    df: pd.DataFrame,
    target: str,
    threshold: float = 0.95,
    min_obs: int = 30,
    max_features: Optional[int] = None,
    max_reported: int = 10,
    max_group_size: int = 3,
    gate: float = 0.30,
    max_candidates_per_level: int = 200,
    domain: Optional[str] = None,
) -> List[Issue]
```

| Parameter | Default | What it does |
| --------- | ------- | ------------ |
| `threshold` | `0.95` | Adjusted R² at or above which a group is flagged. Matches LEK001. |
| `min_obs` | `30` | Minimum complete rows to score a group |
| `max_features` | `None` | Cap on features considered. The pair scan is O(k²): ~0.27s for 100 features. |
| `max_reported` | `10` | Cap on findings, best first, so a family of derived columns can't flood the report |
| `max_group_size` | `3` | Largest group searched. `2` = pairs only; raise to `4`+ for larger identities. |
| `gate` | `0.30` | Adjusted R² a group must reach before it is extended by another column |
| `max_candidates_per_level` | `200` | Cap on groups carried to the next level, best first: bounds cost |

### How it works

For a candidate group, fit `target ~ 1 + columns` by ordinary least squares and take the **adjusted** R². Two algebraic forms are tried:

- **linear**, catches sums, differences and weighted combinations
- **log**, the same fit on `log` of the **absolute** target and columns, which catches **products and ratios**, since `log(a*b) = log a + log b`. Absolute values mean *signed* products and ratios are recovered too (`|a*b| = |a|*|b|`). Skipped when any value touches zero, since `log` of a near-zero would dominate the fit.

Measured coverage (adjusted R², n=500):

| target | linear | log |
| ------ | ------ | --- |
| `x1 - x2` | **1.0000** | 0.0112 |
| `x1 * x2` | 0.9287 | **1.0000** |
| `x1 / x2` | 0.8304 | **1.0000** |
| unrelated control | −0.0026 | −0.0038 |

Neither form alone suffices. An interaction term (`x_i * x_j` as a third predictor) was tested as an alternative and **rejected**: it catches products but not ratios (0.897), and roughly doubles the chance-level R².

The form used is reported in `evidence["form"]`.

**Why plain OLS, not the rank methods used everywhere else in this module.** The leakage here is arithmetic, and ranking destroys it. On the canonical `target = x1 - x2` case, raw adjusted R² is 1.0000 while the rank-transformed version scores **0.9410**, under the threshold, missing the leak entirely. Rank methods answer "is this monotonic?"; the question here is "do these columns reconstruct the target?"

Adjusted rather than raw R² is used because it penalises extra predictors, keeping the null distribution tight across many candidate groups.

`lstsq` (pseudo-inverse) is used rather than a normal-equation solve because candidate groups are frequently collinear, `high`/`low`, a level and its lag, which would make `X'X` singular.

### Larger groups, without paying O(k^n)

Scanning every triple would be C(k,3) fits, 161,700 for 100 features, and quadruples are worse. Both would badly inflate the multiple-comparison problem.

Instead the search **deepens iteratively**. A group is extended by one more column only when it reaches `gate` without reaching the flagging threshold, that is exactly the signature of a sub-group of a larger identity:

| true identity | best sub-group score | clears the 0.30 gate? |
| ------------- | -------------------- | --------------------- |
| `a + b + c` | pair scores ~0.71 | yes |
| `a + b + c + d` | pair ~0.49, triple ~0.74 | yes |

**On random data nothing clears the gate at all**, so deeper levels cost nothing and add no false positives. The gate was verified not to block genuine identities across equal, moderately unequal, cancelling and collinear component shapes.

`max_group_size` defaults to **3**. Raise it to find larger identities:

```python
audit_combination_leakage(df, target="y", max_group_size=4)
```

It is not the default because each level is free on *clean* data but costs time on frames where many features partially explain the target. `max_candidates_per_level` (default 200) bounds that: without it, 40 mutually correlated features took **21s** at depth 4; with it, **0.7s**. The trade-off is that in such a frame a genuine identity outside the top 200 sub-groups could be missed.

A group is not reported when a subset of it was already reported, that would be the same finding with a redundant column attached.

### The single-feature guard

A pair is **skipped when either column alone already reaches the threshold.**

Without this, one leaky column poisons the whole report: if `leak` alone reproduces the target, then every pair containing `leak` also hits R² 1.0, producing k−1 findings for something LEK001 already reported once. That case belongs to LEK001; LEK005 is only for leakage that emerges from the *combination*.

### When it does not fire

- Either column alone already explains the target (LEK001's job)
- Fewer than `min_obs` complete rows for the pair
- The target is constant, non-numeric, or absent
- Fewer than two usable numeric features
- Either column is constant

### Worked example

```python
import pandas as pd, numpy as np
from tsauditor.leakage.combination import audit_combination_leakage

rng = np.random.default_rng(0)
n = 400
x1 = rng.normal(0, 1, n)
x2 = rng.normal(0, 1, n)

df = pd.DataFrame({
    "target": x1 - x2,          # neither input determines this alone
    "x1": x1,
    "x2": x2,
    "noise": rng.normal(0, 1, n),
}, index=pd.date_range("2024-01-01", periods=n, freq="D"))

for issue in audit_combination_leakage(df, target="target"):
    print(issue.evidence)
```

```
{'metric': 'adjusted_r2', 'form': 'linear', 'group': ['x1', 'x2'], 'group_size': 2,
 'group_adjusted_r2': 1.0, 'best_single_adjusted_r2': 0.4613,
 'threshold': 0.95, 'n_obs': 400}
```

`best_single_adjusted_r2` of 0.46 is the key number: neither column comes close on its own, which is exactly why every other leakage check missed it.

The same call catches a product, a ratio, and a three-column identity:

```python
a = rng.uniform(2, 10, n)      # strictly positive, so the log form applies
b = rng.uniform(2, 10, n)
c = rng.uniform(2, 10, n)
```

| target | reported group | `form` | adjusted R² |
| ------ | -------------- | ------ | ----------- |
| `a - b` | `['a', 'b']` | linear | 1.0 |
| `a * b` | `['a', 'b']` | **log** | 1.0 |
| `a / b` | `['a', 'b']` | **log** | 1.0 |
| `a + b + c` | `['a', 'b', 'c']` | linear | 1.0 |
| unrelated control |: |: | no finding |

### Limitations and false positives

**Search depth is capped by `max_group_size` (default 3).** Larger identities are found by raising it, at some cost on frames with many partially-explanatory features.

**False-positive profile is good, and was measured.** On random targets with independent random features, the largest adjusted R² reached by chance was **0.075** for pairs (50 features, 1,225 pairs, 100 rows), typically below 0.03; the log form behaves the same (max 0.028). No triple is ever evaluated on random data, because no pair clears the gate. Innocent but highly collinear pairs (r ≈ 0.96) score ~0.00 against an unrelated target. A real 24-column financial dataset produced zero findings for its actual target.

**Additive and multiplicative only.** Between the linear and log forms this covers sums, differences, products and ratios, signed or not. Genuinely non-monotonic constructions, `target = x1² + sin(x2)`, are not found, and would require either a fitted model or a guessed basis expansion; both would cost the library its "no model, no hyperparameters" property.

**The log form needs values away from zero.** Signed data is fine, absolute values are used, but a column containing an exact zero falls back to the linear form, so a product involving it may be missed.

**Cost is O(k²) for the pair scan.** About 0.27s for 100 features, 0.07s for 50. With several hundred columns, set `max_features`. Triples add essentially nothing on clean data.

**It flags relationships, not culprits.** LEK005 tells you the target is reconstructible from a group of columns. Whether that is leakage or a legitimate identity you already knew about is your call.

**A dominated group is deferred to LEK001.** `target = 100a + b + c` looks like a three-way identity, but `a` alone explains 99.98% of it. The single-feature guard makes LEK005 stay silent so LEK001 owns that finding.

---

## Choosing which check to trust

| Code | Confidence | How to read it |
| ---- | ---------- | -------------- |
| LEK001 | **Very high** | An AUC of 1.0 or ρ of 0.99 is not a judgement call. Investigate immediately. |
| LEK005 | **Very high** | An adjusted R² near 1.0 with a low best-single score is an arithmetic identity, not a coincidence. |
| LEK004 | **High**, conditional on your metadata | The logic is deterministic; only your `available_at` can be wrong. |
| LEK002 | **Medium** | Read `peak_correlation`. Above 0.7 is alarming; near the 0.5 gate is weak evidence, especially on price-level columns. |
| LEK003 | **Medium** | Read `excess_over_persistence`. Above 0.3 is strong; near 0.1 may be estimation noise. |

**Remaining gaps.** LEK001–LEK004 are univariate. LEK005 covers groups of two or three by default (raise `max_group_size` for more), additive and multiplicative, signed or not, but not non-monotonic constructions like `x1² + sin(x2)`. For panel data, [PNL002](Panel-Data#pnl002-cross-sectional-lookahead) covers leaks living between entities. `tsauditor` reduces your risk substantially; it does not eliminate it.
