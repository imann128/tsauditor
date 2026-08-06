# OGDC Case Study

This is the leak `tsauditor` was written to catch. Every number on this page comes from running the code against the real dataset in `examples/ogdc_leakage_case/`.

---

## The situation

A model predicting the daily price direction of OGDC (Oil & Gas Development Company, a Pakistani equity) reached **99.68% accuracy** with a Random Forest.

Stock direction prediction is close to a coin flip. Published results in the 55–65% range are considered good. A number near 100% does not mean you have solved the market; it means something in your pipeline is broken.

---

## The data

```python
import pandas as pd

df = pd.read_csv("ogdc_with_regimes.csv", index_col="Date", parse_dates=True)
df = df.dropna(subset=["Direction"])

print(df[["Price", "ChangeP", "Returns", "Direction"]].head(3))
```

```
             Price  ChangeP   Returns  Direction
Date
2020-01-22  143.89    -3.37 -3.371164          0
2020-01-23  143.05    -0.58 -0.583779          0
2020-01-24  143.71     0.46  0.461377          1
```

1,537 rows, 24 columns, daily, January 2020 through April 2026.

Look at those three rows and the leak is visible once you know to look for it:

| `ChangeP` | `Direction` |
| --------- | ----------- |
| −3.37 | 0 |
| −0.58 | 0 |
| +0.46 | 1 |

`Direction` is **defined as** `1 if ChangeP > 0 else 0`. The target is a function of the feature. `Returns` is the same quantity computed with more decimal places, so it leaks identically.

A model given `ChangeP` is not predicting tomorrow's direction. It is reading today's answer sheet.

---

## Why this survived review

Three reasons, all of which generalise well beyond this dataset.

**The feature name gives nothing away.** `ChangeP` sounds like an ordinary technical feature, sitting in a column list beside `RSI`, `MACD`, `Return_lag1`, and `Regime`. Nothing marks it out.

**The column looks completely normal.** No missing values, no absurd outliers, a sensible distribution. Any profiling tool would report it as clean, because by every conventional measure it *is* clean.

**Correlation looked reassuring.** This is the part that matters most:

```python
from scipy.stats import pearsonr

for col in ["ChangeP", "Returns", "RSI", "MACD"]:
    pair = df[[col, "Direction"]].dropna()
    r, _ = pearsonr(pair[col], pair["Direction"])
    print(f"{col:10s} pearson = {r:+.4f}")
```

```
ChangeP    pearson = +0.6268
Returns    pearson = +0.6267
RSI        pearson = +0.2857
MACD       pearson = +0.0571
```

**0.63.** A feature that *mathematically determines* the target correlates with it at 0.63.

That is a moderate correlation. It is the kind of number you would see from a decent technical indicator. Any leakage check built on a Pearson threshold (0.95, 0.9, even 0.8) waves this straight through. And 0.63 is *higher* than RSI's 0.29, so it does not even stand out as anomalous within its own feature set.

This is not a subtle statistical edge case. It is a hard mathematical ceiling.

---

## The ceiling

When you compute Pearson correlation between a continuous variable and a **binary** 0/1 variable, you get the *point-biserial* correlation. Its maximum possible value is approximately:

```
√(2/π) ≈ 0.798
```

**even when the relationship is perfectly deterministic.**

The intuition: Pearson measures how well a *straight line* fits the relationship. But the true relationship here is a step function: everything below zero maps to 0, everything above maps to 1. No straight line fits a step function well. Pearson is measuring the wrong thing, and its answer is capped by the geometry of the question, not by the strength of the relationship.

So a Pearson-based leakage detector cannot work on binary targets. Not "works poorly": cannot work. The signal it needs is mathematically unavailable to it.

---

## What tsauditor does instead

For binary targets, LEK001 uses **AUC separation** via the Mann-Whitney rank statistic.

AUC asks a different question: *if I pick one random day where Direction was 1 and one where it was 0, how often does the feature rank the up-day higher?*

- 0.5: no separation. Useless feature.
- 1.0: perfect separation. Every up-day outranks every down-day.

This question has no ceiling, because it is about *ordering*, not linear fit. A step function separates the classes perfectly, and AUC reports exactly that.

```python
import tsauditor as tsa

report = tsa.scan(df, target="Direction", domain="finance")

print("leaky columns:", report.leaky_columns())
for issue in report.filter(code="LEK001"):
    print(issue.column, issue.evidence)
```

```
leaky columns: ['ChangeP', 'Returns']

ChangeP {'metric': 'auc', 'auc': 1.0, 'separation': 1.0, 'threshold': 0.95, 'target_type': 'binary', 'n_obs': 1537}
Returns {'metric': 'auc', 'auc': 1.0, 'separation': 1.0, 'threshold': 0.95, 'target_type': 'binary', 'n_obs': 1537}
```

**AUC = 1.0. Exactly perfect separation, on all 1,537 rows.**

Side by side:

| Metric | `ChangeP` vs `Direction` | Verdict at a 0.95 threshold |
| ------ | ------------------------ | --------------------------- |
| Pearson | 0.6268 | Clean, ship it |
| **AUC separation** | **1.0000** | **CRITICAL leak** |

Same data, same relationship, opposite conclusions. The choice of metric *is* the whole detector.

Note also that both leaking columns were caught. `Returns` is `ChangeP` at higher precision, and a check that stopped at the first finding would have left it in place.

---

## The full report

```python
report = tsa.scan(df, target="Direction", domain="finance", run_stationarity=False)

print("critical:", len(report.critical))
print("warnings:", len(report.warnings))
print("health  :", report.health_score(df))
```

```
critical: 2
warnings: 39
health  : 82.5
```

Findings by code:

```
LEK001 ×  2    the two leaking columns
ANO002 × 19    point outliers across the technical indicators
ANO003 × 15    contextual spikes
ANO001 ×  4    stuck values
PRF001 ×  1    one large timestamp gap
```

Two observations worth drawing out.

**The 38 anomaly warnings are mostly not problems.** Financial return series have fat tails; ANO002 and ANO003 fire on real market moves. `domain="finance"` already loosens these thresholds and they still fire. This is the expected cost of a warning-level check on financial data; see [Anomaly Detectors](Detectors-Anomaly#limitations-and-false-positives).

**Health is 82.5%, and the leak contributes nothing to that number.** The score measures cell-level corruption only. The two columns that would destroy this model are perfectly clean data; they are simply the wrong data to have. This is exactly why leakage is excluded from the health score: a single number cannot meaningfully combine "3% of cells are odd" with "your target is in your feature matrix."

**Do not read a good health score as a clean bill of health.** 82.5% with two CRITICAL leaks is a dataset that will produce a worthless model.

---

## The cost of the leak

`examples/ogdc_leakage_case/compare_leakage.py` trains the same models with and without the leaking features. Results, from `leakage_comparison_results.csv`:

| Model | Features | Accuracy | AUC |
| ----- | -------- | -------- | --- |
| Random Forest | with leakage | **99.68%** | 0.9987 |
| Random Forest | realistic | **69.81%** | 0.7795 |
| Gradient Boosting | with leakage | **99.68%** | 0.9967 |
| Gradient Boosting | realistic | **73.70%** | 0.8072 |

Removing two columns costs roughly **30 percentage points** of accuracy.

The honest reading: 69.81% and 73.70% are *good* results for daily equity direction. The real model was never bad. It was a decent model wearing a fraudulent number.

And that is the dangerous shape of this bug. It does not break anything. It makes everything look wonderful. A bug that degrades your results gets found on Monday; a bug that inflates them gets presented to stakeholders.

---

## What the fix actually is

`tsauditor` does not remove the columns for you. It hands you the shortlist:

```python
report.leaky_columns()
```

```
['ChangeP', 'Returns']
```

For this dataset the remedy is to drop both and train on the genuinely predictive features that remain: the lagged returns, rolling statistics, RSI, and MACD, all of which are computed from *past* data only.

If you want `tsauditor` to drop them:

```python
clean = report.apply_fixes(df, leakage="drop")
```

This is opt-in and off by default, because removing columns changes your feature matrix and should always be a deliberate act.

Note that the correct remedy is not always deletion. For [LEK004](Detectors-Leakage#lek004-as-of-availability-leakage) (a value published after the timestamp it sits at) the fix is to *shift* the column to its release schedule. The data is fine; the alignment is wrong. Deleting it throws away a legitimate feature.

---

## Generalising from this

**Suspicion should scale with your score.** Above ~85% on daily equity direction, or above ~95% on any hard problem, treat the result as a bug report until proven otherwise. Investigating a great result is not pessimism; it is the only way to find this class of error.

**Check how your target was constructed.** This leak came from a target *defined in terms of* a feature that stayed in the matrix. Whenever the target is derived (a binarized return, a thresholded value, a lagged difference), trace every input to that definition and make sure none of them survive as features.

**Do not use Pearson to detect leakage against a binary target.** The 0.798 ceiling makes it structurally incapable of the job. Use a rank or separation metric.

**Check every feature, including the ones you trust.** `Returns` and `ChangeP` were the same quantity twice. Duplicated-under-another-name is common in feature sets that grew over time.

---

## Running it yourself

```bash
cd examples/ogdc_leakage_case
jupyter notebook ogdc_leakage.ipynb    # detection walkthrough
python compare_leakage.py              # the model comparison (needs scikit-learn)
```

The notebook focuses on detection; `compare_leakage.py` reproduces the accuracy table above.

→ [Leakage Detectors](Detectors-Leakage) for the full mechanics of all four checks.
