# Prior art: how other libraries detect time-series outliers

Research notes for deciding what, if anything, to change in ANO002.

**Everything in the benchmark section was measured, not quoted.** The script is
`docs/research/outlier_prior_art_benchmark.py`.

---

## The landscape

Two groups of libraries matter here, and they are not the same group.

### Time-series anomaly detection

| Library | Approach | Notes |
| ------- | -------- | ----- |
| **ADTK** (Arundo) | Transformer → detector pipeline. `QuantileAD`, `InterQuartileRangeAD`, `GeneralizedESDTestAD`, `LevelShiftAD`, `PersistAD` | Closest in philosophy to tsauditor: rule-based, no model fitting, interpretable. **Effectively unmaintained** — last release 0.6.2 in 2020, no stated Python 3.12/3.13 support. |
| **Darts** (Unit8) | Anomaly *scorers* + detectors + aggregators. Wraps any forecaster: score = prediction error. `PyODScorer` bridges to PyOD. | Model-based. Powerful, much heavier. |
| **Merlion** (Salesforce) | Model ensembles with automatic post-processing/calibration | Heavy; aimed at production monitoring. |
| **sktime** | `sktime.detection` module, scikit-learn-style API | General framework, detection is one task among many. |
| **PyOD** | 50+ outlier algorithms, mostly not time-aware | Tabular-first; used via Darts for TS. |

### Data-quality / validation — tsauditor's actual peer group

| Library | What it does | Time-aware? | Leakage? |
| ------- | ------------ | ----------- | -------- |
| **Great Expectations** | Declarative expectations, docs, profiling | No | No |
| **pandera** | Schema validation, runtime dataframe type-checking | No | No |
| **ydata-profiling** | EDA reports; *does* have a time-series mode with ACF/PACF and seasonality | Partly | No |
| **Evidently** | Drift and model monitoring | Partly | No |
| **Deepchecks** | Validation suites, includes some train/test contamination checks | Partly | Partial |

### Leakage detection specifically

The existing tools are **static code analysis**, not data analysis:

- **LeakageDetector** (PyCharm plugin / VS Code extension) — reads your *source code* and flags patterns like scaling before `train_test_split`
- **leakage-analysis** — static analysis of notebooks
- **leakr** (R) — train/test contamination, duplicate rows

**None of them look at the values to ask "does this feature reproduce the target?"**

This is worth being clear about: tsauditor's niche is not "another anomaly detector."
It is *data-driven* leakage detection for time series, which is close to unoccupied.
The anomaly checks are supporting cast. That should inform how much effort ANO002
deserves relative to the LEK/PNL family.

---

## The technique that matters: Generalized ESD

Rosner (1983), "Percentage Points for a Generalized ESD Many-Outlier Procedure",
*Technometrics* 25(2). It is the classical, purpose-built answer to exactly the
masking problem documented in [Detectors-Anomaly](../../wiki/Detectors-Anomaly.md).

**How it works.** Grubbs' test finds one outlier. Its weakness is that a second
outlier inflates the standard deviation and hides the first. Generalized ESD fixes
this by iterating:

1. Compute the mean and standard deviation
2. Find the most extreme point, record `R_i = |x - mean| / std`
3. **Remove it**, recompute mean and standard deviation on what remains
4. Repeat up to `k` times
5. Compare each `R_i` against a critical value from the t-distribution; the
   largest `i` where `R_i > lambda_i` is the number of outliers

Because the scale is recomputed *after each removal*, the contaminating points
progressively stop inflating it. Masking cannot occur by construction.

`max_outliers` (k) is an upper bound you must supply, and it is the method's main
practical constraint: it can never report more than k, so k caps sensitivity while
also bounding cost.

---

## Benchmark against tsauditor's current rule

1,000 points of N(0,1), outliers planted at 10σ. `tsa z` and `tsa iqr` are read
from the real `audit_point_anomalies` evidence.

| planted | tsa z-score | tsa IQR | **Generalized ESD** |
| ------- | ----------- | ------- | ------------------- |
| 0 | 0 | **10** (false positives) | **0** |
| 1 | 1 | 11 | **1 exact** |
| 5 | 5 | 15 | **5 exact** |
| 20 | 20 | 29 | **20 exact** |
| 50 | **0 (blind)** | 57 | **50 exact** |
| 150 | **0 (blind)** | 151 | **150 exact** |
| 300 | **0** | **0 — no issue raised** | **300 exact** |

ESD is exact at every level, gives zero false positives on clean Gaussian data,
and works at 30% contamination where tsauditor's rule reports nothing at all.

### Where ESD is weaker — clean data, 0 true outliers

| data | ESD flags | tsa IQR flags |
| ---- | --------- | ------------- |
| gaussian | 0 | 10 |
| **lognormal (skewed)** | **29** | 71 |
| **exponential (skewed)** | **7** | 41 |
| linear trend | 0 | 0 |
| random walk | 0 | 5 |
| seasonal | 0 | 0 |

ESD assumes approximate normality, so it over-flags skewed data — but still
**less than the current IQR rule does** (29 vs 71, 7 vs 41).

### Two results that contradicted my expectations

**ESD does not flag trends.** Linear trend, random walk and seasonal data all give
0. A trend inflates the standard deviation uniformly, so no individual point stands
out. This makes ESD *more* trend-tolerant than a median/MAD robust z-score, which
flags 85 points on the OGDC `Price` column purely because of its uptrend.

**STL detrending made things worse, not better.** The obvious "decompose first,
then detect on residuals" recipe:

| data | raw ESD | STL-residual ESD |
| ---- | ------- | ---------------- |
| linear trend | 0 | 6 |
| random walk | 0 | **39** |
| seasonal | 0 | 1 |

On a random walk, STL residuals are not meaningful and detection on them
manufactures outliers. Detrending is **not** universally correct, contrary to the
common advice.

### Cost

O(k·n). n=5,000 with k=2,000 took 0.2s — acceptable, but k must be capped
(a fraction of n, or an absolute limit) or it becomes the new hot spot next to ADF.

---

## What this suggests for tsauditor

**Do not adopt median/MAD as the primary scale.** It flags trends (85 false
positives on OGDC `Price`) and ESD dominates it on every axis tested.

**Generalized ESD is the strongest candidate** if ANO002 is ever revisited. It is
strictly better than the current rule on contamination, better on clean Gaussian
data, and better on skewed data. Its weaknesses (skew, needing a k cap) are real
but smaller than what it replaces.

**A cautious path**, if pursued: add ESD as a *third* signal alongside z-score and
IQR, reported in evidence, before considering any change to flagging. That directly
fixes the "zero `agreement_count` is ambiguous" problem currently documented as a
limitation, without altering behaviour.

**But weigh the priority honestly.** No other library does data-driven leakage
detection; several do anomaly detection well. Effort spent on LEK/PNL differentiates
tsauditor. Effort spent on ANO002 makes it marginally better at something ADTK,
Darts and PyOD already do — and ADTK, the closest philosophical match, is
unmaintained, which is an argument that this niche is not where the demand is.

---

## Sources

- [ADTK documentation](https://adtk.readthedocs.io/en/stable/) and [detector API](https://adtk.readthedocs.io/en/stable/api/detectors.html)
- [ADTK on PyPI](https://pypi.org/project/adtk/) — version 0.6.2, uploaded 2020
- [Darts documentation](https://unit8co.github.io/darts/) — `darts.ad` module
- [sktime detection API](https://www.sktime.net/en/latest/api_reference/detection.html)
- [Rosner's Test for Outliers (R EnvStats)](https://search.r-project.org/CRAN/refmans/EnvStats/html/rosnerTest.html)
- [Generalized ESD explained](https://brandonpipher.com/post/2025-08-21-outlier-testing-rosner/)
- [The Hampel identifier — SAS blog](https://blogs.sas.com/content/iml/2021/06/01/hampel-filter-robust-outliers.html)
- [LeakageDetector: static analysis tool](https://arxiv.org/abs/2503.14723)
- [awesome-TS-anomaly-detection](https://github.com/rob-med/awesome-TS-anomaly-detection)
- [Pandera vs Great Expectations](https://endjin.com/blog/a-look-into-pandera-and-great-expectations-for-data-validation)
