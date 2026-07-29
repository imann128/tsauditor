# Issue Code Reference

Every finding `tsauditor` raises carries a short code. This page is the lookup table: what each code means, what fires it, and where to read the details.

```python
report.filter(code="LEK001")
```

---

## All codes at a glance

| Code | Severity | Module | Detects | Repaired by `apply_fixes`? | Counts against health score? |
| ---- | -------- | ------ | ------- | -------------------------- | ---------------------------- |
| PRF001 | WARNING | profiler | Large timestamp gap | No | No |
| PRF002 | WARNING | profiler | Clustered missing values | Yes — impute | Yes |
| PRF003 | INFO | profiler | Non-stationary column | No | No |
| PRF004 | CRITICAL | profiler | Duplicate timestamps | No | No |
| PRF005 | WARNING | profiler | Clustered gaps | No | No |
| PRF006 | WARNING | profiler | High missing rate | Yes — impute | Yes |
| PRF007 | CRITICAL | profiler | Infinite values (`inf` / `-inf`) | Yes — NaN then impute | Yes |
| ANO001 | WARNING | anomaly | Stuck / frozen values | Yes — NaN then impute | Yes |
| ANO002 | WARNING | anomaly | Point outliers (global) | Yes — clip or NaN | Yes |
| ANO003 | WARNING | anomaly | Contextual spikes (local) | Yes — clip or NaN | Yes |
| LEK001 | CRITICAL | leakage | Feature reproduces the target | Opt-in — drop column | No |
| LEK002 | WARNING | leakage | Peak correlation at a positive lag | Opt-in — drop column | No |
| LEK003 | WARNING | leakage | Lookahead beyond persistence | Opt-in — drop column | No |
| LEK004 | CRITICAL | leakage | Value used before it was published | Opt-in — drop column | No |
| LEK005 | CRITICAL | leakage | A **group** of features reconstructs the target | Opt-in — drop column | No |
| VAL001 | WARNING | validity | Value outside a declared bound | No | No |
| VAL002 | CRITICAL | validity | Broken ordering relation | No | No |
| PNL001 | WARNING | panel | Entities don't share a common time index | No | No |
| PNL002 | WARNING | panel | Cross-sectional feature knows the future ordering | No | No |
| PNL003 | INFO | panel | Entity too short to audit meaningfully | No | No |
| PNL004 | WARNING | panel | Rows with a null entity id get no checks | No | No |

Two columns worth noticing.

**Leakage never counts against the health score.** A leaky column contains perfectly valid data; it is a modeling risk, not corruption. Mixing the two would make the score uninterpretable.

**Several codes are never repaired.** Index problems, non-stationarity, and validity issues have no safe default remedy — see [Remediation](Remediation#what-gets-repaired-and-what-does-not).

---

## Which codes need opt-in

Most checks run automatically. These need an argument.

| Code | Requires |
| ---- | -------- |
| LEK001, LEK002, LEK003, LEK005 | `scan(target="...")` — silently skipped without it |
| LEK004 | `scan(available_at={...})` |
| VAL001, VAL002 | `scan(constraints={...})` |
| PNL001, PNL003, PNL004 | `scan(group_col="...")` |
| PNL002 | `scan(group_col="...", target="...")` |

Silence from these codes means nothing unless you supplied the argument.

---

## Profiler codes (PRF)

Structural problems with the data's shape. → [Profiler Detectors](Detectors-Profiler)

### PRF001 — Large gap

**WARNING**, dataset-level (`column` is `None`).

A gap between consecutive timestamps exceeds the threshold: **5.0 days** fixed under `domain="finance"`, otherwise **3 × the median gap**.

*Evidence:* `gap_count`, `maximum_gap_days`, `locations`

*What to do:* Resample to a regular frequency, or document why the timestamps are irregular. Under `domain="finance"` this should be rare — weekends already pass.

### PRF002 — Clustered missing values

**WARNING**, per column.

A run of consecutive NaNs reaches the cluster threshold: **5** under `"finance"`, **3** otherwise.

*Evidence:* `missing_percentage`, `longest_consecutive_run`, `cluster_count`, `first_occurrence`, `cluster_threshold`

*What to do:* A clustered outage is not the same as scattered missingness. Interpolating across a long block fabricates data. Consider dropping the affected span, or check whether the missingness itself is informative.

### PRF003 — Non-stationary column

**INFO**, per column.

The Augmented Dickey-Fuller test returned p > 0.05, so the null hypothesis of a unit root cannot be rejected.

*Evidence:* `adf_statistic`, `p_value`, `n_observations`, `alpha`

*What to do:* Usually nothing. A price series *should* be non-stationary. If you are fitting a model that assumes stationarity, difference the series first. This is INFO precisely because it is a modeling note, not a defect.

### PRF004 — Duplicate timestamps

**CRITICAL**, dataset-level.

The index contains repeated timestamps.

*Evidence:* `duplicate_count` (rows involved, not excess rows), `examples`

*What to do:* Resolve before anything else. Duplicates silently corrupt every rolling window, lag, and resample operation without raising a single error. Decide whether to keep first, keep last, or aggregate — `tsauditor` will not choose for you.

### PRF005 — Clustered gaps

**WARNING**, dataset-level.

Two or more consecutive large gaps.

*Evidence:* `cluster_count`, `max_consecutive_gaps`, `cluster_start_locations`

*What to do:* One large gap is usually a holiday. Several in a row means a feed outage or a change in sampling regime — a structural event worth understanding.

### PRF006 — High missing rate

**WARNING**, per column.

A column is at least **30%** missing overall (`missing_rate_threshold`).

*Evidence:* `missing_count`, `missing_percentage`, `threshold_percentage`

*What to do:* Consider dropping the column. If you impute, be aware you are inventing a third of it. Check whether the missingness is informative — sometimes "no value" is itself a signal.

### PRF007 — Infinite values

**CRITICAL**, per column.

The column contains `inf` or `-inf`. There is **no threshold**: one is a defect, because an infinity is never a measurement, only the residue of a division by zero, an overflow, or a log of a non-positive number.

*Evidence:* `non_finite_count`, `positive_inf_count`, `negative_inf_count`, `non_finite_percentage`, `n_finite_remaining`, `below_leakage_min_obs`, `leakage_min_obs`, `first_occurrence`

*What to do:* Fix the upstream computation, do not impute. **Read `below_leakage_min_obs` first** — if true, fewer than 30 finite observations remain and LEK001, LEK002, LEK003 and LEK005 skipped the column silently, so it has not been checked for leakage at all.

The sign split is diagnostic. All one sign usually means a division by zero or an overflow in one direction; both signs more often means a ratio whose denominator crosses zero, which is a different bug upstream.

*Why CRITICAL:* like PRF004, it invalidates other checks rather than describing the data. A single infinity makes the column's mean `inf` and its standard deviation `NaN`, and `scikit-learn` raises at `fit` time.

---

## Anomaly codes (ANO)

Implausible values. → [Anomaly Detectors](Detectors-Anomaly)

### ANO001 — Stuck values

**WARNING**, per column.

A value repeats unchanged for more than the stuck window: **5** under `"finance"` and `None`, **3** under `"sensor"`. The comparison is strictly greater than, so a window of 5 flags runs of 6+.

*Evidence:* `max_stuck_duration`

*What to do:* In sensor data this usually means a frozen instrument — hardware failed but is still reporting. In financial data it usually means a forward-fill happened upstream.

**This is the library's most common false positive.** Binary flags, categorical encodings, regime indicators, and binary targets all consist of long identical runs. If a column is *supposed* to be piecewise-constant, ignore ANO001 for it.

### ANO002 — Point anomalies

**WARNING**, per column.

A value exceeds the z-score threshold (**5.0** finance, **3.5** sensor, **4.0** default) **or** falls outside the 1.5×IQR fence. Either rule alone is enough.

*Evidence:* `zscore_outlier_count`, `iqr_outlier_count`, `agreement_count`, `esd_outlier_count`, `masking_suspected`, `max_zscore`, `worst_value`, `worst_timestamp`

*What to do:* A **non-zero** `agreement_count` — points flagged by both independent rules — is strong evidence of genuine errors.

A **zero** `agreement_count` is ambiguous and must not be read as reassurance. It has two opposite causes: a harmlessly skewed distribution, *or* contamination heavy enough to mask the z-score entirely. At 5% contamination the z-score half of the rule reports zero while the IQR half correctly finds every planted outlier, so both cases look identical in the counts.

**`esd_outlier_count` and `masking_suspected` resolve this.** Generalized ESD recomputes the scale after each removal so masking cannot occur: clean Gaussian data gives `esd=0`, 50 planted outliers give `esd=50`. Computed only for this ambiguous case (`None` otherwise) and it never changes what is flagged. See [Reading a zero agreement_count](Detectors-Anomaly#reading-a-zero-agreement_count).

Note also that beyond roughly 30% contamination **both** rules fail and no issue is raised at all — silence from ANO002 is not proof of clean data.

### ANO003 — Contextual spikes

**WARNING**, per column.

A value deviates from its **local** neighbourhood — a 21-point centered window with the point itself excluded — by more than the spike threshold (**4.0** finance, **3.0** sensor, **3.5** default).

*Evidence:* `n_spikes`, `max_spike_zscore`, `zero_variance_context`

*What to do:* These are values that are globally plausible but locally impossible. Excluding the point from its own window is what makes this work; leaving it in lets a large spike inflate the window's own standard deviation and mask itself.

**ANO002 and ANO003 are not redundant.** In a series ramping 0→240, a value of 200 sitting among neighbours near 58 is invisible to ANO002 and obvious to ANO003. There is a runnable demonstration on the [Anomaly Detectors](Detectors-Anomaly#the-distinction-that-matters-most) page.

---

## Leakage codes (LEK)

The core of the library. → [Leakage Detectors](Detectors-Leakage)

### LEK001 — Target equivalence

**CRITICAL**, per column. Requires `target=`.

A feature near-deterministically reproduces the target at lag 0: **AUC separation ≥ 0.95** for a binary target, or **|Spearman ρ| ≥ 0.95** for a continuous one.

*Evidence (binary):* `metric`, `auc`, `separation`, `threshold`, `target_type`, `n_obs`
*Evidence (continuous):* `metric`, `spearman_rho`, `threshold`, `target_type`, `n_obs`

*What to do:* Investigate immediately. Remove or reconstruct the feature. Keep it only if you can confirm it is genuinely available at prediction time.

**Why AUC and not Pearson.** Pearson correlation against a binary 0/1 target has a hard mathematical ceiling near **√(2/π) ≈ 0.798**, *even for a perfect relationship*. A feature whose sign defines the target scores only ~0.77 in Pearson and would slip under any sensible threshold. AUC scores that same relationship at exactly 1.0. This is the failure mode the library was built around — see [the demonstration](Detectors-Leakage#the-point-biserial-ceiling).

### LEK002 — Positive-lag correlation peak

**WARNING**, per column. Requires `target=`.

The feature's peak cross-correlation with the target falls at a **positive** lag, meaning it aligns better with the target's future than its present. Searches lags −10 to +10 by default; the peak must reach |r| ≥ 0.5.

*Evidence:* `peak_lag`, `peak_correlation`, `min_correlation`, `max_lag`, `metric`

*What to do:* Read `peak_correlation`. Above 0.7 is alarming; near the 0.5 gate is weak evidence. This is a **suspicion flag, not a proof** — a genuine strong predictor and a lookahead leak produce the same signature, and only magnitude separates them.

*Known false positives:* on near-random-walk columns (price levels especially) LEK002 flags **13%** of pairs that are statistically independent, and 8% of independent AR(0.98) pairs. The gate was raised from 0.1 to 0.5 in 0.3.1, which cut those from 37% and 51% at no cost to true detection. Where LEK002 and LEK003 disagree on such data, trust LEK003 — it false-positives at 3% and 15% on the same series. See [Detectors — Leakage](Detectors-Leakage#limitations-and-false-positives).

### LEK003 — Lookahead beyond persistence

**WARNING**, per column. Requires `target=`.

The feature correlates with the target's future by more than the target's own autocorrelation explains:

```
excess(k) = |corr(feature_t, target_{t+k})| − |corr(feature_t, target_t)| × |corr(target_t, target_{t+k})|
```

Flagged when `excess ≥ 0.1` at some lag k in 1..5, and the observed correlation itself reaches 0.1.

*Evidence:* `lag`, `observed_future_corr`, `excess_over_persistence`, `excess_threshold`, `metric`

*What to do:* This is the signature of a centered or forward-looking rolling window. Verify the feature uses only past data. Read `excess_over_persistence`: above 0.3 is strong evidence, near 0.1 may be estimation noise.

**This check exists because a naive one fails.** Autocorrelated targets make every honest feature correlate with the future transitively; subtracting the persistence-explained baseline is what separates real lookahead from that artefact.

### LEK004 — As-of availability leakage

**CRITICAL**, per column. Requires `available_at=`.

A value sits at a timestamp earlier than when it was actually published, so rows before the real release date consume information that did not exist yet.

*Evidence:* `n_violations`, `max_lookahead_days`, `first_violation`, `check`

*What to do:* **Shift the column to its release schedule — do not drop it.** The data is fine; the alignment is wrong.

**This cannot be inferred from values.** Whether a CPI figure was knowable on a given date is a fact about the world, not about your DataFrame. You declare availability, `tsauditor` verifies it, and the check is only as correct as your metadata.

### LEK005 — Combination leakage

**CRITICAL**, per column (the pair is in the evidence). Requires `target=`.

A **group of two or three** features together reconstructs the target — adjusted R² ≥ 0.95 — while **none does alone**.

*Evidence:* `metric`, `form`, `group`, `group_size`, `group_adjusted_r2`, `best_single_adjusted_r2`, `threshold`, `n_obs`

*What to do:* Check how the target was defined. This almost always means the target is an arithmetic function of those columns — a difference, mean, spread, product or ratio.

**Why the other checks miss it.** Every LEK001–LEK004 check is univariate. If `target = x1 - x2` with `x1` and `x2` independent, each correlates with the target at only ~0.7 — far under LEK001's 0.95 — while the pair explains it perfectly. Found in real data: in the OGDC file, `MACD_hist` is exactly `MACD - MACD_signal`, and LEK005 reports adjusted R² 1.0000 where the best single column reaches only 0.12.

**Two algebraic forms.** The `linear` form catches sums and differences; the `log` form catches products and ratios (`log(a*b) = log a + log b`). Which one fired is in `evidence["form"]`. The log form needs strictly positive values. Uses raw OLS rather than ranks — the leakage is arithmetic, and rank-transforming destroys it (the canonical case scores 0.941 on ranks, under the threshold).

**Larger groups without O(k^n).** A group is extended by another column only when it reaches `gate` (0.30) without reaching the flagging threshold. On random data nothing clears the gate, so deeper levels cost nothing and add no false positives. `max_group_size` defaults to 3.

**The single-feature guard.** A group is skipped when any column *alone* already reaches the threshold. Without it, one leaky column would produce a finding for every group it appears in, duplicating something LEK001 already reported once.

**False-positive profile.** On random targets with independent random features, the highest adjusted R² by chance was 0.075 (1,225 pairs, 100 rows), typically under 0.03; the log form is the same (0.028). Innocent but highly collinear pairs (r ≈ 0.96) score ~0.00 against an unrelated target.

**Limitations.** Groups larger than `max_group_size` (default 3) are not searched — raise it to find bigger identities. Non-monotonic constructions (`x1² + sin(x2)`) are not detected at all. Cost is O(k²) for the pair scan — about 0.27s for 100 features.

---

## Validity codes (VAL)

Values that are definitionally impossible. Requires `constraints=`. → [Validity Detectors](Detectors-Validity)

### VAL001 — Out-of-range value

**WARNING**, per column.

A value violates a declared bound: sentiment outside [−1, 1], a non-positive spread, a negative volume.

*Evidence:* `n_violations`, `min`, `max`, `min_exclusive`, `max_exclusive`, `observed_min`, `observed_max`, `check`

*What to do:* Read `observed_min` and `observed_max` — they describe the *violating* values only. 1.8 and −2.4 against a [−1, 1] scale suggests a few glitches; 180 and −240 suggests a missing division by 100.

### VAL002 — Relation violation

**CRITICAL**, per column (set to the *high* column).

A declared `(low, high)` ordering is broken on at least one row — for example `("bid", "ask")` where bid exceeds ask, a crossed book.

*Evidence:* `n_violations`, `low_col`, `high_col`, `first_violation`, `check`

*What to do:* Inspect the flagged timestamps for feed glitches. This is CRITICAL because a crossed book is an impossible market state, not merely an unusual one, and no per-column bound can detect it.

Validity issues are data errors, not leaks — they never appear in `leaky_columns()`.

---

## Panel codes (PNL)

Structural checks for multi-entity (long-format) data. Requires `group_col=`. → [Panel Data](Panel-Data)

### PNL001 — Ragged panel

**WARNING**, dataset-level.

Entities do not share a common time index — AAA has 200 days, BBB has 150.

*Evidence:* `n_groups`, `n_timestamps`, `min_coverage`, `max_coverage`, `n_complete_groups`, `worst_groups`, `group_col`

*What to do:* Invisible per entity, but it silently breaks every cross-sectional operation — a market-wide aggregate at time *t* covers a different set of entities than at *t+1*. Reindex onto the full timestamp set, or restrict to complete entities. Often legitimate (IPOs, delistings, sensor installs), hence WARNING rather than CRITICAL.

### PNL003 — Entity too short to audit

**INFO**, dataset-level.

Some entities have fewer than 30 rows — below the `min_obs` floor the leakage checks and ADF test require.

*Evidence:* `n_short_groups`, `n_groups`, `min_rows`, `shortest_groups`, `group_col`

*What to do:* This exists to stop you misreading silence as health. A 12-row entity produces no LEK001 finding because the check *declined to score it*, not because it is clean. Gather more history or exclude those entities.

### PNL004 — Rows with a null entity id

**WARNING**, dataset-level.

*Evidence:* `n_null_rows`, `n_total_rows`, `pct_null`, `group_col`

*What to do:* Rows where `group_col` is null cannot be grouped into any entity, so `groupby(group_col)`'s default `dropna=True` silently excludes them from every panel and per-entity check, and `apply_fixes()` leaves them unmodified. A null-id row produces zero findings from anything — not evidence of health. Assign the missing entity id or drop the rows explicitly.

### PNL002 — Cross-sectional lookahead

**WARNING**, dataset-level. Requires `group_col=` **and** `target=`.

A cross-sectional feature — a rank, z-score, decile or sector-neutralised value computed *across entities at one timestamp* — that ranks entities in the order their **future** target values will fall, by more than the target's own cross-sectional persistence explains.

*Evidence:* `metric`, `lag`, `observed_cs_corr`, `expected_from_cs_persistence`, `excess`, `excess_threshold`, `contemporaneous_cs_corr`, `n_entities`, `group_col`

*What to do:* Verify the feature is built from the cross-section at each row's own timestamp, not a later one. Read `excess` before acting — like LEK002/LEK003 this is a suspicion flag, and a genuinely predictive factor produces the same signature.

**Why a dedicated check.** LEK002/LEK003 *do* catch this — but only while idiosyncratic variation is large relative to any common factor. A cross-sectional rank measures *relative* position; once a market factor dominates absolute outcomes, relative position decouples from each entity's own result and the within-entity signal collapses. Measured across a common-factor sweep:

| common/idio ratio | LEK002 detection | LEK003 detection | PNL002 |
| ----------------- | ---------------- | ---------------- | ------ |
| 0 | 100% | 100% | detected |
| 5 | 95% | 95% | detected |
| 25 | 32.5% | 22.5% | detected |
| 100 | 22.5% | 12.5% | detected |

**The degradation is worse than a plain miss.** At a 25:1 ratio, LEK002 flagged the *legitimate* cross-sectional feature in 11 of 40 entities and the *leak* in 13 of 40 — essentially no discriminating power. And because `prevalence()` reports the leak as affecting 32% of entities, it reads as "isolated" when it is present in all of them.

**Calibration.** Realistic cross-sectional factor rank-ICs are 0.02–0.08. Tested factors up to IC 0.15 are not flagged; 0.30 and above are. Thresholds (`excess_threshold=0.15`, `min_correlation=0.15`) are deliberately stricter than LEK003's.

**Gates.** Requires at least 20 co-present entities per timestamp and 30 scored timestamps; below either, the check returns nothing rather than a noisy answer.

---

## Severity levels

| Severity | Meaning | Count |
| -------- | ------- | ----- |
| CRITICAL | Will directly corrupt training or evaluation. Resolve before modeling. | 5 codes |
| WARNING | Worth reviewing. May or may not need action depending on context. | 11 codes |
| INFO | Informational. No action required. | 2 codes |

Severity is fixed by the code and never changes with the domain.

---

## Adding a new code

If you are contributing a check:

1. Follow the prefix convention: `PRF*`, `ANO*`, `LEK*`, `VAL*`
2. Add a suggestion template to `tsauditor/report/remediation.py` — without one, your code falls back to a generic message
3. Populate `evidence` with the numbers behind the decision, not just a verdict
4. Add a row to this page's table and a section below
5. Add tests, including a case where it should **not** fire

See [Contributing](Contributing).
