# Internals

Private helpers and implementation details, for contributors and anyone reading the source.

**Nothing on this page is public API.** These functions have no stability guarantee and may change in any release. If you find yourself needing one, that is a signal the public API has a gap worth reporting as an issue.

---

## Validation

`tsauditor/utils/validation.py`

### `validate_dataframe(df, target, time_col)`

Called by `scan()` before any check runs. Normalizes input into a known-good shape.

Steps, in order:

1. **Polars detection and conversion.** `_is_polars(obj)` checks `type(obj).__module__` rather than importing polars, so polars stays optional. Conversion requires `time_col` because polars has no index.
2. **Type and emptiness checks.** `TypeError` for non-DataFrames, `ValueError` for empty.
3. **Copy.** Everything after this point works on a copy.
4. **Index resolution.**
5. **Sort** ascending by the index.
6. **Target existence check.**

**The numeric-index refusal.** Worth reading the source comment on this one. `pd.to_datetime([0, 1, 2])` succeeds and returns three timestamps a few nanoseconds apart in 1970. Every frequency, gap, and clustering result computed on such an index would be wrong, and nothing would visibly fail. So a numeric index raises rather than coercing. String and object index labels *are* coerced, as a last resort, since those are usually genuine date strings.

### `infer_frequency(index)`

Returns a coarse label from the **median** gap between consecutive timestamps:

| Median gap | Label |
| ---------- | ----- |
| < 20 hours | `"sub-daily"` |
| 20–28 hours | `"daily"` |
| 140–196 hours | `"weekly"` |
| 600–960 hours | `"monthly"` |
| anything else | `"irregular"` |
| fewer than 2 rows | `"unknown"` |

The median is used rather than the mode so weekends and holidays do not shift a daily series into another bucket. The gaps between the bands (28–140 hours, for instance) all fall through to `"irregular"`, which is intentional — a series with a median gap of three days does not have a standard name.

This is metadata only. Precise gap analysis is `audit_frequency`'s job.

### `_is_polars(obj)`

```python
type(obj).__module__.split(".", 1)[0] == "polars"
```

A string check on the module path, so polars need not be installed to test for it.

---

## Leakage helpers

### `_auc(feature, y01)` — `leakage/equivalence.py`

AUC via the Mann-Whitney rank statistic:

```
AUC = (rank_sum_of_positives − n1(n1+1)/2) / (n1 × n0)
```

Uses `Series.rank()`, which assigns **average ranks** to ties — important, because a feature with many repeated values would otherwise produce a biased AUC.

Returns `None` when either class is absent, since AUC is undefined then.

Interpretation: the probability that a randomly chosen class-1 point ranks above a randomly chosen class-0 point.

### `_encode_target(series, name)` — `leakage/correlation.py`, `leakage/temporal.py`

Returns a float Series. Numeric input is cast; a binary categorical is mapped to 0.0/1.0 after sorting categories by their string form (so the mapping is deterministic across runs). Anything non-numeric with more than two categories raises.

**This helper is duplicated verbatim in both modules.** `equivalence.py` has its own inline version of the same logic. Three copies of one function is a consolidation opportunity for a future refactor.

### `_align(a, b, tau)` — `leakage/correlation.py`

Slices two arrays so element *i* pairs `a_t` with `b_{t+tau}`:

```python
tau >= 0:  return a[:n-tau], b[tau:]
tau <  0:  return a[-tau:],  b[:n+tau]
```

Alignment by slicing rather than `shift()` avoids creating NaN padding that would then need masking.

### `_spearman(a, b, min_obs)` — `leakage/temporal.py`

Pairwise-complete Spearman with guards. Returns `None` — rather than NaN — when there are fewer than `min_obs` overlapping rows, when either side is constant, or when the correlation is NaN. Callers check for `None` explicitly.

### `_availability(spec, index, col)` — `leakage/asof.py`

Resolves an availability spec into a per-row timestamp Series.

- `pd.Timedelta` → `index + spec`
- `pd.Series` → reindexed onto `df.index` and coerced with `errors="coerce"`

If reindexing produces all-NaT, it raises rather than silently returning zero violations. A misaligned Series would otherwise look exactly like clean data, which is the worst possible failure mode for a leakage check.

### `_adjusted_r2`, `_score_arrays`, `_Matrix` — `leakage/combination.py`

`_adjusted_r2` fits `y ~ 1 + X` with `lstsq` rather than a normal-equation solve, because candidate groups are frequently collinear (`high`/`low`, a level and its lag) and `X'X` would be singular.

`_score_arrays` returns the better of the **linear** and **log** forms, where the log form fits `log|y| ~ log|X|`. Absolute values are deliberate: `|a*b| = |a|*|b|` holds regardless of sign, so signed products and ratios are recovered. It is skipped when any magnitude falls below `_POSITIVE_FLOOR` (1e-12), since `log` of a near-zero would dominate the fit.

`_Matrix` is a column-major view with a precomputed NaN mask. Building a `pd.concat` per candidate group was the dominant cost — over a second for 50 features; slicing preextracted arrays with a boolean mask brought the same scan to 0.07s.

### `_generalized_esd(values, alpha)` — `anomaly/point.py`

Rosner (1983). Removes the most extreme point and **recomputes** the mean and standard deviation before testing the next, so masking cannot occur by construction. Reported as evidence only — it never changes what ANO002 flags — and computed solely for the ambiguous case (z-score count 0, IQR count above 0), since it is O(k·n).

Capped at `_ESD_MAX_FRACTION` (40%) of the column length: beyond that the "outliers" are a second population rather than anomalies.

### Cross-sectional helpers — `panel.py`

`_cross_sectional_corr` averages, over timestamps, the Spearman correlation *across entities*. It is vectorised: both frames are masked to their co-present entries, ranked row-wise once, then correlated with array arithmetic. A per-timestamp `Series.corr` loop was ~80x slower and made PNL002 unusable on a realistic panel (4.0s vs 0.05s for 40 entities).

### Performance notes

Two loops were deliberately hoisted:

**`audit_correlation_leakage`** rank-transforms the target **once** before the feature loop, then correlates shifted ranks. Since Spearman is Pearson-on-ranks, this is equivalent to re-ranking at each lag but far cheaper. Re-ranking inside the loop was the previous hot path.

**`audit_temporal_leakage`** computes the target's shifted series and persistence correlations **once** before the feature loop. They do not depend on any feature, so computing them per-feature was pure waste.

---

## Anomaly and remediation helpers

`tsauditor/remediate.py` recomputes the detector masks in order to repair them. **These formulas are duplicated from the detector modules.**

### The duplication problem

| Helper in `remediate.py` | Mirrors |
| ------------------------ | ------- |
| `_outlier_mask` | `anomaly/point.py` ANO002 |
| `_stuck_mask` | `anomaly/contextual.py` ANO001 |
| `_spike_info` | `anomaly/contextual.py` ANO003 |
| `_zscore_threshold` | ANO002 domain defaults |
| `_stuck_window` | ANO001 domain defaults |
| `_spike_threshold` | ANO003 domain defaults |

If a detector's formula changed and its mirror did not, `apply_fixes` would repair cells the report never flagged — silently corrupting data the user believes was audited.

**`tests/test_fix.py` pins them together** by asserting the repaired cell count matches the detector's own evidence. If you touch either side, keep that test passing. It is the only thing preventing drift.

### `_outlier_mask(values, z_thresh)`

`z_mask | iqr_mask`. Returns all-`False` when the standard deviation is 0 or NaN.

### `_clip_bounds(values, z_thresh)`

```python
lower = max(mean - z_thresh * std,  q25 - 1.5 * iqr)
upper = min(mean + z_thresh * std,  q75 + 1.5 * iqr)
```

The **intersection** of the two rules' acceptance regions, not the union. This is the region a point must be in to be flagged by *neither* rule — so clipping to it pulls in exactly the flagged outliers and leaves every inlier untouched.

### `_stuck_mask(series, window)`

```python
diffs  = series.diff().ne(0).cumsum()          # new group ID on every change
counts = series.groupby(diffs).transform("count")
return (counts > window) & series.notna()
```

Strictly greater than, so `window=5` flags runs of 6+. `diff()` against NaN yields NaN (which is `!= 0`), so a NaN correctly splits a run rather than joining two.

### `_spike_info(values, window, threshold)`

The excluding-self rolling z-score, plus the local clip band. Returns `(mask, lower, upper)`.

```python
n_excl     = roll.count() - 1
sum_excl   = roll.sum() - values
sumsq_excl = roll_sq.sum() - sq

local_mean = sum_excl / n_excl
local_var  = sumsq_excl / n_excl - local_mean**2
local_std  = sqrt(local_var.clip(lower=0))
```

Two details matter:

**`clip(lower=0)` on the variance.** The identity `E[x²] − E[x]²` is exact in real arithmetic but can produce values like `−1e-17` in floating point when the true variance is near zero. Taking the square root of that yields NaN and silently drops the point. Clipping first prevents it.

**`min_periods = max(3, window // 2)`.** Points near the series edges are still evaluated, with a partial window. Below 3 neighbours the local standard deviation is too unstable to be meaningful.

**Why excluding-self is essential** is documented at length on the [Anomaly Detectors](Detectors-Anomaly#how-ano003-works--contextual-spikes) page. In short: a point inside its own window inflates that window's mean and standard deviation, and the two effects cancel. A 50× spike scored z ≈ 1.8 in a 5-point centered window. It masked itself. That was a real bug.

**`_SPIKE_WINDOW = 21`** is a module constant, not a parameter. The window must be wide enough for the neighbours' standard deviation to be stable; with 3–4 neighbours it collapses toward zero by chance and floods the output with false positives.

### `_impute(series, method, datetime_index)`

`"interpolate"` uses `method="time"` on a `DatetimeIndex` and `method="linear"` otherwise, with `limit_direction="both"` so leading and trailing NaNs are also filled. `"ffill"` and `"bfill"` are straightforward.

---

## Health score internals

### `affected_cells(report, df)`

Counts distinct cells implicated by a **quality** issue. Only these codes count:

```python
_QUALITY_CODES = ("PRF002", "PRF006", "ANO001", "ANO002", "ANO003")
```

Per column, it builds a boolean mask by OR-ing together the relevant per-code masks, then counts. Because it is a union, a cell flagged by both ANO002 and ANO003 counts **once**.

The masks are recomputed here, not read from the issues — the `Issue` objects carry counts, not positions.

### `health_score(report, df)`

```python
100 * (1 - affected_cells / (len(df) * n_numeric_columns))
```

Rounded to 1dp. Returns `100.0` when there are no numeric columns.

`GuardReport.health_score` wraps this and **re-scans first**, with `run_leakage=False` and `run_stationarity=False` (neither affects the score). Reusing the report's own issues would give a stale answer when scoring a repaired frame, defeating the whole point of a before/after comparison.

The cost is a full scan per call. Do not put it in a loop.

---

## Report internals

### `_json_default(obj)` — `report/summary.py`

Serialization fallback for `to_json`. Converts `np.integer` → `int`, `np.floating` → `float`, `np.ndarray` → `list`, and falls back to `str()`.

Without this, NumPy scalars in `evidence` would serialize as quoted strings and every consumer of the JSON would have to parse numbers back out of text.

### `_append_issue(report, issue)` — `scanner.py`

Routes an `Issue` into `critical`, `warnings`, or `info` by its `severity`. Detectors do not know which bucket they are writing to; they set a severity and this function does the routing.

### `_SafeDict` — `report/remediation.py`

A `dict` subclass whose `__missing__` returns `"{key}"` — the placeholder, unchanged.

This lets `str.format_map` render a suggestion template even when the evidence dict lacks a referenced key. Without it, a detector that omitted an evidence key would raise `KeyError` from inside `Issue.suggestion` — turning a missing-documentation problem into a crash.

### `suggest(code, column, evidence)`

Looks up `_REMEDIATIONS[code]`, falls back to a generic message for unknown codes, then fills the template from `evidence` plus two synthetic fields:

- `{target}` → `"column 'X'"`, or `"the dataset"` when `column` is `None`
- `{column}` → the column name, or `"this column"`

Suggestions live here rather than in the detectors so each detector does not carry remediation prose. **Add an entry here when you add a check** — without one, your code silently falls back to `"Review this issue before using the data for modeling."`

---

## Known rough edges

Honest notes for anyone working on the codebase.

**`_encode_target` exists three times** — in `correlation.py`, `temporal.py`, and inline in `equivalence.py`.

**Detector formulas are duplicated in `remediate.py`**, held together only by `tests/test_fix.py`.

**`domain` is accepted and ignored** by all four leakage detectors and by `audit_stationarity`. Signature consistency at the cost of a slightly misleading API.

**Issues are per-column, not per-cell.** Recovering *which* rows were flagged requires rebuilding the mask yourself. This is the most frequently felt limitation of the report format.

**`to_dict()` omits the info count**, while `to_json()` includes it.

**LEK001-LEK004 are univariate.** Leakage emerging from a *combination* of features is handled separately by `leakage/combination.py` (LEK005), and cross-entity leakage by `panel.py` (PNL002).

---

## Adding a new detector

1. Create the function in the appropriate module. Signature: `(df, ..., domain=None) -> List[Issue]`.
2. Validate the index is a `DatetimeIndex` if your check is temporal; return `[]` for an empty frame.
3. Skip degenerate inputs (constant columns, too few observations) rather than raising — one odd column must not abort the whole scan.
4. Populate `evidence` with the numbers behind the decision, including the threshold applied. Users need to see *why*, not just *what*.
5. Add a suggestion template to `report/remediation.py`.
6. Wire it into `scanner.py` behind the appropriate `run_*` flag.
7. Add tests, **including a case where it should not fire**.
8. Document it: a section on the relevant detector page and a row in [Issue Code Reference](Issue-code-reference).

Step 4 is what makes this library useful rather than merely opinionated. A detector that reports a verdict without its reasoning cannot be trusted or debugged.

→ [Contributing](Contributing)
