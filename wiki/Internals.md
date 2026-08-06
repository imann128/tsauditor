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
5. **Sort** ascending by the index, `kind="mergesort"` specifically (see below).
6. **Target existence check.**

**The numeric-index refusal.** Worth reading the source comment on this one. `pd.to_datetime([0, 1, 2])` succeeds and returns three timestamps a few nanoseconds apart in 1970. Every frequency, gap, and clustering result computed on such an index would be wrong, and nothing would visibly fail. So a numeric index raises rather than coercing. String and object index labels *are* coerced, as a last resort, since those are usually genuine date strings.

**Why `kind="mergesort"`, not the default `quicksort`.** `quicksort` is not stable: two rows sharing an identical duplicate timestamp can come out reordered relative to the input, in a way that depends on the input's own row order rather than preserving it. A duplicate timestamp is already its own PRF004 CRITICAL finding, and downstream `keep="first"` dedup logic (`audit_frequency`'s, among others) depends on "first" meaning what the caller actually supplied. `mergesort` is the sort pandas/numpy document as stable.

### `ensure_sorted_datetime_index(df, context)`

Called at the entry point of every detector whose logic depends on row *position*, not just on the index being a `DatetimeIndex`: rolling windows, `.shift()`, consecutive-run detection, positional lag alignment. Validates the index is a `DatetimeIndex` (same error message every detector already used) and returns it sorted ascending, `kind="mergesort"` again for the same tie-order reason.

This exists because `validate_dataframe` sorting once inside `scan()` only covers the `scan()` path. Every `audit_*` function is also public API, callable (and called, in this codebase's own test suite) directly on a DataFrame that never went through `scan()`. A valid-but-out-of-order `DatetimeIndex` handed straight to a detector previously produced a wrong-but-silent result rather than an error: confirmed concretely, a shuffled version of the exact same rows made `audit_correlation_leakage` and `audit_temporal_leakage` miss a constructed lag+1 leak entirely (`[]`, no exception), and made `audit_contextual_anomalies` miss an 8-point stuck run.

Wired into: `audit_contextual_anomalies`, `audit_correlation_leakage`, `audit_temporal_leakage`, `audit_missing`, `audit_non_finite`, `audit_stationarity`.

`apply_fixes` in `remediate.py` has the same problem from the repair side and needs its own fix, since it receives `df` directly from the caller rather than through `scan()`: see "Detection reads from the pristine input" below.

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

The median is used rather than the mode so weekends and holidays do not shift a daily series into another bucket. The gaps between the bands (28–140 hours, for instance) all fall through to `"irregular"`, which is intentional: a series with a median gap of three days does not have a standard name.

This is metadata only. Precise gap analysis is `audit_frequency`'s job.

### `_is_polars(obj)`

```python
type(obj).__module__.split(".", 1)[0] == "polars"
```

A string check on the module path, so polars need not be installed to test for it.

---

## Leakage helpers

### `_auc(feature, y01)`: `leakage/equivalence.py`

AUC via the Mann-Whitney rank statistic:

```
AUC = (rank_sum_of_positives − n1(n1+1)/2) / (n1 × n0)
```

Uses `Series.rank()`, which assigns **average ranks** to ties, important, because a feature with many repeated values would otherwise produce a biased AUC.

Returns `None` when either class is absent, since AUC is undefined then.

Interpretation: the probability that a randomly chosen class-1 point ranks above a randomly chosen class-0 point.

### `encode_target(series, name)`: `leakage/_common.py`

Shared by `correlation.py` and `temporal.py` (imported as `_encode_target`). Returns a float Series. Numeric input is cast unchanged; a *non-numeric* binary categorical is mapped to 0.0/1.0 after sorting categories by their string form (so the mapping is deterministic across runs). Anything non-numeric with more than two categories raises.

`equivalence.py` does **not** use this helper, despite doing something that looks similar. Its binary path forces *any* two-valued target, numeric or categorical, to 0.0/1.0, because `_auc()` requires the positive class to be labeled exactly `1` (it does `y01.sum()` for the positive count and masks with `y01 == 1`). A numeric binary target like `{1, 2}` would silently break the AUC math if run through the shared helper's numeric-passthrough behavior. This is a real semantic difference, not incidental duplication — see the comment at the top of `audit_equivalence`'s binary branch.

`combination.py` uses this same "any two-valued target → 0.0/1.0" encoding too, inline, deliberately mirroring `equivalence.py`'s rule rather than reusing either it or the shared `encode_target` helper — its own comment says so explicitly: "matching equivalence.py's own rule exactly, so the guard below agrees with LEK001 about what 'explains the target alone' means." The single-feature guard in `combination.py` calls `equivalence._score_feature` directly to check a column against LEK001's own metric, and that guard only means what it claims to mean if both modules agree on what "binary" encodes to — so this one *is* deliberate consistency, not oversight, even though it means the same handful of lines exist in three places.

### `_align(a, b, tau)`: `leakage/correlation.py`

Slices two arrays so element *i* pairs `a_t` with `b_{t+tau}`:

```python
tau >= 0:  return a[:n-tau], b[tau:]
tau <  0:  return a[-tau:],  b[:n+tau]
```

Alignment by slicing rather than `shift()` avoids creating NaN padding that would then need masking.

### `_spearman(a, b, min_obs)`: `leakage/temporal.py`

Pairwise-complete Spearman with guards. Returns `None` (rather than NaN) when there are fewer than `min_obs` overlapping rows, when either side is constant, or when the correlation is NaN. Callers check for `None` explicitly.

### `_aligned_correlations(x, y, future_y, min_obs)`: `leakage/temporal.py`

Computes `r0` (feature vs target), `persistence` (target vs shifted target), and `observed` (feature vs shifted target) all on **one common mask**: rows where `x`, `y`, and `future_y` are simultaneously non-null. Used per feature and lag inside `audit_temporal_leakage`'s main loop.

This exists because computing the three on independent pairwise-complete samples let them silently describe different populations whenever a feature has its own missingness (e.g. a column only recorded starting partway through the series). Confirmed concretely: on a target with a genuine regime change (persistence 0.97 early, 0.0 late) and a feature recorded only in the low-persistence half, whole-series persistence came out ~0.75 against the feature's own-population persistence of ~0.22, a gap large enough to hide a real lag −1 leak (excess ~0.06, unflagged, versus ~0.14, correctly flagged, once aligned).

### `_score_feature(x, y, target_type, min_obs)`: `leakage/equivalence.py`

LEK001's own scoring function (AUC for a binary target, absolute Spearman for continuous), extracted so `audit_combination_leakage`'s single-feature guard can call it directly. Before this existed as a shared function, `combination.py`'s guard only checked its own adjusted-R² metric, which could miss a column LEK001 already flagged via AUC/Spearman on a strong monotonic-but-nonlinear relationship (R² well below LEK005's threshold, AUC/Spearman near 1.0). The guard now takes `max(adjusted_r2, equivalence_score)` per column.

### `_availability(spec, index, col)`: `leakage/asof.py`

Resolves an availability spec into a per-row timestamp Series.

- `pd.Timedelta` → `index + spec`
- `pd.Series` → reindexed onto `df.index` and coerced with `errors="coerce"`

If reindexing produces all-NaT, it raises rather than silently returning zero violations. A misaligned Series would otherwise look exactly like clean data, which is the worst possible failure mode for a leakage check.

Also checks `index.tz` against the availability Series' `.dt.tz` and raises a `ValueError` naming the mismatch if one is tz-aware and the other tz-naive. Before this check existed, that mismatch failed as a raw `TypeError` from deep inside the `avail > idx` comparison in `audit_asof_leakage`, a confusing failure for a mundane mistake (the index was `tz_localize`'d, the release-date metadata came from somewhere that wasn't).

### `_adjusted_r2`, `_score_arrays`, `_Matrix`: `leakage/combination.py`

`_adjusted_r2` fits `y ~ 1 + X` with `lstsq` rather than a normal-equation solve, because candidate groups are frequently collinear (`high`/`low`, a level and its lag) and `X'X` would be singular.

`_score_arrays` returns the better of the **linear** and **log** forms, where the log form fits `log|y| ~ log|X|`. Absolute values are deliberate: `|a*b| = |a|*|b|` holds regardless of sign, so signed products and ratios are recovered. It is skipped when any magnitude falls below `_POSITIVE_FLOOR` (1e-12), since `log` of a near-zero would dominate the fit.

`_Matrix` is a column-major view with a precomputed NaN mask. Building a `pd.concat` per candidate group was the dominant cost, over a second for 50 features; slicing preextracted arrays with a boolean mask brought the same scan to 0.07s.

### `_generalized_esd(values, alpha)`: `anomaly/point.py`

Rosner (1983). Removes the most extreme point and **recomputes** the mean and standard deviation before testing the next, so masking cannot occur by construction. Reported as evidence only (it never changes what ANO002 flags), and computed solely for the ambiguous case (z-score count 0, IQR count above 0), since it is O(k·n).

Capped at `_ESD_MAX_FRACTION` (40%) of the column length: beyond that the "outliers" are a second population rather than anomalies.

### Cross-sectional helpers: `panel.py`

`_cross_sectional_corr` averages, over timestamps, the Spearman correlation *across entities*. It is vectorised: both frames are masked to their co-present entries, ranked row-wise once, then correlated with array arithmetic. A per-timestamp `Series.corr` loop was ~80x slower and made PNL002 unusable on a realistic panel (4.0s vs 0.05s for 40 entities).

### Performance notes

Two loops were deliberately hoisted:

**`audit_correlation_leakage`** rank-transforms the target **once** before the feature loop, then correlates shifted ranks. Since Spearman is Pearson-on-ranks, this is equivalent to re-ranking at each lag but far cheaper. Re-ranking inside the loop was the previous hot path.

**`audit_temporal_leakage`** builds the target's shifted series once before the feature loop, since they do not depend on any feature. A whole-series persistence correlation is also computed once, but only as a cheap pre-filter to skip a lag outright when even that loosest possible sample is too small; the persistence value actually used in the `expected(k)` math is now computed per feature and lag by `_aligned_correlations` (see Leakage helpers above), since using the whole-series value there was the mismatched-sample bug that check exists to fix.

---

## Anomaly and remediation helpers

`tsauditor/remediate.py` needs to recompute the detector masks in order to repair them. **These used to be a second, hand-maintained copy of every detector formula in `anomaly/point.py` and `anomaly/contextual.py`, connected only by a comment** ("Match anomaly/point.py ANO002"). That drifted out of sync in practice once (a single-row-gap fix landed in the ANO001 detector without a matching update to `remediate.py`'s own copy, so `scan()` flagged a run that `apply_fixes` then silently failed to repair). Rather than add more tests around the duplication, it was removed: every domain preset and masking formula now lives in one place, `tsauditor/anomaly/_common.py`, imported by both the detectors and `remediate.py`. There is no second copy left to drift.

### `tsauditor/anomaly/_common.py`

| Function | What it does |
| -------- | ------------ |
| `zscore_preset(domain)` | ANO002's z-score threshold (finance 5.0, sensor 3.5, otherwise 4.0) |
| `stuck_window_preset(domain)` | ANO001's stuck-run window (sensor 3, otherwise 5) |
| `spike_threshold_preset(domain)` | ANO003's spike threshold (finance 4.0, sensor 3.0, otherwise 3.5) |
| `SPIKE_WINDOW` | Module constant, 21: ANO003's local-context window width |
| `zscore_iqr_masks(series, z_thresh)` | `(z_mask, iqr_mask, z_scores, degenerate)`; `degenerate` is `True` when std is 0 or NaN |
| `clip_bounds(series, z_thresh)` | Winsorization bounds, the intersection of the z-band and the IQR fence |
| `stuck_run_mask(series, window)` | `(mask, counts)`, bridging a single-row gap (see below) |
| `spike_stats(values, window, threshold)` | `(mask, z_scores, flat_context_spike, local_mean, local_std)`, the excluding-self rolling z-score |
| `spike_bounds(values, window, threshold)` | `(mask, lower, upper)`, wraps `spike_stats` for repair |

`anomaly/point.py` and `anomaly/contextual.py` call these directly for detection; `remediate.py` calls the same functions for repair. `tests/test_fix.py::test_detector_and_repair_share_the_same_threshold_and_mask_functions` asserts this by *identity* (the detector and repair modules resolve to the same function object, not just matching output), which is what makes the old drift structurally impossible to reintroduce by accident rather than merely tested against.

### `zscore_iqr_masks` / `clip_bounds`

```python
z_mask | iqr_mask   # combined outlier mask
```

`clip_bounds` returns the **intersection** of the two rules' acceptance regions, not the union:

```python
lower = max(mean - z_thresh * std,  q25 - 1.5 * iqr)
upper = min(mean + z_thresh * std,  q75 + 1.5 * iqr)
```

This is the region a point must be in to be flagged by *neither* rule, so clipping to it pulls in exactly the flagged outliers and leaves every inlier untouched.

### `stuck_run_mask(series, window)`

```python
bridge_series = series.interpolate(method="linear", limit=1)
diffs  = bridge_series.diff().ne(0).cumsum()   # new group ID on every change
counts = bridge_series.groupby(diffs).transform("count")
mask   = (counts > window) & series.notna()
```

Strictly greater than, so `window=5` flags runs of 6+. Groups on a **bridged** view rather than the raw series: a lone missing reading inside an otherwise-flat run is still a stuck run, but grouping on the raw series would split it into two, because `diff()` against a `NaN` reads as "changed" both at the gap and at the row right after it, and neither half might cross `window` even though the true, uninterrupted run does. Interpolating a single NaN only produces a zero diff when both neighbours already agree, so a genuine transition (a gap between two *different* values) still breaks the group correctly; this never masks a real change, only bridges a real stuck run.

### `spike_stats(values, window, threshold)` / `spike_bounds`

The excluding-self rolling z-score, plus (via `spike_bounds`) the local clip band.

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

**Why excluding-self is essential** is documented at length on the [Anomaly Detectors](Detectors-Anomaly#how-ano003-works-contextual-spikes) page. In short: a point inside its own window inflates that window's mean and standard deviation, and the two effects cancel. A 50× spike scored z ≈ 1.8 in a 5-point centered window. It masked itself. That was a real bug.

**`SPIKE_WINDOW = 21`** is a module constant in `_common.py`, not a parameter. The window must be wide enough for the neighbours' standard deviation to be stable; with 3–4 neighbours it collapses toward zero by chance and floods the output with false positives.

### Detection reads from the pristine input, not the in-progress repair

`apply_fixes` runs its repair steps in a fixed order (outliers/spikes, then stuck values, then imputation). Every step's mask/bounds computation reads from the *original* `df` passed in, never from `out` (the copy earlier steps have already modified). A column can carry more than one finding at once (an ANO002 outlier and an ANO003 spike, or a value stuck at an extreme constant that is both an ANO002 outlier and an ANO001 stuck run); detecting against an already-repaired value would let an earlier step change what a later step finds. `cells_changed` in the change log likewise counts only the cells *that step* newly changes, not cells another step already touched.

### `apply_fixes` sorts internally, then restores the caller's row order

`apply_fixes(report, df)` receives `df` directly from the caller, independent of whatever sorted copy `scan()` validated internally (same underlying gap as `ensure_sorted_datetime_index` above, on the repair side). If that `df` has a valid but chronologically out-of-order `DatetimeIndex`, every repair mask (`stuck_run_mask`'s consecutive-run walk, `spike_bounds`' rolling window) computed on the unsorted order could find nothing at all, even for a column `report` says was flagged, silently returning a "repaired" frame that still contains the original anomaly.

`apply_fixes` now sorts its own working copy by position first:

```python
sort_positions = np.argsort(df.index.values, kind="mergesort")
if not np.array_equal(sort_positions, np.arange(len(df))):
    restore_positions = np.empty_like(sort_positions)
    restore_positions[sort_positions] = np.arange(len(sort_positions))
    df = df.iloc[sort_positions]
```

Position-based (`np.argsort` + `.iloc`), not `.sort_index()` + relabel: a duplicate timestamp is already its own CRITICAL PRF004 finding, and a label-based reindex would make its tie order ambiguous rather than matching `mergesort`'s stable order. The original row order is restored (`out.iloc[restore_positions]`) before returning, both so the "byte-for-byte unchanged for untouched columns" guarantee holds in the caller's own row order, and because `_apply_fixes_by_group` (the panel path) writes each entity's result back by raw position and requires the row order it receives to match what it passed in.

**`time_col` callers need a separate fix.** The sort-safety above only engages when `df.index` is already a `DatetimeIndex`. A caller who scanned via `scan(df, time_col="date")` and then calls `report.apply_fixes(df)` (or `fix(df, time_col="date")`) hands `apply_fixes` the *original* `df`, still with `time_col` as a plain column and a meaningless `RangeIndex`, not the `DatetimeIndex` `scan()` resolved internally, so the position-sort above never triggers at all. `scanner.py` now records `metadata["time_col"]`; `apply_fixes` resolves it into a working `DatetimeIndex` the same way `validate_dataframe` does (`pd.to_datetime` + `set_index`) before repairing, then restores the caller's original column layout (`time_col` back as a plain column) and row order before returning. This composes with `group_col`: the per-entity recursive call in `_apply_fixes_by_group` inherits the already-resolved index rather than re-resolving it.

### `_impute(series, method, datetime_index)`

`"interpolate"` uses `method="time"` on a `DatetimeIndex` and `method="linear"` otherwise, with `limit_direction="both"` so leading and trailing NaNs are also filled. `"ffill"` and `"bfill"` are straightforward.

---

## Health score internals

### `affected_cells(report, df)`

Counts distinct cells implicated by a **quality** issue. Only these codes count:

```python
_QUALITY_CODES = ("PRF002", "PRF006", "PRF007", "ANO001", "ANO002", "ANO003")
```

Per column, it builds a boolean mask by OR-ing together the relevant per-code masks, then counts. Because it is a union, a cell flagged by both ANO002 and ANO003 counts **once**.

The masks are recomputed here, not read from the issues: the `Issue` objects carry counts, not positions. The actual per-column mask logic is factored into `_affected_cells_single(issues, df, z_thresh, window, spike_thresh)`, given only one series' worth of data and Issues.

**Panel-aware.** When `report.metadata["group_col"]` is set, `affected_cells` calls `_affected_cells_single` once per entity, on that entity's own slice of `df` and only its own Issues, then sums. Every quality detector actually ran per-entity during the original scan (`scanner.py`'s per-partition loop), so recomputing one mask across the whole interleaved panel instead would mix every entity's values into a single mean/std/rolling-window: a real outlier in a small-scale entity can be diluted below a large-scale entity's ordinary range and vanish from the count entirely, or the reverse. Rows with a null entity id (PNL004) are skipped, since they were never scanned per-entity to begin with.

### `health_score(report, df)`

```python
100 * (1 - affected_cells / (len(df) * n_numeric_columns))
```

Rounded to 1dp. Returns `100.0` when there are no numeric columns.

`GuardReport.health_score` wraps this and **re-scans first**, with `run_leakage=False` and `run_stationarity=False` (neither affects the score). Reusing the report's own issues would give a stale answer when scoring a repaired frame, defeating the whole point of a before/after comparison.

The cost is a full scan per call. Do not put it in a loop.

---

## Report internals

### `_json_default(obj)`: `report/summary.py`

Serialization fallback for `to_json`. Converts `np.integer` → `int`, `np.floating` → `float`, `np.ndarray` → `list`, and falls back to `str()`.

Without this, NumPy scalars in `evidence` would serialize as quoted strings and every consumer of the JSON would have to parse numbers back out of text.

### `_append_issue(report, issue)`: `scanner.py`

Routes an `Issue` into `critical`, `warnings`, or `info` by its `severity`. Detectors do not know which bucket they are writing to; they set a severity and this function does the routing.

### `_SafeDict`: `report/remediation.py`

A `dict` subclass whose `__missing__` returns `"{key}"`: the placeholder, unchanged.

This lets `str.format_map` render a suggestion template even when the evidence dict lacks a referenced key. Without it, a detector that omitted an evidence key would raise `KeyError` from inside `Issue.suggestion`, turning a missing-documentation problem into a crash.

### `suggest(code, column, evidence)`

Looks up `_REMEDIATIONS[code]`, falls back to a generic message for unknown codes, then fills the template from `evidence` plus two synthetic fields:

- `{target}` → `"column 'X'"`, or `"the dataset"` when `column` is `None`
- `{column}` → the column name, or `"this column"`

Suggestions live here rather than in the detectors so each detector does not carry remediation prose. **Add an entry here when you add a check**; without one, your code silently falls back to `"Review this issue before using the data for modeling."`

---

## Known rough edges

Honest notes for anyone working on the codebase.

**`domain` is accepted and ignored** by all four leakage detectors and by `audit_stationarity`. Signature consistency at the cost of a slightly misleading API.

**Issues are per-column, not per-cell.** Recovering *which* rows were flagged requires rebuilding the mask yourself. This is the most frequently felt limitation of the report format.

**LEK001-LEK004 are univariate.** Leakage emerging from a *combination* of features is handled separately by `leakage/combination.py` (LEK005), and cross-entity leakage by `panel.py` (PNL002).

---

## Adding a new detector

1. Create the function in the appropriate module. Signature: `(df, ..., domain=None) -> List[Issue]`.
2. If your check depends on row order (rolling windows, `.shift()`, consecutive-run detection, positional lag alignment), call `ensure_sorted_datetime_index(df, "your_function_name")` at the top rather than just checking `isinstance(df.index, pd.DatetimeIndex)` yourself, so a valid-but-unsorted index is handled instead of silently mis-scored (see Validation above). Return `[]` for an empty frame.
3. Skip degenerate inputs (constant columns, too few observations) rather than raising; one odd column must not abort the whole scan.
4. Populate `evidence` with the numbers behind the decision, including the threshold applied. Users need to see *why*, not just *what*.
5. Add a suggestion template to `report/remediation.py`.
6. Wire it into `scanner.py` behind the appropriate `run_*` flag.
7. Add tests, **including a case where it should not fire**.
8. Document it: a section on the relevant detector page and a row in [Issue Code Reference](Issue-code-reference).

Step 4 is what makes this library useful rather than merely opinionated. A detector that reports a verdict without its reasoning cannot be trusted or debugged.

→ [Contributing](Contributing)
