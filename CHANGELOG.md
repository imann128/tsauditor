# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/), and the project adheres to
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added: `n_jobs` and `chunk_size` parallelize `group_col` scans

- **`scan(df, group_col=..., n_jobs=-1)` now audits entities in parallel via
  `joblib`, instead of a plain sequential loop.** Previously the only way to
  get real parallel speedup on panel data was to bypass `group_col` entirely:
  split the frame into separate per-entity DataFrames and drive
  `Parallel(delayed(scan))` externally, exactly as the README's own scaling
  section documents. That workaround got the speedup; the more ergonomic
  single-call `group_col=` path did not, silently, since nothing about
  `scan()`'s panel-mode branch used `joblib`, `concurrent.futures`, or any
  other parallelism.

  `n_jobs` defaults to `1` (sequential, matches prior behavior exactly, no
  change for existing callers). Set `n_jobs=-1` to use all available cores.

  **Dispatched in chunks, not one task per group.** Real panel data often
  has many small entities (tens of rows each). Dispatching one `joblib` task
  per group makes per-task overhead, pickling, worker dispatch, dominate for
  datasets shaped like that, which can make naive one-task-per-group
  parallelism net *slower* than sequential. `chunk_size` controls how many
  groups are bundled per dispatched task; left at its default (`None`), it's
  auto-sized from `len(groups)` and the actual worker count
  (`joblib.cpu_count()` when `n_jobs=-1`) so each worker gets a handful of
  tasks rather than exactly one.

  **Correctness, not just speed, is the property being guarded.**
  Parallelizing must never change *what* gets reported, only how long it
  takes: `Parallel` preserves submission order (not completion order), and
  the report is only mutated in the main process after every worker's
  results come back, so issue order and content are identical to the
  sequential path regardless of `n_jobs` or `chunk_size`. Verified directly:
  the same panel scanned at `n_jobs=1, 2, -1` and a custom `chunk_size`
  produces byte-identical issue lists (code, group, column, severity, and
  order) in every case, and this is now a regression test, not just a
  one-time check.

### Internal

- Panel-mode helper functions (`_audit_one_group`, `_audit_group_chunk`,
  `_auto_chunk_size`) are module-level, not closures, so they pickle
  correctly for `joblib`'s default `loky` backend, a nested function here
  would fail to parallelize at all, silently, only surfacing as a runtime
  error the first time someone actually passed `n_jobs != 1`.

## [0.4.0] - 2026-07-30

### Fixed: `audit_combination_leakage` was unreachable from `tsauditor.leakage`

- **LEK005 was importable only via `tsauditor.leakage.combination`, not
  `tsauditor.leakage`**: every other detector in the package (equivalence,
  correlation, temporal, asof) is reachable both ways. A user following the
  pattern documented on the wiki's API Reference page,
  `from tsauditor.leakage import audit_...`, would hit an `ImportError`
  specifically for combination leakage. `tsauditor/leakage/__init__.py` now
  re-exports it and lists LEK005 alongside the other codes in its module
  docstring. Caught while cross-checking the wiki against the actual public
  surface.

### Fixed: pre-publish sweep

A final correctness/docs pass before release, run specifically because the
project's original motivation was a statistical bug that looked clean on the
surface (the OGDC `ChangeP` leak). Nothing below changes detector behavior;
it's the surrounding infrastructure and docs that were wrong.

- **`.gitattributes` was UTF-16 encoded.** Git cannot parse a `text=auto`
  rule from a UTF-16 file, so line-ending normalization silently never
  worked: the root cause of "every file shows as modified" recurring
  throughout this project's history. Rewritten as plain UTF-8. Re-running
  `git add --renormalize .` afterward took the working-tree diff from 81
  files / ~27,600 changed lines down to the ~40 files actually touched this
  release.
- **README told users to write `domain="None"`** (the string) for
  domain-agnostic scans. That actually raises `ValueError`: only Python's
  `None`, or omitting `domain=` entirely, is accepted. Text corrected to
  say so explicitly.
- **A wiki anchor link was broken.** `Detectors-Anomaly.md` linked to
  `Internals#_outlier_mask`, but the actual heading is
  `` `_outlier_mask(values, z_thresh)` ``, which slugifies differently.
  Found via a full link-integrity check across all 17 wiki pages (now
  zero broken internal links); fixed.
- **`tsauditor/anomaly/__init__.py`'s docstring described a `classifier`
  module** that was never implemented. Removed the phantom reference.
- **Two tests asserted less than their own names promised.**
  `test_drop_is_an_alias_for_nan_and_never_deletes_rows` only checked row
  count, not that the cell actually became `NaN`; `test_cleans_the_target_column`
  (TimesFM adapter) only checked the imputed value was finite, not that it
  was a sane interpolation. Both strengthened.
- **README's test count and a wiki version string were stale** (430 vs. the
  actual 448; two `0.3.0` references left over from the last release).

### Fixed: `to_dict()` undercounted issues

- **`GuardReport.to_dict()` omitted the `info` count**, so its `counts` total
  didn't match `to_json()`'s for the same report: a caller reading counts
  from `to_dict()` alone would see fewer issues than actually exist.
  ([#43](https://github.com/imann128/tsauditor/issues/43), fixed in
  [#51](https://github.com/imann128/tsauditor/pull/51) by
  [@LuisMend12](https://github.com/LuisMend12))

- **`to_dict()` and `to_json()` no longer maintain two independent copies of
  the same payload.** ([#47](https://github.com/imann128/tsauditor/issues/47))
  The fix above patched the symptom, not the cause: `to_json()` built its
  `metadata`/`issues`/`counts` block as its own literal dict, separate from
  `to_dict()`'s, which is exactly how the `info`-count omission happened in
  the first place and could happen again the next time either one gains a
  field. `to_json()` now builds its payload from `to_dict()` and only adds
  the JSON-specific extras (`leaky_columns`, `panel`, `health`) on top, so
  the two structurally cannot drift apart again.

### Added: `fix()` accepts `available_at` and `constraints`

- **`tsa.fix()` can now run LEK004 (as-of leakage) and VAL001/VAL002
  (validity) as part of a one-shot repair.**
  ([#42](https://github.com/imann128/tsauditor/issues/42))
  Both checks are opt-in on `scan()` because tsauditor cannot infer a
  release schedule or a validity bound on its own, but `fix()` had no
  parameter to pass either through, so the only way to exercise them
  together with a one-shot repair was to call `scan()` and `apply_fixes()`
  separately. `fix()` silently skipped them with no error, which reads as
  "nothing wrong" rather than "not checked." `fix(df, available_at=...,
  constraints=...)` now forwards both to the underlying `scan()` call.

### Added: PNL004 reports rows with a null entity id

- **New issue code `PNL004` (WARNING) for panel rows where `group_col` is
  null.** ([#48](https://github.com/imann128/tsauditor/issues/48))

  `df.groupby(group_col)` drops null keys by default, which is the only sound
  choice for coverage/short-history comparisons: there is no entity identity
  to compare. But it meant rows with a null entity id received **zero**
  checks under `scan()` (not panel-level, not per-entity; they simply never
  entered the loop) and were silently left untouched by `apply_fixes()`,
  with nothing in the report saying so. A null-id row looked identical to a
  clean one.

  PNL004 reports the count and percentage of such rows up front. Behavior is
  otherwise unchanged: these rows are still excluded from every other check
  (there genuinely is nothing sound to compare them against) and still left
  unrepaired by `apply_fixes()` (no single entity's distribution to repair
  them from): the skip in `apply_fixes()` is now explicit and logged in
  `last_fixes` rather than an accidental consequence of `NaN != NaN`.

### Internal

- **Doctests now run in CI.** ([#39](https://github.com/imann128/tsauditor/issues/39))
  Illustrative snippets in `scan()`, `fix()`, and `GuardReport`'s docstrings
  reference a `df`/`report` from the reader's own session and are marked
  `# doctest: +SKIP`; the ones that are fully self-contained (e.g.
  `audit_equivalence`) are actually executed. Runs once per CI matrix
  (ubuntu, Python 3.11), not on every cell: this catches a docstring going
  stale after a signature or return-type change without adding 18x runtime.
- **Consolidated `_encode_target`.** ([#40](https://github.com/imann128/tsauditor/issues/40))
  `correlation.py` and `temporal.py` now import a shared `encode_target()`
  from the new `leakage/_common.py` instead of each carrying a byte-identical
  copy. `equivalence.py` keeps its own inline version deliberately: it
  forces any binary target to 0.0/1.0 (needed for its AUC math), which is a
  real behavioral difference from the shared helper's numeric-passthrough,
  not incidental duplication.

### Added: PRF007 reports infinite values

- **New issue code `PRF007` (CRITICAL) for `inf` and `-inf` in numeric columns,
  and `apply_fixes()` now removes them.**
  ([#46](https://github.com/imann128/tsauditor/issues/46))

  Infinities passed through the entire pipeline unreported and unrepaired.
  `isna()` is False for an infinity, so PRF002 and PRF006 never saw one, and
  every anomaly and leakage detector quietly replaced it with NaN on its own
  working copy so its arithmetic would not break. Nothing owned reporting it.
  A user could run `scan()`, see no relevant issue, run `fix()`, and still hand
  infinities to their model.

  The asymmetry, on identical 200-row frames with 10 consecutive bad values:

  |                     | codes reported                   | after `fix()` |
  | ------------------- | -------------------------------- | ------------- |
  | 10 consecutive NaN  | PRF002 (plus anomaly codes)      | 0 NaN left    |
  | 10 consecutive inf  | anomaly codes only, none about the infs | 10 inf left |

  **`fix()` could also multiply them.** Validated on real data: five features
  built from the OGDC series by ordinary feature engineering (`Returns /
  Return_lag1`, `log(Returns)`, `Volume / ChangeP`) produce 19 infinities across
  3 columns, because those denominators are genuinely zero on some days. Under
  0.3.0 `fix()` returned **35** of them: interpolating a NaN that neighbours an
  infinity propagates the infinity into the gap, so `log_ret` went from 6 to 22.
  On the same frame with PRF007, `fix()` returns 0.

  **No threshold parameter, by design.** PRF006 needs one because some
  missingness is normal and the question is how much is too much. That question
  does not arise here: an infinity is never a measurement, only the residue of a
  division by zero, an overflow, or a log of a non-positive number. One is a
  defect, so the threshold is one.

  **CRITICAL, matching PRF004.** It invalidates other checks rather than
  describing the data. A single inf makes a column's mean inf and its standard
  deviation NaN, and scikit-learn raises at `fit` time rather than degrading.

  Evidence includes `n_finite_remaining` and `below_leakage_min_obs`. The second
  is the one to read: below 30 finite observations the leakage detectors skip the
  column entirely rather than merely losing precision.

  **Repair.** `apply_fixes()` converts infinities to NaN and imputes them with
  genuine missing values. This step runs unconditionally, unlike every other
  repair, because there is no reading under which keeping an infinity is
  correct. With `missing=None` the cell is left as NaN, which is honest about the
  value being unknown rather than making a false claim about its size.

  Infinite cells are now counted by `affected_cells()` in their own right. In
  practice this rarely moves `health_score()`, because ANO003's spike mask
  already marked those positions: the deviation of an infinity from its local
  mean is infinite, so it always exceeds the spike threshold. Measured on real
  data (OGDC-derived features, 19 infinities across 3 columns) the score is 85.2
  either way. The change matters for correctness of attribution rather than for
  the number.

### Changed: LEK002 no longer reports leakage between independent columns

- **`audit_correlation_leakage` default `min_correlation` raised from 0.1 to 0.5.**
  ([#49](https://github.com/imann128/tsauditor/issues/49))

  LEK002 fires when the peak cross-correlation over lags lands at a positive lag.
  For two persistent series (a price level, a random walk, a slow AR process)
  spurious correlation is large by construction while *which* lag wins is close to
  a coin flip, so the old 0.1 gate was not a real gate. It reported leakage between
  columns that were statistically independent.

  Measured over 100 trials per cell on 400-point series. FP columns are two
  independently generated series, so every flag is a false positive. TP columns are
  a genuine t+1 lookahead:

  | `min_correlation` | FP random walk | FP AR(0.98) | TP i.i.d. | TP random walk |
  | ----------------- | -------------- | ----------- | --------- | -------------- |
  | 0.1 (until now)   | 37%            | 51%         | 100%      | 100%           |
  | 0.5 (new default) | 13%            | 8%          | 100%      | 100%           |

  No true positive was lost in 200 trials.

  **This is a behaviour change.** Features whose peak correlation at a positive lag
  falls between 0.1 and 0.5 are no longer flagged. Given the rates above, most such
  flags were noise. To restore the previous behaviour, pass
  `min_correlation=0.1` explicitly.

  This affects the `finance` preset most, because random-walk-like price series are
  exactly the case the old gate handled worst.

  13% and 8% remain higher than ideal. This is the validated improvement, not the
  final answer; the underlying rule still uses a bare argmax over lags.

## [0.3.0] - 2026-07-26

Panel (multi-entity) support, multivariate leakage detection, and two data-corruption
fixes. All additive to the public API (nothing was removed or renamed) so existing
single-series code runs unchanged.

**Headline:** `scan(df, group_col="ticker")` audits long-format panels entity by
entity; **LEK005** catches leakage that emerges from a *group* of features rather than
one; **PNL002** catches cross-sectional lookahead that the per-entity checks lose track
of once a common factor dominates. Two silent-corruption bugs are fixed: `apply_fixes`
filling one entity's gaps with another's values, and `audit_point_anomalies` crashing
on any frame with duplicate timestamps.

### Fixed: panel repair corrupted data across entities
- **`apply_fixes()` and `fix()` are now panel-aware.** When the report came from a
  `group_col=` scan, each entity is repaired as its own independent time series.

  Previously the frame was repaired as one interleaved series, which carried
  values across entity boundaries. Measured on a two-entity panel where one
  series sits near 10 and the other near 1000, a gap in the low series was filled
  with **~1000** (the other entity's values) silently and with no warning. It
  now fills with ~10, correctly.

  Write-back is positional rather than label-based, because a panel index repeats
  each timestamp once per entity and a `.loc` assignment would scatter one
  entity's repairs across all of them. Row order, index and shape are preserved
  exactly. Each `last_fixes` entry gains a `group` key. `leakage="drop"` removes a
  column frame-wide rather than once per entity. Single-series behaviour is
  unchanged.

### Added: multivariate leakage detection
- **LEK005: combination leakage.** A group of **two or three** features that
  together reconstruct the target while none does alone: the shape produced by a
  target defined as a difference, mean, spread, product or ratio. Every previous
  leakage check was univariate and structurally could not see this: with
  `target = x1 - x2` and independent inputs, each feature correlates with the
  target at only ~0.7, far below LEK001's 0.95. Runs by default whenever
  `target=` is given.

  Uses adjusted R² from an OLS fit, over two algebraic forms: **linear** (sums,
  differences) and **log** (products, ratios, since `log(a*b) = log a + log b`).
  Neither alone is sufficient; measured coverage: `x1-x2` scores 1.00 linear /
  0.01 log, `x1*x2` scores 0.93 / 1.00, `x1/x2` scores 0.83 / 1.00. An
  interaction term was tested as an alternative and rejected: it catches products
  but not ratios, and doubles the chance-level R². The form used is reported in
  `evidence["form"]`.

  Raw values rather than ranks, because the leakage is arithmetic and ranking
  destroys it: the canonical case scores 1.0000 raw but 0.9410 on ranks, which
  would slip under the threshold.

  The log form fits `log|y| ~ log|X|` on **absolute** values, so products and
  ratios of *signed* data are recovered too: `|a*b| = |a|*|b|` holds regardless
  of sign. On signed inputs the linear form scores 0.009 for a product, i.e.
  completely blind, while the absolute-log form scores 1.000. Columns whose
  values touch zero fall back to the linear form, since `log` of a near-zero
  would dominate the fit.

  **Groups larger than two, without O(k^n).** Rather than scanning C(k,3) groups
  (161,700 for 100 features), the search deepens iteratively: a group is extended
  by one column only when it reaches `gate` (0.30) without reaching the flagging
  threshold: the signature of a sub-group of a larger identity. If
  `target = a+b+c`, any pair from those three already scores ~0.71; inside a
  4-way identity a pair scores ~0.49. On random data nothing clears the gate, so
  deeper levels cost nothing and add no false positives.

  `max_group_size` defaults to 3. Raise it to find larger identities;
  `max_candidates_per_level` (default 200) bounds the cost: without it a frame of
  40 mutually correlated features took 21s at depth 4, and 0.7s with it.

  A group is skipped when any column *alone* already reaches the threshold, so a
  single leaky column cannot generate duplicate findings for something LEK001
  already reported. A triple is not reported when one of its own pairs was.

  Evidence keys are `group`, `group_size`, `group_adjusted_r2` and `form`
  (previously `pair` / `pair_adjusted_r2`).

  Validated on real data: in the OGDC example, `MACD_hist` is exactly
  `MACD - MACD_signal`, and LEK005 reports adjusted R² 1.0000 where the best
  single column reaches only 0.12. Zero false positives on that file's actual
  target. On random data the highest adjusted R² reached by chance across 1,225
  pairs was 0.075.

- **PNL002: cross-sectional lookahead.** The panel-native leak: a rank, z-score
  or sector-neutralised feature computed across entities at t+1 and joined back to
  t. Runs automatically for panel scans with a target.

  LEK002/LEK003 do detect this, but only while idiosyncratic variation dominates.
  As a common market factor grows, relative position decouples from each entity's
  own outcome and their detection falls from 100% of entities to 22.5%, while the
  cross-sectional signal is unaffected. Worse than a plain miss: at a 25:1 ratio
  LEK002 flagged the *legitimate* cross-sectional feature in 11 of 40 entities and
  the *leak* in 13 of 40 (no discriminating power) and `prevalence()` then
  reports a leak present in 100% of entities as affecting 32%, which reads as
  "isolated". Reproduce with `python docs/proposals/pnl002_evidence.py`.

  Calibrated against simulated factors: realistic rank-ICs (0.02-0.15) are not
  flagged; 0.30 and above are. Gated on 20+ co-present entities per timestamp and
  30+ scored timestamps. Vectorised: 0.05s for 40 entities where a per-timestamp
  loop took 4.0s, and 0.72s for 200 entities x 1500 timestamps.

### Added
- **Panel (long-format, multi-entity) support.** `scan(df, group_col="ticker")`
  partitions the frame by entity and audits each as its own independent time
  series; every issue is tagged with `Issue.group`. Previously a panel had to be
  scanned as one interleaved series, which made the structural, anomaly and
  rolling checks meaningless: a 21-point rolling window would span several
  entities at once. No detector logic changed: the partition happens in the
  scanner, so panel and single-series results stay consistent by construction.
- `GuardReport.prevalence()`: how widely each finding occurs across entities,
  the headline output of a panel scan. A 500-entity panel can produce tens of
  thousands of issues; what matters is whether a finding is systemic (100% of
  entities → pipeline bug) or isolated (a few percent → inspect those entities).
  `summary()` prints this instead of the full issue list for panels.
- `GuardReport.groups()`, `.groups_affected(code=, column=, severity=)`, and
  `.is_panel`; `filter()` gains `column=` and `group=` arguments.
- New `tsauditor.panel` module with `audit_panel_structure`:
  - **PNL001** (WARNING): ragged panel: entities do not share a common time
    index, which silently breaks every cross-sectional operation while looking
    fine per entity.
  - **PNL003** (INFO): entities below the 30-row `min_obs` floor, so their
    *absence* of findings is not evidence of health.
  - PNL002 is reserved for cross-sectional lookahead detection, proposed with
    measured evidence in `docs/proposals/pnl002-cross-sectional-leakage.md`,
    reproducible via `docs/proposals/pnl002_evidence.py`.
- `to_json()` gains a `panel` block (`group_col`, `n_groups`, `prevalence`) for
  panel scans. Single-series JSON output is unchanged.

- **ANO002 evidence gains `esd_outlier_count` and `masking_suspected`**, from
  Rosner's Generalized ESD test (1983). **Diagnostic only: flagging is unchanged.**

  This resolves a genuine ambiguity in the existing output. A zero
  `agreement_count` has two opposite causes: a harmlessly skewed column, or
  contamination heavy enough to blind the z-score half of the rule, and the
  counts alone cannot tell them apart. At 5% contamination the z-score reports 0
  while the IQR rule correctly finds every planted outlier, which looks identical
  to ordinary skew. ESD removes the most extreme point and *recomputes* the scale
  before testing the next, so masking cannot occur by construction; measured
  against planted outliers it is exact at every level (1, 5, 20, 50, 150, 300) and
  reports 0 on clean Gaussian data where the IQR rule gives 10 false positives.

  Computed only for the ambiguous case (z-score count 0, IQR count above 0) and
  reported as `None` otherwise, since it is O(k*n): about 27ms on 1,000 points.

### Changed
- **Threshold resolution is now consistent across detectors.** `domain` supplies
  *defaults*; an explicitly passed threshold always wins. Previously
  `audit_point_anomalies` consulted `domain` first and silently discarded an
  explicit `zscore_threshold`, so `audit_point_anomalies(df, zscore_threshold=2.0,
  domain="finance")` used 5.0. It now uses 2.0. Domain presets with no explicit
  threshold are unchanged (finance 5.0, sensor 3.5, None 4.0), and `scan()` is
  unaffected because it forwards only `domain`.
- Threshold resolution uses `is None` rather than the `x or default` idiom, so a
  deliberate `0` is honoured instead of being treated as "unset". Affects
  `stuck_window`, `spike_threshold`, and `spike_window` in
  `audit_contextual_anomalies`.

### Fixed
- **`audit_point_anomalies` crashed on any frame with duplicate timestamps** (i.e.
  on all panel data) with `TypeError: cannot convert the series to
  <class 'float'>`. `series.loc[z.idxmax()]` returns a Series rather than a
  scalar when the index has repeats. The worst point is now located
  positionally.
- PRF004 now distinguishes panel-shaped duplication (every timestamp repeating a
  uniform number of times) from a genuine duplication bug, and points the user at
  `group_col=` rather than reporting their well-formed panel as corrupt. The
  evidence carries `looks_like_panel` and `repeats_per_timestamp`.
- `audit_point_anomalies` now neutralises `inf` / `-inf` before computing
  statistics, as every other detector already did. Previously a single `inf` made
  the column mean `inf` and its standard deviation `NaN`, so all comparisons
  evaluated `False` and the column was skipped whole: hiding genuine outliers
  sitting alongside the `inf`, and emitting numpy `RuntimeWarning`s.
- `audit_point_anomalies` now guards against a `NaN` standard deviation
  (`std == 0 or pd.isna(std)`), matching the guard already present in its repair
  mirror `remediate._outlier_mask`.

### Documentation
- `CONTRIBUTING.md` now lists all three checks CI runs (`pytest -q`,
  `ruff check .` and `ruff format --check .`) rather than only the tests. It
  previously mentioned `pytest` alone, so a contributor could pass everything
  documented and still fail CI on formatting. Also states that `[dev]` is the
  extra to install, since a plain `pip install -e .` provides neither `ruff` nor
  the optional dependencies several test files need.
- `audit_equivalence` gained a worked `Examples` section in its docstring,
  showing a leaky feature caught alongside a clean one. Thanks to @azeque-art
  (#37).

### Internal
- New test `test_multiple_leaky_features_all_flagged` covers a case the suite
  missed: every existing equivalence test plants a *single* leaky feature, so
  none of them would notice the detector stopping after the first. Verified by
  mutation: breaking the loop to report only one column fails this test and no
  other. Thanks to @azeque-art (#37).
- Added `MANIFEST.in`. setuptools walks the working tree when building an sdist
  and does not read `.gitignore`, so a local `.venv` was traversed and
  `python -m build` failed with `RecursionError: maximum recursion depth
  exceeded`. The prunes make the build reproducible from a dirty working tree,
  not only a fresh clone.
- Added integration tests for the optional extras against the new panel state,
  which nothing previously covered: PDF export of a report containing
  `Issue.group` and PNL codes, joblib round-tripping of `is_panel` / `groups()`
  / `prevalence()`, and polars input combined with `group_col`. These only run
  when the `[pdf]` / `[polars]` / `[dev]` extras are installed, so a plain
  `pip install -e .` skips them: install `.[dev]` to exercise them locally as
  CI does.
- `test_scaffold.py` no longer asserts a hardcoded version string. It now checks
  that `tsauditor.__version__` is well-formed and **agrees with
  `pyproject.toml`**, which catches the real bug (bumping one and forgetting the
  other) and needs no edit at each release.
- CI reads the ruff pin from `pyproject.toml` instead of hardcoding it in the
  workflow. The two had drifted (`0.15.18` in CI vs `0.15.20` in `[dev]`), which
  meant contributors could format locally with a different ruff than CI checked
  against. Dependabot updates `pyproject.toml` only, so this keeps them in step.
- Added `tests/test_threshold_resolution.py` and `tests/test_edge_cases.py`
  (115 tests total; suite goes from 160 to 275) covering threshold precedence,
  degenerate columns (all-NaN, all-inf,
  constant, single-value), degenerate targets, and time-index edge cases (unsorted
  input, timezones, DST transitions, leap days, sub-second sampling, duplicate
  timestamps, string labels, numeric-index refusal).

## [0.2.0] - 2026-07-05

Feature release: as-of leakage detection, domain-validity checks, a one-shot
repair API, the TimesFM adapter, and the remediation/health/PDF/polars/joblib
work: all additive and backward compatible with 0.1.x.

### Added
- As-of / point-in-time leakage check (LEK004): `scan(df, available_at=...)` flags
  a feature whose value sits at a timestamp earlier than when it was actually
  published (macro releases, sentiment, earnings). Opt-in: availability cannot be
  inferred from values alone; declare it per column as per-row publish timestamps
  (a `pd.Series`) or a fixed publication lag (a `pd.Timedelta`). CRITICAL.
- Domain-validity checks (`validity` module): `scan(df, constraints=...)` verifies
  declared rules: per-column `bounds` (e.g. a spread must be strictly positive,
  sentiment within [-1, 1]; VAL001, WARNING) and `relations` such as `("bid","ask")`
  to catch a crossed book (VAL002, CRITICAL). Validity issues are not counted as
  leakage.
- TimesFM adapter: `tsa.adapters.to_timesfm(df, target_col=...)` audits, repairs,
  and formats a single series into a 1-D float32 array for Google TimesFM. Cleans
  the target as an ordinary column (not protected; it's the series to forecast),
  verifies the result is finite before returning (so no NaN reaches the model), and
  can return the audit trail via `return_report=True`. Adds no `timesfm` dependency.
- Example notebook `examples/new_features_walkthrough.ipynb` (built and executed by
  `examples/build_new_features_notebook.py`) demonstrating LEK004, validity checks,
  `tsa.fix`, and the TimesFM adapter end-to-end.
- `tsa.fix(df, target=..., domain=...)`: one-shot scan-and-repair convenience
  wrapper returning `(clean_df, report)`. Always returns both, so the audit trail
  (`report.last_fixes`, `leaky_columns()`, issue list) is never silently discarded.
  The original frame is untouched; `clean_df` is an independent copy.
- Performance: LEK002 cross-correlation rank-transforms each series once instead of
  re-ranking on every lag: ~12x faster on wide frames, with identical flags/peak-lags
  (verified). `scan(run_stationarity=False)` skips the ADF test (the runtime hot spot,
  ~6x faster full scan), and `audit_stationarity(max_lag=...)` caps the ADF lag search.
- polars support (issue #28): `scan()` accepts a polars DataFrame, converting to pandas
  at the boundary. polars has no index, so a polars input must pass `time_col=`: the
  error message says so. Optional `[polars]` extra; no new hard dependency.
- joblib/pickle hardening: `GuardReport` and `Issue` round-trip through `pickle`/`joblib`
  (tested), enabling `joblib.Parallel` audits across a symbol universe. README recipe added.
- `leakage` module fully implemented: LEK001 (rank-based target equivalence:
  Spearman for continuous targets, AUC separation for binary), LEK002 (positive-lag
  cross-correlation), LEK003 (rolling-window lookahead via excess-over-persistence).
- Test suites for the leakage module: `test_equivalence.py`, `test_correlation.py`,
  `test_temporal.py`, covering clean/leak/edge cases.
- Standard repository files: `README.md`, `LICENSE`, `CHANGELOG.md`, CI workflow.
- Advisory layer: `Issue.suggestion`, `GuardReport.suggestions()` and `leaky_columns()`.
- Report-driven auto-remediation: `GuardReport.apply_fixes(df, ...)` returns a repaired
  copy (original untouched), fixing only flagged columns: clip/NaN outliers, NaN+impute
  stuck runs, impute missing clusters, opt-in leakage-column drop. Records `report.last_fixes`.
  Contextual spikes (ANO003) are also repaired: clipped to their local rolling band or
  NaN-ed, distinct from global outlier (ANO002) bounds.
- Data Health Score: `GuardReport.health_score(df)` = % of numeric cells not implicated by
  quality issues (leakage excluded). Surfaced in `to_json` with affected/total cells and an
  optional before/after delta.
- PDF export: `GuardReport.to_pdf(path, df=..., fixed_df=...)`: a formal, vector,
  text-selectable report (Times New Roman, black text, headings, tables): Data Health
  Scorecard, dataset overview, before/after, target-leakage callout, executive summary,
  and a paginated issues table. No charts (visualising the series is left to the user)
  and no colour coding. Requires the optional `[pdf]` extra (`pip install 'tsauditor[pdf]'`).

### Fixed
- `scan()` (and `fix()` / `to_timesfm()` through it) no longer crashes on a constant
  numeric column. The ADF stationarity check (PRF003) now skips a zero-variance column
  instead of letting statsmodels' `adfuller` raise "Invalid input, x is constant", and
  guards other numerical failures so one column can never abort the whole audit.
- `apply_fixes` no longer touches the target column. A binary target trips ANO001 (long
  identical runs), and the fixer would NaN-and-interpolate the label into fractions;
  the target (`report.metadata["target"]`) is now excluded from every repair.
- `GuardReport.health_score(df)` re-scans the frame it is given instead of reusing the
  report's (possibly stale) issue list, so the "after" score on an `apply_fixes` output is
  correct. The re-scan skips leakage and ADF, which don't affect the score.
- ANO003 contextual spike detection no longer self-masks: rolling statistics exclude
  the current observation, use a wider window, and handle zero-variance context.
- `scan()` runs end-to-end now that all non-stub modules are implemented; stale
  scaffold tests updated to assert real behavior.
- `.gitignore` re-encoded from UTF-16 to UTF-8 so its patterns take effect.

## [0.1.0]

### Added
- Initial architecture: `profiler`, `anomaly`, `leakage` modules behind a single
  `tsa.scan()` entry point returning a `GuardReport`.
- Profiler checks (PRF001–PRF006), point anomalies (ANO002), CLI/JSON report output.
