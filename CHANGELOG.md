# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/), and the project adheres to
[Semantic Versioning](https://semver.org/).

## [0.3.0] - 2026-07-26

Panel (multi-entity) support, multivariate leakage detection, and two data-corruption
fixes. All additive to the public API — nothing was removed or renamed — so existing
single-series code runs unchanged.

**Headline:** `scan(df, group_col="ticker")` audits long-format panels entity by
entity; **LEK005** catches leakage that emerges from a *group* of features rather than
one; **PNL002** catches cross-sectional lookahead that the per-entity checks lose track
of once a common factor dominates. Two silent-corruption bugs are fixed: `apply_fixes`
filling one entity's gaps with another's values, and `audit_point_anomalies` crashing
on any frame with duplicate timestamps.

### Fixed — panel repair corrupted data across entities
- **`apply_fixes()` and `fix()` are now panel-aware.** When the report came from a
  `group_col=` scan, each entity is repaired as its own independent time series.

  Previously the frame was repaired as one interleaved series, which carried
  values across entity boundaries. Measured on a two-entity panel where one
  series sits near 10 and the other near 1000, a gap in the low series was filled
  with **~1000** — the other entity's values — silently and with no warning. It
  now fills with ~10, correctly.

  Write-back is positional rather than label-based, because a panel index repeats
  each timestamp once per entity and a `.loc` assignment would scatter one
  entity's repairs across all of them. Row order, index and shape are preserved
  exactly. Each `last_fixes` entry gains a `group` key. `leakage="drop"` removes a
  column frame-wide rather than once per entity. Single-series behaviour is
  unchanged.

### Added — multivariate leakage detection
- **LEK005: combination leakage.** A group of **two or three** features that
  together reconstruct the target while none does alone — the shape produced by a
  target defined as a difference, mean, spread, product or ratio. Every previous
  leakage check was univariate and structurally could not see this: with
  `target = x1 - x2` and independent inputs, each feature correlates with the
  target at only ~0.7, far below LEK001's 0.95. Runs by default whenever
  `target=` is given.

  Uses adjusted R² from an OLS fit, over two algebraic forms: **linear** (sums,
  differences) and **log** (products, ratios, since `log(a*b) = log a + log b`).
  Neither alone is sufficient — measured coverage: `x1-x2` scores 1.00 linear /
  0.01 log, `x1*x2` scores 0.93 / 1.00, `x1/x2` scores 0.83 / 1.00. An
  interaction term was tested as an alternative and rejected: it catches products
  but not ratios, and doubles the chance-level R². The form used is reported in
  `evidence["form"]`.

  Raw values rather than ranks, because the leakage is arithmetic and ranking
  destroys it — the canonical case scores 1.0000 raw but 0.9410 on ranks, which
  would slip under the threshold.

  The log form fits `log|y| ~ log|X|` on **absolute** values, so products and
  ratios of *signed* data are recovered too — `|a*b| = |a|*|b|` holds regardless
  of sign. On signed inputs the linear form scores 0.009 for a product, i.e.
  completely blind, while the absolute-log form scores 1.000. Columns whose
  values touch zero fall back to the linear form, since `log` of a near-zero
  would dominate the fit.

  **Groups larger than two, without O(k^n).** Rather than scanning C(k,3) groups
  (161,700 for 100 features), the search deepens iteratively: a group is extended
  by one column only when it reaches `gate` (0.30) without reaching the flagging
  threshold — the signature of a sub-group of a larger identity. If
  `target = a+b+c`, any pair from those three already scores ~0.71; inside a
  4-way identity a pair scores ~0.49. On random data nothing clears the gate, so
  deeper levels cost nothing and add no false positives.

  `max_group_size` defaults to 3. Raise it to find larger identities;
  `max_candidates_per_level` (default 200) bounds the cost — without it a frame of
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

- **PNL002: cross-sectional lookahead.** The panel-native leak — a rank, z-score
  or sector-neutralised feature computed across entities at t+1 and joined back to
  t. Runs automatically for panel scans with a target.

  LEK002/LEK003 do detect this, but only while idiosyncratic variation dominates.
  As a common market factor grows, relative position decouples from each entity's
  own outcome and their detection falls from 100% of entities to 22.5%, while the
  cross-sectional signal is unaffected. Worse than a plain miss: at a 25:1 ratio
  LEK002 flagged the *legitimate* cross-sectional feature in 11 of 40 entities and
  the *leak* in 13 of 40 — no discriminating power — and `prevalence()` then
  reports a leak present in 100% of entities as affecting 32%, which reads as
  "isolated". Reproduce with `python docs/proposals/pnl002_evidence.py`.

  Calibrated against simulated factors: realistic rank-ICs (0.02-0.15) are not
  flagged; 0.30 and above are. Gated on 20+ co-present entities per timestamp and
  30+ scored timestamps. Vectorised — 0.05s for 40 entities where a per-timestamp
  loop took 4.0s, and 0.72s for 200 entities x 1500 timestamps.

### Added
- **Panel (long-format, multi-entity) support.** `scan(df, group_col="ticker")`
  partitions the frame by entity and audits each as its own independent time
  series; every issue is tagged with `Issue.group`. Previously a panel had to be
  scanned as one interleaved series, which made the structural, anomaly and
  rolling checks meaningless — a 21-point rolling window would span several
  entities at once. No detector logic changed: the partition happens in the
  scanner, so panel and single-series results stay consistent by construction.
- `GuardReport.prevalence()` — how widely each finding occurs across entities,
  the headline output of a panel scan. A 500-entity panel can produce tens of
  thousands of issues; what matters is whether a finding is systemic (100% of
  entities → pipeline bug) or isolated (a few percent → inspect those entities).
  `summary()` prints this instead of the full issue list for panels.
- `GuardReport.groups()`, `.groups_affected(code=, column=, severity=)`, and
  `.is_panel`; `filter()` gains `column=` and `group=` arguments.
- New `tsauditor.panel` module with `audit_panel_structure`:
  - **PNL001** (WARNING) — ragged panel: entities do not share a common time
    index, which silently breaks every cross-sectional operation while looking
    fine per entity.
  - **PNL003** (INFO) — entities below the 30-row `min_obs` floor, so their
    *absence* of findings is not evidence of health.
  - PNL002 is reserved for cross-sectional lookahead detection — proposed with
    measured evidence in `docs/proposals/pnl002-cross-sectional-leakage.md`,
    reproducible via `docs/proposals/pnl002_evidence.py`.
- `to_json()` gains a `panel` block (`group_col`, `n_groups`, `prevalence`) for
  panel scans. Single-series JSON output is unchanged.

- **ANO002 evidence gains `esd_outlier_count` and `masking_suspected`**, from
  Rosner's Generalized ESD test (1983). **Diagnostic only — flagging is unchanged.**

  This resolves a genuine ambiguity in the existing output. A zero
  `agreement_count` has two opposite causes — a harmlessly skewed column, or
  contamination heavy enough to blind the z-score half of the rule — and the
  counts alone cannot tell them apart. At 5% contamination the z-score reports 0
  while the IQR rule correctly finds every planted outlier, which looks identical
  to ordinary skew. ESD removes the most extreme point and *recomputes* the scale
  before testing the next, so masking cannot occur by construction; measured
  against planted outliers it is exact at every level (1, 5, 20, 50, 150, 300) and
  reports 0 on clean Gaussian data where the IQR rule gives 10 false positives.

  Computed only for the ambiguous case (z-score count 0, IQR count above 0) and
  reported as `None` otherwise, since it is O(k*n) — about 27ms on 1,000 points.

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
- **`audit_point_anomalies` crashed on any frame with duplicate timestamps** —
  i.e. on all panel data — with `TypeError: cannot convert the series to
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
  evaluated `False` and the column was skipped whole — hiding genuine outliers
  sitting alongside the `inf`, and emitting numpy `RuntimeWarning`s.
- `audit_point_anomalies` now guards against a `NaN` standard deviation
  (`std == 0 or pd.isna(std)`), matching the guard already present in its repair
  mirror `remediate._outlier_mask`.

### Documentation
- `CONTRIBUTING.md` now lists all three checks CI runs — `pytest -q`,
  `ruff check .` and `ruff format --check .` — rather than only the tests. It
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
  mutation — breaking the loop to report only one column fails this test and no
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
  `pip install -e .` skips them — install `.[dev]` to exercise them locally as
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
work — all additive and backward compatible with 0.1.x.

### Added
- As-of / point-in-time leakage check (LEK004): `scan(df, available_at=...)` flags
  a feature whose value sits at a timestamp earlier than when it was actually
  published (macro releases, sentiment, earnings). Opt-in — availability cannot be
  inferred from values alone; declare it per column as per-row publish timestamps
  (a `pd.Series`) or a fixed publication lag (a `pd.Timedelta`). CRITICAL.
- Domain-validity checks (`validity` module): `scan(df, constraints=...)` verifies
  declared rules — per-column `bounds` (e.g. a spread must be strictly positive,
  sentiment within [-1, 1]; VAL001, WARNING) and `relations` such as `("bid","ask")`
  to catch a crossed book (VAL002, CRITICAL). Validity issues are not counted as
  leakage.
- TimesFM adapter: `tsa.adapters.to_timesfm(df, target_col=...)` audits, repairs,
  and formats a single series into a 1-D float32 array for Google TimesFM. Cleans
  the target as an ordinary column (not protected — it's the series to forecast),
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
  re-ranking on every lag — ~12x faster on wide frames, with identical flags/peak-lags
  (verified). `scan(run_stationarity=False)` skips the ADF test (the runtime hot spot,
  ~6x faster full scan), and `audit_stationarity(max_lag=...)` caps the ADF lag search.
- polars support (issue #28): `scan()` accepts a polars DataFrame, converting to pandas
  at the boundary. polars has no index, so a polars input must pass `time_col=` — the
  error message says so. Optional `[polars]` extra; no new hard dependency.
- joblib/pickle hardening: `GuardReport` and `Issue` round-trip through `pickle`/`joblib`
  (tested), enabling `joblib.Parallel` audits across a symbol universe. README recipe added.
- `leakage` module fully implemented: LEK001 (rank-based target equivalence —
  Spearman for continuous targets, AUC separation for binary), LEK002 (positive-lag
  cross-correlation), LEK003 (rolling-window lookahead via excess-over-persistence).
- Test suites for the leakage module: `test_equivalence.py`, `test_correlation.py`,
  `test_temporal.py`, covering clean/leak/edge cases.
- Standard repository files: `README.md`, `LICENSE`, `CHANGELOG.md`, CI workflow.
- Advisory layer: `Issue.suggestion`, `GuardReport.suggestions()` and `leaky_columns()`.
- Report-driven auto-remediation: `GuardReport.apply_fixes(df, ...)` returns a repaired
  copy (original untouched), fixing only flagged columns — clip/NaN outliers, NaN+impute
  stuck runs, impute missing clusters, opt-in leakage-column drop. Records `report.last_fixes`.
  Contextual spikes (ANO003) are also repaired: clipped to their local rolling band or
  NaN-ed, distinct from global outlier (ANO002) bounds.
- Data Health Score: `GuardReport.health_score(df)` = % of numeric cells not implicated by
  quality issues (leakage excluded). Surfaced in `to_json` with affected/total cells and an
  optional before/after delta.
- PDF export: `GuardReport.to_pdf(path, df=..., fixed_df=...)` — a formal, vector,
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
