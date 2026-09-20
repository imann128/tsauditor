# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/), and the project adheres to
[Semantic Versioning](https://semver.org/).

## [0.6.0] - 2026-09-20

### Added: `audit_combination_leakage` gained a `seed` parameter for the binary-target AUC path

The binary-target extension shipped earlier in this release (`_binary_combination_auc`) fits K-fold
cross-validated splits and draws 200 permutation shuffles, both driven by `numpy.random.default_rng`.
Until now that generator was seeded unconditionally inside the function, so there was no way for a
caller to ask for a second, independent draw on a borderline finding, and no way for a test to pin a
specific run.

`audit_combination_leakage(..., seed: int = 0)` is new, threaded straight through to
`_binary_combination_auc`. Default `0` reproduces the exact behavior this function already had, so this
is purely additive and changes nothing for an existing caller. Only the binary-target path uses it; a
continuous target is scored by plain OLS, which has no randomness to seed.

`tests/test_combination.py` gained `test_binary_path_is_deterministic_by_default` (same seed must give
a bit-identical score) and `test_binary_path_seed_param_changes_the_draw_but_not_the_verdict` (a
different seed can shift the exact score but must agree on flag versus no-flag for an unambiguous leak).

### Added: test coverage for the binary-target AUC path in LEK005 (previously untested)

The binary-target extension to `audit_combination_leakage` (added earlier in this release) shipped with
no dedicated tests at all. It is a randomized statistical procedure (K-fold cross-validation plus a
permutation test), so its correctness is a rate across many trials, not a single true/false assertion,
and that gap meant a regression in it could only have been caught by chance.

`tests/test_combination.py` now covers, among other cases: detection of a clean two-column and
three-column binary threshold-rule combination; confirmation that the same construction is invisible to
plain adjusted R-squared (the whole justification for the path existing, since R-squared alone cannot
clear the point-biserial ceiling of roughly 0.637 against a binary target); a bounded false-positive
rate swept across 60 independent trials of unrelated random features against an unrelated binary
target; that the single-feature guard uses AUC rather than R-squared for a binary target, the same way
LEK001 does; that `_kfold_fitted`'s degenerate-fold fallback does not crash when training rows are near
the design's own floor; and that the permutation test's cost (200 refits per candidate) stays off the
bulk of a scan, mirroring the existing triples-cost test for the continuous path.

### Added: LEK005 (`audit_combination_leakage`) now catches combination leakage into a binary target

Adjusted R² (LEK005's only metric until now) has the same point-biserial
ceiling against a binary target that LEK001 had to work around for a single
column (see `leakage/equivalence.py`'s module docstring): it cannot exceed
roughly 0.637, however perfectly a group's linear combination actually
determines the class through a threshold rule (`target = 1{a - b > 0}` is
exactly this shape). A genuine binary combination leak cleared LEK005's
`gate` (0.30) comfortably but could never reach the default `threshold`
(0.95) through R² alone, so it was never flagged, regardless of how
`threshold` was set.

Added a second scoring path, used only when the target is binary and a
candidate's plain R² clears `gate` without reaching `threshold`: fit the
group's linear combination with 5-fold cross-validation (`_kfold_fitted`,
literal per-fold refitting, not the closed-form leave-one-out shortcut.
That shortcut was tried and rejected because its per-point prediction is
algebraically linear in that point's own label, which manufactures rank
separation even against a target the group carries zero information about;
measured mean \|corr(fitted, y)\| of 0.27 for the LOOCV shortcut against an
independent random target and random features, versus under 0.05 for
literal 5-fold refitting on the same data), score the out-of-fold fit's AUC
separation against the target, and validate that AUC against its own
permutation null (200 reshuffles, refit and rescore identically each time,
significant at p < 0.01) rather than a fixed cutoff. AUC's small-sample
variance is too large at this module's `min_obs` floor for a fixed
threshold to be well-calibrated the way it is for adjusted R² (measured
mean in-sample AUC of 0.63, max 0.77, over 50 trials of two independent
random features against an independent random binary target at n=30, from
chance alone).

Verified by direct simulation: a two-feature exact threshold-rule
reconstruction of a balanced binary target was flagged 30/30 trials by the
new path (0/30 under R² alone, since adjusted R² tops out near 0.63-0.65
for that construction and never reaches the default 0.95 threshold); an
adversarial sweep of 10 mutually independent random features against an
independent random binary target, all 45 candidate pairs run through the
real gate/threshold/permutation pipeline across 30 trials, produced 0 false
flags. A finding reached this way reports `evidence["form"] = "linear-auc"`
and `evidence["metric"] = "cv_auc_separation"` instead of `"adjusted_r2"`;
`evidence["group_score"]` is added alongside the existing
`group_adjusted_r2` key so a `"linear-auc"` finding's actual flagging value
is available under an unambiguous name (`group_adjusted_r2` still holds the
same number for continuity, but it is an AUC-separation value, not an R²,
for this form). No existing evidence key was removed or renamed, and every
non-binary-target LEK005 finding is scored exactly as before.

### Fixed: ESD's masking-recovered points were reported but never actually flagged or repaired

ANO002's Generalized ESD diagnostic (`0.3.0`) told you when the z-score
rule looked blind because heavy contamination had inflated the standard
deviation enough to mask itself (`evidence["masking_suspected"]`), and
named which points ESD itself considered outliers
(`evidence["esd_outlier_count"]`), but that information stopped at
evidence. Flagging stayed z-score OR IQR regardless, so a column under
confirmed masking could report `masking_suspected: True` and
`esd_outlier_count: 50` while the issue's own `combined_mask` (and
therefore `apply_fixes()`/`fix()`'s repair) still touched none of those 50
points. A reader who trusted `scan()`'s own flagged-point count, rather than
independently re-reading `esd_outlier_count` and manually deciding what to
do about it, would have repaired nothing on the exact column the diagnostic
existed to warn about.

Fixed on both sides. Detection (`anomaly/point.py`): when
`masking_suspected` is `True`, ESD's own flagged positions are folded into
`combined_mask`, so they are now part of what the ANO002 issue actually
reports as anomalous; `evidence["esd_recovered_count"]` counts how many of
them weren't already caught by z-score or IQR (0 unless masking is
suspected). Repair (`remediate.py`): `_outlier_mask` (used by
`outliers="nan"`/`"drop"`) and a new `_esd_masking_repair` (used by
`outliers="clip"`) apply the identical recovery, via a shared
`esd_masking_recovery` helper both modules now call: not two independent
re-derivations of "masking suspected," the same failure mode the
`anomaly/_common.py` centralization (`0.5.0`) already exists to prevent.
`_generalized_esd` and the new `esd_masking_recovery`/`EsdRecovery` moved
from `anomaly/point.py` into `anomaly/_common.py` accordingly (`point.py`
still imports and re-exports `_generalized_esd` for backward compatibility).

For `outliers="clip"`, ESD-recovered points are set to NaN and left to the
`missing` imputation step rather than clipped, logged separately as
`esd_masked_outliers_to_nan` in `last_fixes`. An earlier internal version
of this repair instead invented a clip band from `_generalized_esd`'s
removal sequence; adversarial simulation (n=200, contamination fraction in
{0.15, 0.2, 0.246, 0.3}, magnitude in {4.0, 4.5, 5.0, 5.58, 6.0}, 59 seeds)
found that band left the point's value completely unchanged in 24 of 951
masking-suspected cases, i.e. exactly the "flagged but not actually
repaired" bug this fix exists to close, just moved one step later. NaN has
no such gap. `_generalized_esd`'s own return signature changed accordingly:
`(count, positions, bound)`, with `bound` now always `None`, since no
single `(mean, std, critical)` triple can be guaranteed to exclude every
point Rosner's sequential test flags; this was verified directly against
the same adversarial sweep before removing the field's contents rather than
assumed from the algorithm's description.

`tests/test_anomaly.py` and `tests/test_remediation.py` gain coverage for:
detection now flagging ESD-recovered points (previously evidence-only);
`esd_recovered_count`'s value under masking vs. its absence when not
suspected; `outliers="nan"`/`"drop"`/`"clip"` all actually changing the
recovered cells' values, not just reporting them as detected; and a
regression test that a point flagged by both IQR and ESD is clipped, not
NaN-ed a second time, so the new repair path never overrides a perfectly
good ordinary clip for a point that never needed the ESD fallback.

**Known limitation, not addressed here.** If IQR itself finds nothing
(contamination heavy enough to defeat even Tukey's fence), there is no
z/IQR mask for the recovery step to extend, and the column is skipped with
no ANO002 issue at all, however contaminated it actually is. This is a
harder, separate case from ordinary z-score masking and is left open.

### Fixed: an explicit `zscore_threshold=`/`stuck_window=`/`spike_threshold=`/`spike_window=`/`handle_missing=` override silently never reached repair

`scan(df, domain="finance", zscore_threshold=6.0)` used your explicit `6.0`
for detection, but `apply_fixes()`, `affected_cells()`, and therefore
`health_score()` had no way to know an override had been passed at all:
none of these five settings were ever recorded in `report.metadata`, so
repair always fell back to the `domain` preset (`5.0` for `"finance"`)
regardless of what `scan()` actually used. Repair could therefore run
against a materially different mask than the one that produced the
report's own issues, or (for `spike_threshold`/`spike_window` together
with `handle_missing="interpolate"`) miss a spike detection's own gap-
bridging behavior entirely, since `apply_fixes`'s spike-repair pass used to
always `dropna()` the raw series rather than mirroring
`audit_contextual_anomalies`'s own `handle_missing`-aware bridging.

Fixed by recording all five in `report.metadata` from `scan()` regardless
of entry point, and adding `remediate._resolve_detector_settings(report)`,
which `apply_fixes`/`affected_cells` now call: an explicit (non-`None`)
value in `report.metadata` wins, a genuinely unset entry falls back to the
domain preset, the same "`None` means unset, not falsy" precedence the
detectors themselves already use internally, applied consistently to the
repair side for the first time. A report built from an older or
hand-constructed `metadata` dict without these keys degrades gracefully to
the previous domain-only behavior via `.get(..., None)`, not a `KeyError`.

`fix()` gains matching `zscore_threshold=`, `stuck_window=`,
`spike_threshold=`, `spike_window=`, and `handle_missing=` keyword
arguments, forwarded to its internal `scan()` call. Before this, tuning
detection for a one-shot `fix()` call meant calling `scan()` and
`apply_fixes()` separately, since `fix()` had no way to pass these through
at all. `apply_fixes()` itself needs no new argument for them: like
`group_col`, it reads them back off `report.metadata`.

`tests/test_fix.py` and `tests/test_health.py` gain coverage asserting an
explicit override is honored by `apply_fixes`/`affected_cells`/
`health_score`, not just by `scan()`'s own detectors, and that the
`handle_missing="interpolate"` spike-repair bridging now matches detection
exactly.

### Fixed: PNL002 (cross-sectional lookahead) had the identical persistence-collapse bug as LEK003

Found while double-checking LEK003's Fisher-z fix (below) for similar constructions
elsewhere in the codebase. `audit_cross_sectional_leakage`'s `expected(k) = |contemporaneous|
* |cs_persistence(k)|` / `excess(k) = |observed(k)| - expected(k)` is the cross-sectional
analogue of LEK003's formula, and shares its structural flaw: correlation is bounded in
[-1, 1] and compresses near the endpoints, so once a panel's cross-sectional ranking is
highly persistent (a slow-moving characteristic, e.g. sector, size tier, factor loading, not
just a literal random walk), both a legitimate and a leaky feature's `observed(k)` sit near
the same ceiling as `expected(k)`, and the raw subtraction between them loses almost all
resolving power.

Verified independently by direct simulation, not assumed from the LEK003 parallel: a panel
whose entities' underlying levels each follow their own AR(1) path (so cross-sectional
persistence decays with lag, the cross-sectional counterpart of LEK003's own φ sweep) shows
the identical curve: an obvious one-step cross-sectional leak caught 100% of the time
through φ=0.9, then undetected from φ=0.95 through a cross-sectional near-random-walk
(φ=1.0). A first simulation attempt used a static, non-decaying fixed-effect design instead
of an AR(1)-driven one; that construction produced false positives on an *honest* trailing
feature at moderate φ, traced to a separate cause (the multiplicative `expected(k)` bound
only holds under a decaying persistence structure; a flat, non-decaying one isn't a fair
analogue of LEK003's φ sweep) and discarded before drawing any conclusion from it, rather
than left as an unexplained result.

**Fixed** the same way as LEK003: `excess(k) = arctanh(|observed(k)|) - arctanh(expected(k))`,
Fisher-z transforms instead of raw correlations. `excess_threshold`'s default (0.15) needed
no change: arctanh is close to the identity at the small-to-moderate correlations this
module's own pre-existing tests exercise, and all 24 of them still pass unchanged. Swept
again after the fix: 100% detection at every φ tested including φ=1.0 (previously 0%), no
measured increase in false positives on an honest trailing feature at the same persistence
level. `tests/test_panel.py` adds `test_pnl002_catches_cross_sectional_leak_across_persistence_levels`
(parametrized φ = 0.7/0.9/0.95/0.99/1.0, confirmed to fail pre-fix: the leak went
undetected from φ=0.95 on, with this exact generator and seed) and
`test_pnl002_honest_trailing_feature_not_flagged_near_random_walk`. `Issue.evidence` for
PNL002 gained `excess_scale: "fisher_z"`, documenting the units `excess` and
`excess_threshold` are now both in; no existing evidence key was removed or renamed.

### Fixed: `n_jobs` raised a bare `ModuleNotFoundError` instead of naming the extra to install

`scan(df, group_col=..., n_jobs=2)` did `from joblib import Parallel, delayed` with no
guard, so a caller without joblib installed got a raw `ModuleNotFoundError: No module named
'joblib'` pointing at `scanner.py`'s internals, with no indication this is an expected, optional
dependency, or what to install. `joblib` was also never given its own extra: it was listed
only under `dev`, meant for contributors running the test suite, not something a real
caller adding `n_jobs=` to their own code would think to reach for. Compare
`tsauditor.report.pdf`'s `_require_matplotlib`, which has always guarded matplotlib's
identical optional-import situation with a clear message. This was the one optional
dependency that never got the equivalent treatment.

Fixed by adding `scanner._require_joblib()`, mirroring `_require_matplotlib` exactly, and a
new `parallel` extra (`pip install 'tsauditor[parallel]'`) in `pyproject.toml`; `dev`
already depended on `joblib` directly and is unchanged.
`test_n_jobs_without_joblib_raises_actionable_import_error` in `tests/test_panel.py` pins
the message, simulating joblib's absence with `sys.modules["joblib"] = None` (the standard
technique for forcing `import joblib` to raise without actually uninstalling it from the
test environment) rather than leaving this path untested, which it previously was.

### Added: `n_jobs` parallelizes panel scans internally

[#61](https://github.com/imann128/tsauditor/issues/61): `scan(df,
group_col=...)` audited every entity sequentially, one at a time, in this
process, even though the identical per-entity workload already parallelizes
cleanly through the README's external joblib recipe: a caller with a real
long-format panel (many entities in one frame, not separate frames per
entity) had no equivalent lever short of manually re-partitioning back into
separate DataFrames first, defeating the point of `group_col` in the first
place. Reported concretely: a 164k-row, 3,617-entity panel (8-90 rows per
entity) took roughly 12 minutes wall-clock.

`scan()` now accepts `n_jobs` (default `1`, unchanged sequential behavior).
When `group_col` is set and `n_jobs != 1`, entities are audited through
`joblib.Parallel(n_jobs=n_jobs, batch_size="auto")` instead of a plain
Python loop: `joblib`'s adaptive batching is what actually addresses the
issue's own stated concern about per-task dispatch overhead dominating for
many small entities, rather than picking one process per entity
unconditionally. Issues are collected back in the same per-entity,
sorted-by-key order the sequential path produces regardless of worker
count or completion order, so `n_jobs` changes nothing about a report's
content or ordering, verified directly:
`test_n_jobs_matches_sequential_result` in `tests/test_panel.py` asserts
structural equality (`Issue` is a plain dataclass) between an `n_jobs=1`
and an `n_jobs=2` scan of the same panel, and
`test_n_jobs_dispatches_through_joblib_parallel` proves parallel dispatch
actually happens (a spy on `joblib.Parallel`) rather than `n_jobs` being
silently accepted and ignored, which output equality alone can't
distinguish.

Measured on a synthetic panel matching the issue's shape (3,617 entities,
8-90 rows each), on a 2-core sandbox, with `stationarity_max_lag` left
unset (full ADF autolag search, the worst case) but the ESD-vectorization
fix below already baked in unconditionally: sequential (`n_jobs=1`)
extrapolates to ~274s from a 300-entity subsample, already well under the
reported ~720s; `n_jobs=2` extrapolates to ~161s, a 1.71x speedup limited
by this sandbox's 2 cores. A machine with more cores available should see
correspondingly more, and `stationarity_max_lag` stacks with `n_jobs` for a
further cut on top of either alone.

### Fixed: LEK002 crashed outright on any series at or below `max_lag` in length

**Found while building the `n_jobs` benchmark above, on a synthetic panel
deliberately shaped like the real one from
[#61](https://github.com/imann128/tsauditor/issues/61) (entities as short
as 8 rows).** `audit_correlation_leakage`/`lag_correlation_matrix` (LEK002)
raised a raw `ValueError: operands could not be broadcast together with
shapes (0,) (N,)` for any series no longer than `max_lag` (default 10), a
routine size for a panel entity, not an edge case: this package's own
default panel shape guidance ("500 stocks, 200 sensors, 50 stores") and
`min_obs=30`'s own default both implicitly assume entities well over 10
rows, but nothing enforced that, and `scan(group_col=...)` runs the leakage
suite on every entity regardless of length.

Root cause: `_align(a, b, tau)` slices two arrays so element *i* pairs
`a_t` with `b_{t+tau}`. For `tau >= 0` it computed `a[: n - tau]`; once
`tau > n`, `n - tau` goes negative, and Python's *stop*-slice does not
clamp a negative stop to zero the way a *start*-slice does: it wraps and
reads from the end instead, returning `abs(n - tau)` elements instead of
the empty array the code needed. Paired against `b[tau:]`, which already
(correctly) came back empty, the two arrays landed at different lengths,
and `np.isnan(a) | np.isnan(b)` raised before the existing `mask.sum() <
min_obs` guard (written for exactly this case) ever got a chance to skip
the lag. The mirror-image bug existed on the negative-`tau` branch's
`b[: n - s]`.

Fixed by clamping both stop-slice bounds to zero explicitly
(`max(n - tau, 0)` / `max(n - s, 0)`), so every out-of-range lag now
produces two genuinely empty, shape-matched arrays and falls through to the
`min_obs` skip as originally intended, instead of crashing before reaching
it. `tests/test_correlation.py` adds
`test_series_shorter_than_max_lag_does_not_crash` (parametrized n = 1, 2,
5, 9, 10, 11, 15, spanning the boundary in both directions) and
`test_short_series_lag_values_agree_with_a_longer_equivalent_slice`
(confirms the clamped-empty lags are cleanly `NaN`, not merely
non-crashing). Verified to fail with the exact pre-fix `ValueError` when
the clamp is reverted, then pass again restored. This bug had zero prior
test coverage anywhere in the suite, since every existing LEK002 test used
series comfortably longer than the default `max_lag`.

### Fixed: three bugs found in a dedicated correctness sweep, all reproduced before fixing

**1. Detector-tuning overrides never reached `apply_fixes`/`health_score`/`to_json`/`to_pdf`, silently
falling back to the domain-only preset (most severe of the three).** `scan()` accepts
`zscore_threshold=`/`stuck_window=`/`spike_threshold=`/`spike_window=`/`handle_missing=` and correctly
used them for *detection*, but never recorded them in `report.metadata`. Only `domain` was kept. Every
downstream re-derivation of "what counts as anomalous" (`apply_fixes`'s internal repair masks,
`affected_cells()`, `GuardReport.health_score()`'s and `to_json()`'s/`to_pdf()`'s internal re-scan calls)
re-derived its threshold from `domain` alone, silently discarding any explicit override the original
`scan()` call had used. A caller who passed `scan(df, stuck_window=2)` because the domain-preset window
(5) missed a real stuck run got a report that correctly flagged it, and a repair, and a health score,
and a JSON/PDF export, that all quietly went back to window 5 and left it unrepaired or unscored.
Detection and repair disagreeing about the same input is exactly the "drift" class of bug this
codebase's own module docstrings and past CHANGELOG entries warn about; this was a new instance of it.

Fixed by recording all five tuning parameters in `metadata` alongside `domain`, and adding a single
`_resolve_detector_settings(report)` helper in `remediate.py` (explicit override wins, `None` still falls
through to the domain preset, the same "is None, not falsy" precedence the detectors themselves use) that
`apply_fixes`, `affected_cells`, `health_score()`, `to_json()`, and `export_pdf()` all now call instead of
each re-deriving thresholds independently. `fix()` also gained the same five parameters so a one-shot
`tsa.fix(df, stuck_window=2, ...)` call can set them without a separate `scan()`.
`tests/test_fix.py` adds `test_stuck_window_override_is_forwarded_to_apply_fixes` (a
`stuck_window=2`-only run flags a 3-point run that the domain-default window of 5 would have missed
entirely, and proves `apply_fixes` actually repairs it) and
`test_zscore_threshold_override_is_forwarded_to_health_score` (a strict `zscore_threshold=2.0` override
catches more z-score outliers than the default preset, and `health_score()`'s re-scan reflects the same
stricter picture). Both verified to fail with the exact pre-fix behavior when the metadata forwarding is
reverted.

**2. PNL001 (ragged panel) coverage was inflated by rows with no entity id.** `audit_panel_structure`
computed `n_all` (the panel's total distinct timestamp count, used as the baseline every entity's coverage
is compared against) from the raw, unfiltered `df.index`, including rows whose `group_col` value is null.
A null-entity row belongs to no entity and is never scanned per-entity (PNL004 documents this exclusion
explicitly), so a timestamp that exists *only* on such a row is not a timestamp any real entity could ever
have coverage for. Left unfiltered, those timestamps inflated `n_all`, and therefore every real entity's
reported shortfall: a panel where every real entity has complete, identical coverage of each other could
be reported as fully ragged (`n_complete_groups=0`) purely because null-entity rows existed at timestamps
outside every real entity's own range, with none of those entities' actual data having changed at all.

Fixed by restricting `all_timestamps` to the same non-null rows PNL004 already computes
(`df.loc[~null_mask].index.unique()`), not the raw frame. `tests/test_panel.py` adds
`test_pnl001_ignores_null_entity_timestamps_when_computing_coverage`: a genuinely complete entity (AAA)
alongside a genuinely short one (BBB) plus null-entity rows outside AAA's range: verified AAA still
reports complete (`n_complete_groups == 1`) and `n_timestamps` still excludes the 10 null-entity
timestamps, and verified to fail (`n_timestamps` inflates to 110, `n_complete_groups` drops to 0) when the
filter is reverted.

**3. LEK004's tz-mismatch guard rejected two different-but-comparable timezones, not just genuine
aware/naive mismatches.** `_availability`'s explicit tz-mismatch check (added to turn a confusing raw
pandas `TypeError` into a clear message) compared `index.tz != avail.dt.tz` directly: two tzinfo objects.
That also raised for, e.g., a UTC index against a `US/Eastern` availability Series, even though pandas
compares those perfectly well by converting to a common instant; only an aware-vs-naive mismatch is
actually unresolvable. The guard was rejecting valid, comparable data as if it were the error it exists to
catch.

Fixed by comparing awareness (`index.tz is not None`) rather than tzinfo identity: the check now fires
only when one side is aware and the other naive. `tests/test_asof.py` adds
`test_different_but_both_aware_timezones_do_not_raise` (UTC index vs. `US/Eastern` availability, published
before use in both, must not raise) and `test_aware_vs_naive_still_raises` (the genuine mismatch the
guard exists for must still be caught). Verified the first fails with the original `ValueError` when the
comparison is reverted to direct tzinfo equality.

### Added
- Lead/lag cross-correlation heatmap page in `GuardReport.to_pdf()`: renders the
  full feature x lag Spearman grid that LEK002 searches, so a reader can see
  *why* a feature was (or nearly was) flagged instead of only its reported
  peak lag. Built on a new public
  `tsauditor.leakage.correlation.lag_correlation_matrix(df, target, max_lag=,
  min_obs=)`; `audit_correlation_leakage` was refactored to consume the same
  underlying `_lag_correlation_core` so the heatmap and the detector cannot
  silently disagree about the same numbers: verified directly in
  `tests/test_correlation.py` (`test_matrix_peak_matches_flagged_issue` and
  friends compare the two against each other, not against separately
  recomputed expected values). Truncated to `heatmap_top_n` features (default
  30, ranked by peak |correlation|) with the truncation stated on the page;
  blank cells (below `min_obs`, or a constant subset) render as light grey,
  not a misleading zero. Opt out with `to_pdf(...,
  include_correlation_heatmap=False)`.

  Not yet supported for panel scans (`scan(..., group_col=...)`): per-entity
  vs. pooled aggregation for a panel heatmap is a real design decision that
  hasn't been made, so the PDF prints a one-line explanatory note on the
  Findings page instead of guessing with a pooled-across-entities heatmap.

  This is the one chart in an otherwise black-and-white, no-colour-coding
  report: `tsauditor/report/pdf.py`'s module docstring explains why this
  page specifically earns the exception.

### Performance: two concrete bottlenecks found by profiling a real-world-shaped scan

Prompted by a user report of a 12-minute `scan()` on a 3,279-row, 27-column
dataframe, far slower than a dataframe that size should take. Profiled with
`cProfile`/`pstats` on a synthetic dataframe matching that shape (25 numeric
feature columns plus a binary target, deliberately noisy so every check has
real work to do) before changing anything, rather than guessing at what was
slow.

- **`_generalized_esd` (the Rosner ESD diagnostic behind ANO002's
  `masking_suspected` field) called `scipy.stats.t.ppf` as a scalar, once per
  removal step, up to `0.4 * n` times per ambiguous column.** Every one of
  those `criticals[i]` values is a pure function of `n`, the step index, and
  `alpha`; it never depends on the actual data, so there was no reason to
  compute it inside the per-column, per-step loop at all. On the benchmark
  dataframe this scalar `ppf` loop alone accounted for roughly two-thirds of
  `audit_point_anomalies`'s total time (~6.9s of a 12.1s scan). Replaced with
  one vectorized call that computes the full `criticals` array up front,
  before the (inherently sequential, see the function's own docstring for
  why) removal loop starts. Verified bit-identical to the original elementwise
  loop across n = 50 / 500 / 3,279 / 10,000. The sequential removal loop
  itself (recomputing mean/std after each point is dropped, which is the
  actual point of Rosner's test) is unchanged.

  Measured on the benchmark dataframe: `audit_point_anomalies` dropped from
  ~6.9s to well under 0.1s for this piece; the full `scan()` went from ~12.1s
  to ~5s.

- **`scan()` never exposed `audit_stationarity`'s own `max_lag` parameter.**
  The ADF autolag search (`statsmodels.tsa.stattools.adfuller(...,
  autolag="AIC")`) is `scan()`'s single most expensive check by construction:
  it fits a separate OLS regression for every candidate lag it tries, and
  the default lag ceiling grows with the sample size. `audit_stationarity`
  itself already accepts `max_lag` to cap that search, but `scan()` had no
  parameter forwarding it, so the only way to use it was to bypass `scan()`
  entirely and call `audit_stationarity` directly: an existing lever every
  `scan()` caller was locked out of. Added `stationarity_max_lag` to
  `scan()`'s signature, forwarded through `_ScanOptions` to the
  `audit_stationarity` call in `_run_checks`. Default is `None` (unchanged
  behavior: an uncapped autolag search), so this is purely additive.
  `test_scan_forwards_stationarity_max_lag` in `tests/test_scaffold.py` pins
  the forwarding with a spy on `audit_stationarity`, verified to fail
  (`assert [None] == [4]`) with the forwarding line reverted.

  Measured on the benchmark dataframe with `stationarity_max_lag=4`: ~5s
  combined with the ESD fix above down to ~2.4s, roughly a 5x improvement
  over the original 12.1s.

  **Not yet resolved: neither fix explains a 12-minute real-world runtime on
  data this size.** ~2.4-5s on a synthetic dataframe of the same shape is
  nowhere near 720s. Two differences between the synthetic benchmark and real
  data could plausibly account for the gap and haven't been ruled out: (1) a
  slower BLAS backend on the affected machine (the reference LAPACK build vs.
  an optimized OpenBLAS/MKL build changes SVD-heavy OLS fits (used by both
  `adfuller`'s autolag search and `combination.py`'s LEK005) by a large,
  environment-dependent constant factor invisible to this profiling run);
  (2) real data typically has a higher outlier/contamination density than
  clean synthetic noise, and `_generalized_esd` is still `O(k*n)` even after
  vectorizing the `ppf` calls, so it scales with however many ambiguous
  columns and removal steps the real data actually triggers. This needs
  profiling against the real dataframe, not further guessing against a
  synthetic one, before deciding whether actual parallelism
  (multiprocessing/threading across columns or entities) is warranted on top
  of these fixes.

### Fixed: five bugs found in a dedicated correctness sweep, all reproduced before fixing

- **PNL002 (`audit_cross_sectional_leakage`) folded null-entity rows into a
  phantom `"nan"` entity.** On pandas < 3 (this package's declared support
  range, `pandas>=1.5,<3`; masked on pandas 3.x, whose new string dtype
  preserves `NaN` through `.astype(str)` instead of stringifying it),
  casting the group column to string turned a missing entity id into the
  literal string `"nan"`, a real, non-null groupby key. Those rows then
  participated fully in the cross-sectional correlation, contradicting
  PNL004's own documented guarantee elsewhere in the same file that
  null-entity rows are "excluded from every panel check... never examined."
  Verified directly in a pandas-2.x environment: a 24-ticker panel plus 60
  null-ticker rows produced a phantom 25th pivoted column that shifted every
  reported PNL002 statistic. Fixed by dropping null-entity rows before the
  string cast, mirroring what `audit_panel_structure` and `apply_fixes`
  already do. `test_pnl002_ignores_null_entity_rows` pins it by asserting
  identical results with and without wildly-out-of-scale null-ticker rows
  appended: confirmed to fail pre-fix on pandas 2.x, pass after.

- **LEK003 (`audit_temporal_leakage`) lost almost all detection power on
  persistent (near-unit-root) targets**, which includes ordinary undifferenced
  price-level data: this library's own stated finance use case. Swept
  detection rate of an obviously-leaky centered rolling window across target
  autocorrelation: 10/10 trials caught at φ=0.7 (what the existing test suite
  exercises), 9/10 at φ=0.9, then collapse: 1/10 at φ=0.95, 0/10 from φ=0.97
  through a literal random walk (φ=1.0). This was a structural limit of the
  `expected(k) = |r0| * |persistence(k)|` formula, not a simple coding
  mistake: as persistence approaches 1, the persistence-explained ceiling and
  a genuine leak's observed correlation both approach 1 and become
  numerically indistinguishable by subtraction: correlation is bounded in
  [-1, 1] and compresses near the endpoints, so a fixed raw-difference
  threshold can't resolve real excess dependence once both sides are pinned
  near the same ceiling.

  **Fixed** by comparing Fisher-z transforms (`arctanh`) instead of raw
  correlations: `excess(k) = arctanh(|observed(k)|) - arctanh(|expected(k)|)`.
  `arctanh` is the standard variance-stabilizing transform for a correlation
  coefficient: it stretches the space back out near ±1 while staying close
  to the identity for small-to-moderate correlations, so `excess_threshold`'s
  default (0.1) needed no change: every pre-existing LEK003 test still passes
  with it unchanged, confirmed by direct comparison (z-space excess values on
  this module's own existing test fixtures came out within 0.01 of the
  original raw-difference values).

  A ratio reformulation (`excess(k) = (observed(k) - expected(k)) /
  (1 - expected(k))`) was tried first and rejected: it also restores recall,
  but its denominator approaches zero as persistence approaches 1, which
  amplifies ordinary Spearman sampling noise on *honest* trailing features
  into false positives: measured 12-30% false-positive rate at φ ≥ 0.95
  across a 60-trial sweep, against 0% for the Fisher-z transform at matching
  recall on the same scenarios. The z-transform was chosen because it fixed
  the collapse without introducing that tradeoff, not merely because it
  worked first.

  Swept again after the fix, same construction as above: 100% detection at
  every φ tested, including φ=1.0 (previously 0%). A weaker, heavily-noised
  leak (as opposed to an obvious centered window) is still only caught
  13-57% of the time at φ ≥ 0.97 rather than the near-total collapse the old
  formula produced: distinguishing a barely-there leak from noise on a
  near-random-walk target is intrinsically harder, not merely a units
  problem, so this is a real improvement rather than a full fix of that
  weak-signal case. `tests/test_temporal.py` adds
  `test_centered_rolling_caught_across_persistence_levels` (parametrized
  φ = 0.7/0.9/0.95/0.99/1.0; the φ=1.0 case is confirmed to fail
  pre-fix: raw excess 0.032, under the 0.1 threshold, for an obvious leak)
  and `test_honest_trailing_feature_not_flagged_near_random_walk` (pins that
  the shipped fix doesn't have the ratio alternative's false-positive
  failure mode). `Issue.evidence` for LEK003 gained
  `expected_from_persistence` (the raw-scale expected correlation, for
  transparency) and `excess_scale: "fisher_z"` (documenting the units
  `excess_over_persistence` and `excess_threshold` are now both in); no
  existing evidence key was removed or renamed.

- **ANO002's `worst_value`/`worst_timestamp`/`max_zscore` evidence could name
  a row that was never actually flagged.** All three were computed from
  `z_scores.abs().argmax()` over the whole column, not restricted to the
  points `combined_mask` (z-score OR IQR) actually flagged. This surfaces
  whenever the two rules disagree (`zscore_outlier_count == 0`,
  `iqr_outlier_count > 0`, the same "ambiguous" branch the ESD masking
  diagnostic already exists to disambiguate) and Tukey's fence, which can be
  asymmetric on skewed data, flags a different point than the raw
  column-wide z-score ranking would. Reproduces on plain Gaussian noise with
  no injected contamination (deterministic repro: seed=243, n=111); a sweep
  of 20,000 random trials found 19 such mismatches. A reader investigating
  "the worst anomaly" in a flagged column would have looked at the wrong
  timestamp. Fixed by restricting the argmax to `combined_mask`;
  `test_worst_evidence_points_at_a_flagged_row` and
  `test_worst_evidence_always_matches_a_flagged_row` pin the invariant.

- **`to_timesfm`'s `min_context` floor was checked before `context_len`
  truncation**, not against the array actually returned, so a caller-supplied
  `context_len` smaller than `min_context` silently returned a shorter array
  than the documented hard minimum ("the adapter raises below it") instead of
  raising. `test_context_len_smaller_than_min_context_still_raises` pins it.

- **Two stale docstring/comment claims**, both found to be materially wrong
  rather than just imprecise: `audit_missing`'s docstring claimed it emits
  PRF002/PRF005/PRF006, but PRF005 (gap clustering) is actually raised by the
  unrelated `audit_frequency`. `audit_missing` never emits it. And
  `leakage/_common.py`'s module docstring claimed `equivalence.py` shares its
  `encode_target` helper, when `equivalence.py` deliberately implements its
  own separate encoding (its own code comment says so explicitly) because its
  AUC computation needs strictly `{0.0, 1.0}` labels, a stricter requirement
  than the Spearman-based detectors that do use the shared helper. Both
  corrected to describe actual behavior.

## [0.5.0] - 2026-08-06

Maintenance release. Every entry below is a bug fix, a docstring/wiki correction, or test coverage for existing behavior: no new public API surface beyond `fix(group_col=...)`, which closes a gap every other panel-aware entry point already had.

### Fixed: several stale numbers and one real cross-module doc error, found in a dedicated consistency sweep

- **README's "tests passing" count went stale repeatedly across this release** as tests were added during review (430 → 448 in an earlier release; 493 → 487 → 493 → 499 → 500 across this one, the 487 dip caused by momentarily computing it in an environment missing the optional `polars`/`joblib` dependencies rather than the full `[dev]` set CI actually installs). Now 500, matching a fresh `pytest --collect-only` with `pip install -e ".[dev]"`.
- **Two wiki version-string examples were still `0.4.0`** (`wiki/API-Reference.md`'s `tsa.__version__` example, `wiki/Installation.md`'s "verifying the install" output) against the actual current `0.5.0`. Same class of drift `to_dict()`'s own [0.4.0] CHANGELOG entry already flagged once before for a prior release's leftover `0.3.0` references.
- **`wiki/Issue-code-reference.md`'s "Severity levels" summary table said 5 CRITICAL / 11 WARNING**, summing to 18 — the code count from before panel support (PNL001-004) existed. The actual, current split (verified by counting every `severity=` in source, cross-checked against the same page's own complete code table just above, which was already correct) is 6 CRITICAL / 13 WARNING / 2 INFO = 21.
- **The same page's PNL002 row claimed "No" for "Repaired by `apply_fixes`?".** False: `leaky_columns()` explicitly includes PNL002-tagged columns (by design, per its own docstring), so `apply_fixes(leakage="drop")` does remove them — verified empirically with a constructed panel carrying a genuine cross-sectional lookahead feature flagged only by PNL002, not by any per-entity LEK00x code, confirming the drop path specifically rather than inferring it from reading `leaky_columns()`'s source alone. Added `test_apply_fixes_drops_a_column_flagged_only_by_pnl002` to `tests/test_panel.py` to lock this in.
- **`infer_frequency`'s "weekly" and "monthly" branches, and its final "irregular" fallback, had never been exercised by any test in the suite** — every existing test used daily or sub-daily data. Verified all three classify correctly by direct call; added `test_infer_frequency_weekly`/`_monthly`/`_irregular` to `tests/test_scaffold.py`.
- **Three wiki pages (`How-it-works.md`, `Internals.md`, `Remediation.md`) described `remediate.py` as keeping its own hand-duplicated copy of the detector formulas** — "these duplicate the detector modules' values," "a real maintenance hazard," "held together only by `tests/test_fix.py`." This was true once, but not since `tsauditor/anomaly/_common.py` was introduced: `remediate.py` now imports the threshold presets and masking functions directly (`zscore_preset`, `stuck_window_preset`, `spike_threshold_preset`, `zscore_iqr_masks`, `clip_bounds`, `stuck_run_mask`, `spike_bounds`), and `test_detector_and_repair_share_the_same_threshold_and_mask_functions` asserts this by object identity, not just matching output. All three pages rewritten to describe the current, structurally-drift-proof architecture, with the historical incident (the ANO001 single-row-gap bridge that once *did* drift) kept as context for why the change mattered.
- **`wiki/Internals.md`'s `Anomaly and remediation helpers` section went further: its code snippets and prose described the *old* stuck-run implementation, and directly asserted the opposite of the current, correct behavior** — "a NaN correctly splits a run rather than joining two." The current `stuck_run_mask` deliberately *bridges* a single interior NaN so a lone missing reading inside an otherwise-flat run still counts as one continuous stuck run, not two shorter ones (verified directly: `[1,1,1,NaN,1,1,1]` bridges to one run of length 7; `[1,1,1,NaN,2,2,2]`, a genuine transition, still correctly splits into two runs of 3). The same wrong claim was repeated on `wiki/Detectors-Anomaly.md`, the primary user-facing page explaining ANO001 — the more consequential of the two, since it's what a user reads to understand the detector's actual behavior. Both corrected and re-verified against a fresh `stuck_run_mask` call before editing.
- **A contradiction within the same paragraph of `wiki/Internals.md`**: one sentence correctly explained that `equivalence.py`'s own target-encoding is a deliberate semantic difference from the shared `encode_target` helper, "not incidental duplication" — then the very next sentence called it "duplicated verbatim... a consolidation opportunity for a future refactor." Rewritten to drop the contradiction and note that `combination.py` uses the same encoding a third time, deliberately, so its single-feature guard agrees with LEK001 about what "binary" means.
- **`wiki/Internals.md`'s "Known rough edges" section still listed the (now-fixed) remediate.py duplication as a live limitation.** Removed; a resolved issue documented in the CHANGELOG isn't a "known rough edge" anymore. The section's other three entries were individually re-verified against current source before leaving them in place (`domain=` really is accepted-and-ignored by all four leakage detectors and `audit_stationarity`, confirmed by grepping every function body for an actual use).

### Fixed: two docstring inaccuracies in scan()

- `run_leakage`'s docstring claimed leakage checks are "silently skipped if target is None." False for LEK004 (as-of leakage): it is target-independent by design and runs whenever `available_at` is supplied, target or not — confirmed with a direct repro (`scan(df, target=None, available_at=...)` still raises LEK004) before correcting the wording to name the exception explicitly.
- `group_col`'s docstring said "Panel-level structure checks (PNL001, PNL003) also run," omitting PNL004 (null-entity rows), which the same `audit_panel_structure` call always raises alongside the other two. Docstring now lists all three and notes they are not gated by any `run_*` toggle.

### Fixed: profiler package docstring claimed a KPSS test that does not exist

- `tsauditor/profiler/__init__.py`'s module docstring described `stationarity` as running "ADF/KPSS stationarity tests." `profiler/stationarity.py` only ever implemented ADF (`statsmodels.tsa.stattools.adfuller`); grepping the whole package for `kpss`/`KPSS` turned up this one docstring line and nothing else — no implementation, no import, no code path. Corrected to describe only what the module actually does.

### Fixed: detectors called directly (not via scan()) silently mis-scored an out-of-order DatetimeIndex

- **Every detector that depends on row *position* (rolling windows, `.shift()`, consecutive-run detection, positional lag alignment) assumed its input was already chronologically sorted.** That assumption only held on the `scan()` path, where `validate_dataframe` sorts once upstream. Every `audit_*` function is also public API, documented and used directly in this codebase's own tests, and a caller who imports a detector and calls it on a DataFrame with a *valid* but out-of-order `DatetimeIndex` got no error: the detector ran on the scrambled row order and produced a wrong-but-silent result instead. Confirmed concretely, not just in theory: shuffling the exact same rows made `audit_correlation_leakage` and `audit_temporal_leakage` miss a perfect, constructed lag+1 leak entirely (`[]`, no exception), and made `audit_contextual_anomalies` miss an 8-point stuck run outright.

  Added `ensure_sorted_datetime_index(df, context)` in `tsauditor/utils/validation.py`: validates the index is a `DatetimeIndex` (matching every existing detector's error message) and returns it sorted ascending. Every detector that depends on row order now calls it at its own entry point: `audit_contextual_anomalies`, `audit_correlation_leakage`, `audit_temporal_leakage`, `audit_missing`, `audit_non_finite`, `audit_stationarity`. This changes nothing on the `scan()` path (input is already sorted there) but fixes silent false negatives for direct callers.

  Uses `kind="mergesort"` (pandas/numpy's documented stable sort), not the default `quicksort`: two rows sharing a duplicate timestamp must keep their original relative order, not an unspecified one, since a duplicate timestamp is already its own PRF004 CRITICAL finding and downstream `keep="first"` dedup logic depends on "first" meaning what the caller supplied. `validate_dataframe`'s own `df.sort_index()` (the `scan()` path) is fixed the same way, for the same reason.

### Fixed: apply_fixes silently no-op'd on a caller-supplied out-of-order DatetimeIndex

- **`apply_fixes(report, df)` takes `df` directly from the caller, completely independent of the sorted copy `scan()` validated internally.** If that `df` had a valid but chronologically out-of-order index, every repair mask (`stuck_run_mask`'s consecutive-run walk, `spike_bounds`' rolling window) was computed directly on the unsorted row order and could find nothing at all, even for a column `report` says was flagged, silently returning a "repaired" DataFrame that still contained the original, unfixed anomaly. Worse than a missed detection, since `report.last_fixes` and the caller both believe the data was cleaned.

  `apply_fixes` now sorts its own working copy by position (`np.argsort(..., kind="mergesort")`, not `.sort_index()` + relabel, so a duplicate-timestamp tie order matches `validate_dataframe`'s own stable-sort behavior) before computing any mask, and restores the caller's original row order before returning, both so the "byte-for-byte unchanged for untouched columns" guarantee holds in the caller's own row order, and because `_apply_fixes_by_group` writes each entity's result back by raw position and requires the row order it receives to match what it passed in.

### Fixed: affected_cells()/health_score() computed one global mask across an entire interleaved panel

- **`affected_cells` (backing `health_score()` and `to_dict()`/`to_json()`'s health block) recomputed its outlier/spike/stuck masks from the raw `df` as one series, even for a panel report.** Every quality detector actually ran per-entity during the original `scan(group_col=...)` (see `scanner.py`'s per-partition loop), so re-deriving a mask from the whole interleaved panel at once computes one mean/std/rolling-window mixing every entity's values together: a real outlier in a small-scale entity can be diluted below a large-scale entity's ordinary range and vanish entirely from the health score, or the reverse.

  `affected_cells` is now panel-aware: when `report.metadata["group_col"]` is set, it recomputes each mask per entity, against only that entity's own Issues, mirroring what the original scan and `apply_fixes`'s own per-entity repair already do. Rows with a null entity id (PNL004) are skipped, since they were never scanned per-entity in the first place.

### Fixed: to_pdf() re-scanned a panel ungrouped, and dumped raw issues instead of prevalence

- `export_pdf`'s internal re-scan of `fixed_df` (used to compute the "after" health score) did not pass `group_col` through, the same gap `to_json()`'s equivalent re-scan had before its own earlier fix: an independent copy of the same bug.
- The PDF's issues table rendered `report.all_issues` directly for a panel scan, with no entity label on the row at all (`Issue.group` was never included). A systemic finding present in every entity of a large panel produced hundreds or thousands of visually-identical, unlabeled rows spread across as many continuation pages as it took to fit them.

  `export_pdf` now threads `group_col` through its re-scan, and renders a prevalence table (`_prevalence_table`, mirroring `report.prevalence()`, one row per `(code, column)` with the fraction of entities it hit) instead of the raw issue dump whenever `report.is_panel`, matching what `report.summary()`'s CLI output already did.

### Added: fix() accepts group_col for panel data

- Every other panel-aware entry point (`scan()`, `GuardReport.apply_fixes()`, `health_score()`, `to_json()`, `to_pdf()`) accepted or threaded `group_col`, but `fix()` itself had no parameter for it: `tsa.fix(panel_df, group_col=...)` raised `TypeError: unexpected keyword argument`, forcing panel callers to always fall back to the separate `scan()` + `apply_fixes()` form `fix()` exists to avoid. `apply_fixes` itself needed no change: it already reads `group_col` back off `report.metadata`, which `scan()` populates.

### Fixed: apply_fixes()/fix() never resolved time_col, so a time_col caller's row order was never sort-checked at all

- **`apply_fixes(report, df)` only defended against an out-of-order *DatetimeIndex* — it had no idea `time_col` existed.** A caller who used `scan(df, time_col="date")` and then `fix(df, time_col="date")` or `report.apply_fixes(df)` handed `apply_fixes` the *original* `df`, which still had `time_col` as a plain column and a meaningless `RangeIndex`, not the `DatetimeIndex` `scan()` resolved internally. Since the DatetimeIndex sort-safety added above only triggers when `df.index` actually *is* a `DatetimeIndex`, it never engaged at all for `time_col` callers: the identical "found the issue, repaired zero cells" failure as an out-of-order `DatetimeIndex`, just reached through `time_col` instead. Confirmed concretely: `scan(df, time_col="date")` correctly found an 8-point stuck run on shuffled rows; `fix()`'s subsequent repair silently changed zero cells.

  Root cause was two-layered: `report.metadata` never recorded which `time_col` was used in the first place, so there was no way for `apply_fixes` to know. `scanner.py` now records `metadata["time_col"]`, and `apply_fixes` resolves it into a working `DatetimeIndex` (mirroring `validate_dataframe`'s own `set_index`) before repairing, then restores the caller's original column layout and row order before returning. Verified this composes correctly with `group_col` too (`time_col` + `group_col` together): the per-entity recursive call in `_apply_fixes_by_group` inherits the already-resolved index and correctly skips re-resolving it a second time.

### Fixed: fix()/apply_fixes() crashed on polars input (never actually reachable)

- **`polars.DataFrame` has neither `.copy()` nor `.index`**, so `apply_fixes`'s very first operation on the non-panel path (`out = df.copy()`), or, after the `time_col`/row-order fixes above, the `isinstance(df.index, pd.DatetimeIndex)` check, raised a raw `AttributeError` on any polars input. This predates every other fix in this file: `scan()` has supported polars since it was added, but `fix()`/`apply_fixes()` never did, and nothing caught it because `tests/test_polars.py` only ever exercised `scan()`, never the repair path. Found while resolving an open uncertainty from this review's own critical self-assessment (the polars/joblib paths were flagged as untested in the review sandbox); installing the actual dependencies and testing `fix()` directly reproduced the crash immediately.

  `apply_fixes` now converts polars input to pandas at its own entry point, the same way `validate_dataframe` does at the `scan()` boundary (`_is_polars`/`_polars_to_pandas`, reused rather than duplicated). Returns a pandas `DataFrame`, consistent with `scan()`'s own existing convention that polars is an input-only conversion: internals, and now repair output, stay pandas. Verified against polars input combined with `time_col`, an out-of-order (shuffled) row order, and `group_col` panel data, together and separately.

### Added: to_timesfm() raises clearly on a non-numeric target_col

- A non-numeric `target_col` (categorical, string) previously failed much later and much less clearly: `remediate.py`'s own numeric-dtype guards silently pass a non-numeric column through unrepaired, and `clean[target_col].to_numpy(dtype=np.float32)` then raised a raw `ValueError: could not convert string to float: '...'` with no mention of `target_col` or what to fix. `to_timesfm` now raises `TypeError` naming the column and its dtype, consistent with the adapter's existing finite-value guard (raise clearly rather than let something unusable reach the model silently).

### Added: evidence-key drift guard for remediation suggestion templates

- `report.remediation.suggest()` renders each `_REMEDIATIONS` template through a `_SafeDict` that leaves an unresolved placeholder untouched rather than raising, deliberately, so a missing key never crashes `report.summary()`. That safety has a cost: if a detector's evidence dict key is ever renamed without updating the matching template, nothing fails: the suggestion text just silently starts showing a literal `{old_key_name}` to the user. Added `test_every_template_placeholder_is_a_real_evidence_key` in `tests/test_remediation.py`, which checks every template's placeholders against a schema of each code's actual evidence keys, transcribed from the detector source. No production code change; a regression test for a class of drift that had no coverage.

### Fixed: leaky_columns() silently excluded PNL002 findings

- `leaky_columns()` filtered strictly on `module="leakage"`. PNL002 (cross-
  sectional lookahead leakage, in `panel.py`) is tagged `module="panel"`
  instead, because it is emitted alongside the panel-*structure* checks
  (PNL001/PNL003/PNL004), not from the leakage module. The effect: a feature
  flagged only by PNL002, and by nothing else, never appeared in the
  shortlist `leaky_columns()` exists to produce.

  Fixed by carving PNL002 in by code (`code="PNL002"`) rather than widening
  the module filter to `"panel"` wholesale, since PNL001/PNL003/PNL004 are
  dataset-level findings with no `column`, not features to review: a
  module-level carve-out would have been correct in practice (those issues
  have no column to leak into the list) but code-level is the honest
  statement of intent.

### Fixed: groups() dropped entities with zero issues, despite its docstring

- `groups()`'s docstring promised "every entity scanned", but its body
  collected `{i.group for i in self.all_issues if i.group is not None}`,
  so an entity that scanned completely clean produced no `Issue` and was
  silently absent from the list, indistinguishable from an entity that was
  never scanned at all.

  Fixed by threading the full partition list through `scan(group_col=...)`
  itself: `scanner.py` now records `metadata["groups"]` (sorted entity keys)
  from the same `df.groupby(group_col)` call that drives the per-entity
  loop, before any check runs. `groups()` now reads from there, falling back
  to the old issue-derived behavior only for a `GuardReport` built by hand
  without going through `scan()` (e.g. in tests), where no other source for
  the full list exists.

### Added: PNL002 unstack-branch test coverage

- `audit_cross_sectional_leakage` branches on whether `df.index.name` is
  truthy: named -> `pivot_table`, unnamed -> `groupby().unstack()`. Every
  existing PNL002 test built its panel via `.set_index("date")`, which sets
  `index.name`, so only the `pivot_table` branch was ever exercised; the one
  test with an unnamed index (`test_pnl002_validates_input`) raises on the
  `DatetimeIndex` check before reaching the branch at all. Verified by hand
  that both branches agree (an unnamed-index panel flags the same leak as
  its named-index equivalent) and added that as a direct regression test.

### Removed: dead bounds guards in audit_frequency (PRF001/PRF005)

- `frequency.py` mapped a gap/cluster's positional index to the DataFrame
  row after it (`i + 1`) behind a guard checking `i + 1 < len(df_sorted)`.
  The guard could never be false: `gap_days` (and the `is_large_gap` array
  derived from it) always has length `len(df_sorted) - 1`, so the largest
  possible `i + 1` is `len(df_sorted) - 1`, always in bounds. No behaviour
  change; removed the two list comprehensions and the guard they never
  needed, replacing each with a direct vectorized `+ 1` on the index array.
  Verified with the existing test suite (unchanged) plus a direct check of
  a gap ending at the DataFrame's final row, the boundary case the dead
  guard looked like it was protecting.

### Fixed: LEK003's expected(k) bound could be computed from mismatched populations

- **`r0`, `persistence(k)`, and `observed(k)` were each computed on their own
  independent pairwise-complete sample, which could silently describe
  different populations whenever a feature has its own missingness** (e.g.
  a column only recorded starting partway through the series, added later
  as a new data source, or from a sensor installed after the target series
  began). `persistence(k)` in particular is computed from the target
  against itself, so nothing gated it to the rows the feature actually
  occupies; it could reflect a completely different period of the series
  than the one `r0` and `observed(k)` were implicitly restricted to via
  their own pairwise-complete dropna.

  Confirmed concretely, not just in theory: on a synthetic target with a
  genuine regime change (highly persistent for the first half, close to
  white noise for the second) and a feature recorded only in the
  low-persistence half, persistence measured on the full series came out
  0.75; measured on just the rows the feature occupies, 0.22. That gap is
  far larger than the default `excess_threshold` of 0.1 and is large enough
  to flip a verdict in either direction: a real forward-looking leak
  recorded only in a low-persistence period can have its excess masked by
  an inflated whole-series persistence estimate (false negative, reproduced
  concretely: a genuine lag -1 leak scored an excess of ~0.06, under
  threshold, using the old unaligned computation); the reverse (false
  positive on an honest feature) is possible with a low-persistence period
  during a feature's own missing stretch and a high-persistence period
  where it's present.

  Added `_aligned_correlations` in `tsauditor/leakage/temporal.py`, which
  computes `r0`, `persistence(k)`, and `observed(k)` on one common mask per
  feature and lag: rows where the feature, the target, and the
  lag-`k`-shifted target are all simultaneously non-null. The previous
  whole-series persistence computation is kept as a cheap pre-filter only
  (never used in the `expected(k)` math itself): since the aligned sample
  is always a subset of it, if the loosest possible sample is already
  smaller than `min_obs`, the aligned one will be too, so this never
  discards a lag the aligned computation could otherwise use. Fully
  populated features are unaffected (their aligned mask equals the whole
  series), so this only changes behaviour, and only costs extra
  computation, for features that actually have missing values.

  **Behaviour change:** `audit_temporal_leakage`/LEK003 can now flag or
  clear a feature differently than before on data with staggered
  missingness across columns. This is expected and intended; the previous
  behaviour was not measuring what it claimed to measure in that case.

  Verified with a mutation test: reverting `_aligned_correlations`'s call
  site back to the three independent `_spearman` calls makes the new
  regression test (`test_staggered_feature_leak_caught_despite_regime_change`
  in `tests/test_temporal.py`, built from the exact scenario above) fail
  with the previous, unaligned result (excess ~0.06, unflagged); restoring
  the fix makes it pass again (excess ~0.14, flagged).

### Fixed: apply_fixes could let one repair step change what a later step detected

- **`apply_fixes`'s outlier, spike, and stuck-run repair steps re-detected
  their targets from the column *as already modified by earlier steps*
  (`out[col]`), not from the original input.** A column can carry more than
  one finding at once (an ANO002 outlier and an ANO003 spike; or a value
  stuck at an extreme constant, which is both an ANO002 outlier and an
  ANO001 stuck run), and the repair steps run in a fixed order. Detecting
  the later step's targets on data an earlier step had already clipped or
  NaN'd could silently change what that step found: an earlier NaN could
  make a later step "rediscover" nothing for cells the original audit had
  in fact flagged (so `last_fixes` credited only the first action even
  though the report raised both), and an earlier clip could shift the local
  rolling statistics ANO003's spike detection uses for *other*, unrelated
  nearby cells whose value never changed.

  Every detection step in `tsauditor/remediate.py`'s `apply_fixes` now reads
  from `df[col]` (the pristine input) instead of `out[col]` (the
  in-progress repair), so each step always finds exactly what its own
  `Issue` reported, regardless of what an earlier step already touched.
  `cells_changed` in `last_fixes` is now also computed as *only the cells
  this step newly changes* (cells another step already NaN'd are no longer
  double-counted), and a step that ends up changing nothing no longer adds
  a zero-`cells_changed` log entry. Documented in `GuardReport.apply_fixes`'s
  `Notes` section.

### Fixed: LEK004 raised a confusing raw TypeError on a timezone mismatch

- **Comparing a tz-aware `df.index` against a tz-naive `available_at` Series
  (or vice versa) failed deep inside pandas** (`avail > idx`) with a raw
  `TypeError` that named neither column nor the actual mismatch. A common,
  mundane mistake -- the index was `tz_localize`'d, the release-date metadata
  came from a source that wasn't -- surfaced as an opaque internal error.

  `tsauditor/leakage/asof.py`'s `_availability` now checks `index.tz` against
  `avail.dt.tz` explicitly and raises a `ValueError` naming which side is
  tz-aware and which is tz-naive, before the comparison that used to fail
  blindly. Documented in `audit_asof_leakage`'s `Raises` section.

### Fixed: LEK005 could double-report a column LEK001 already flagged

- **`audit_combination_leakage`'s single-feature guard (the check that skips
  a candidate group when one of its columns already explains the target
  alone) only checked its own adjusted-R² metric against the threshold.**
  `audit_equivalence` (LEK001) scores the same question with a different
  metric (AUC for binary targets, Spearman for continuous). The two
  disagreed hardest on exactly the case LEK001 exists for: a strong
  monotonic but non-linear relationship (AUC/Spearman near 1.0, adjusted R²
  well below LEK005's threshold). A column already flagged by LEK001 could
  slip past LEK005's R²-only guard and get reported a second time inside a
  LEK005 group, with a description claiming none of the group's columns
  explains the target alone -- true under R², false under LEK001's own
  metric, on the same data, in the same scan.

  Extracted `_score_feature` out of `audit_equivalence` in
  `tsauditor/leakage/equivalence.py` (pure refactor there: `audit_equivalence`
  now calls it instead of scoring inline, same behaviour) and imported it into
  `tsauditor/leakage/combination.py` as `_equivalence_score`. The
  single-feature guard now takes `max(adjusted_r2, equivalence_score)` per
  column, so a column excluded from LEK005's search either belongs to LEK001
  (by either metric) or genuinely doesn't explain the target alone by any
  metric this library computes. `best_single_adjusted_r2` in LEK005's
  evidence is unchanged in meaning (still the pure R² value; falls back to
  0.0 if every column in a reported group only qualified via the
  equivalence-score half of the guard, avoiding a crash on `max()` over an
  all-`None` set). Documented in `audit_combination_leakage`'s `Notes`
  section.

### Fixed: scan()'s constraints dispatch crashed on a column named "bounds" or "relations"

- **`constraints={"spread": {...}, "relations": {"min": 0}}` -- a flat bounds
  dict for two columns, one of them named "relations" -- crashed** with
  `ValueError: too many values to unpack (expected 2)`. `scan()` used to
  decide between the nested `{"bounds": ..., "relations": ...}` form and the
  flat `{col: spec}` shorthand by key presence alone
  (`.get("bounds")`/`.get("relations")` both `None` => flat); the moment a
  real column happened to be named "bounds" or "relations", that key's
  presence made the flat-dict fallback never fire, and `audit_validity`
  received a per-column spec dict where it expected a relations pair (or a
  nested bounds mapping), and broke trying to unpack or iterate it.

  `tsauditor/scanner.py`'s `_run_checks` now distinguishes the two forms
  *structurally* instead: a nested `"bounds"` value is always a dict mapping
  every column to its own spec dict (dict-of-dicts), and a nested
  `"relations"` value is always a list/tuple of pairs -- shapes a flat
  per-column spec can never take. Documented in `scan()`'s `constraints`
  parameter docstring, with the "relations"-named-column example as the
  worked case. Added
  `test_scan_flat_bounds_column_named_relations_no_longer_crashes`,
  `test_scan_flat_bounds_column_named_bounds_no_longer_crashes`, and
  `test_scan_nested_bounds_and_relations_together_still_works` in
  `tests/test_validity.py`.

### Documented: the finance-vs-everything-else domain asymmetry in the profiler checks

- **`audit_frequency` and `audit_missing` only special-case `domain="finance"`;
  `"sensor"` silently falls into the same branch as `domain=None`**, unlike
  every domain-aware anomaly check, which gives `finance`/`sensor`/`None`
  three distinct values. Investigated rather than assumed to be a bug or
  reflexively "fixed" with a third branch:

  `audit_frequency`'s `maximum_gap_threshold` is fine as a two-way branch.
  Its non-finance default is already `3.0 * median_gap` -- relative to the
  series' own sampling cadence, so it self-calibrates for sensor data
  regardless of whether that data is sampled every second or every hour.
  Finance is the one domain that needs an absolute constant instead (5.0
  days), because trading calendars have a known, bounded gap structure
  (weekends/holidays) that a relative multiplier would handle less
  predictably. A sensor branch here would need its own relative multiplier,
  and there's no measured basis for a different number yet.

  `audit_missing`'s `cluster_threshold` is a genuine, open gap: it's a flat
  row count (3 or 5), not relative to anything, so "3 consecutive missing
  rows" means something very different for a 1-second sensor feed than for
  daily data. Deliberately **not** patched with a guessed "sensor" constant
  here -- doing so would be exactly the unvalidated-heuristic-dressed-as-
  domain-expertise pattern already flagged for `masking_suspected`'s `0.5`
  multiplier. If this gets fixed, the defensible direction is making the
  default scale with the series' own inferred sampling frequency (the way
  `audit_frequency` already does), not adding a domain-keyed guess.

  Both functions' docstrings now document this reasoning directly so a
  future reader (including a future reviewer of this codebase) doesn't
  mistake the two-way branch for an oversight, or "fix" `cluster_threshold`
  by inventing a number. No behaviour changed.

### Removed: duplicated run-length-encoding logic in the profiler checks

- **`audit_frequency` (PRF001/PRF005) and `audit_missing` (PRF002) each
  computed run starts/ends/lengths from a 0/1 array using the same
  three-step numpy pattern, independently** -- the same category of
  duplication as the anomaly/remediate thresholds below, just not yet
  caught drifting. Extracted into `tsauditor/profiler/_common.py`
  (mirroring the `anomaly/_common.py` and `leakage/_common.py` pattern
  already used elsewhere in this codebase): `consecutive_run_lengths(flags)`
  is now the single implementation both callers import.

  Pure refactor, no behaviour change: verified against the previous inline
  logic on seven hand-checked boundary cases (empty array, single 0, single
  1, no runs, all-1s, and two mixed patterns), and the full suite (460
  pre-existing tests) passes unchanged.

  Verified structurally: added
  `test_frequency_and_missing_share_the_same_rle_function` in
  `tests/test_profiler.py`, asserting both modules resolve to the exact
  same function object.

### Fixed: ANO001 could false-negative when a single dropout split a stuck run

- **A lone missing reading inside an otherwise-flat run could prevent the
  run from ever being flagged**, even when the true, uninterrupted run
  comfortably exceeded `stuck_window`. Root cause: stuck-run grouping used
  `series.diff().ne(0).cumsum()`, and `diff()` on a `NaN` reads as "changed"
  twice for a single gap, once at the missing row, once at the row right
  after it (`value - NaN` is also `NaN`). A run like five repeats, one
  dropout, five more repeats split into groups of 5/1/5, and with
  `stuck_window=5` neither half crossed the threshold, so nothing fired at
  all under the *default* `handle_missing="strict"`.

  Fixed by grouping on a local, ANO001-only bridged view
  (`series.interpolate(method="linear", limit=1)`) that only produces a
  zero diff when both neighbours already agree, so a genuine transition
  (a gap between two *different* values) still breaks the group correctly.
  This is independent of `handle_missing`, which continues to govern only
  what ANO003 sees. Verified with a mutation test: reverting to the plain
  `series.diff()` grouping makes the new regression test fail exactly as
  expected.

  **Behaviour change:** a run split by a single-row gap that previously
  went unflagged (each half at or under `stuck_window`) will now be
  flagged, with `max_stuck_duration` reflecting the full bridged span
  (e.g. 11 for a 5/gap/5 run), not just the longer observed half.

### Added: anomaly detector tuning is now reachable through `scan()`

- **`scan()` previously forwarded only `domain=` to the anomaly
  detectors.** `zscore_threshold`, `stuck_window`, `spike_threshold`,
  `spike_window`, and `handle_missing` existed on
  `audit_point_anomalies`/`audit_contextual_anomalies` but were only
  reachable by importing and calling those functions directly, bypassing
  the documented entry point. A sensor user with known single-row dropouts
  had no way to opt into gap handling short of forking the call.
  `scan()` now accepts and forwards all five; every default is unchanged
  (`None`/`"strict"`, matching the underlying functions exactly), so this
  is purely additive.

### Documented: `masking_suspected`'s `0.5` threshold is a heuristic

- Added a `Notes` section to `audit_point_anomalies`'s docstring stating
  plainly that the `n_esd > n_iqr * 0.5` multiplier isn't derived from the
  ESD/Rosner literature or a validated benchmark, matching CONTRIBUTING.md's
  existing honesty policy on float thresholds. It only affects the
  diagnostic `masking_suspected` evidence field, never which points get
  flagged as anomalies.

### Fixed: `remediate.py`'s stuck-value mask had silently drifted out of sync with ANO001

- **`report.apply_fixes(..., stuck="nan")` could silently fail to repair a
  run that `scan()` itself had just flagged.** `remediate.py` keeps its own
  copy of the ANO001 stuck-run grouping logic (`_stuck_mask`), separate from
  `anomaly/contextual.py`'s `audit_contextual_anomalies`, connected only by a
  comment ("identical to ANO001"). When the single-row-gap bridging fix
  above was made to the detector, `_stuck_mask` was not updated to match, so
  a run the detector correctly reported as one 11-long stuck span
  (`max_stuck_duration: 11`) was still seen by the repair step as two
  separate 5-long halves, neither of which crossed `stuck_window`, and
  `apply_fixes` left the original values untouched with no error or warning.

  Fixed by mirroring the same bridge (`series.interpolate(method="linear",
  limit=1)`) into `remediate._stuck_mask`. Verified with a mutation test:
  reverting the bridge makes the new regression test fail exactly as
  expected, then restoring it makes the suite pass again.

  Added `test_stuck_mask_matches_detector_evidence` in `tests/test_fix.py`,
  the missing counterpart to the existing
  `test_outlier_nan_count_matches_detector_evidence` (which already covers
  ANO002/outliers, but had no ANO001/stuck-value equivalent): it asserts
  `_stuck_mask` flags exactly the rows the detector's own evidence claims
  are stuck, so this category of drift fails a test instead of failing
  silently in production.

  **Scope note (superseded):** this entry originally fixed only the
  `_stuck_window`/`_stuck_mask` sync gap and left the rest of the
  duplication (`_zscore_threshold`, `_spike_threshold`, `_outlier_mask`,
  `_spike_info`, `_SPIKE_WINDOW`) as hand-maintained copies for a follow-up.
  That follow-up is the "Removed" entry below: all of it has since been
  extracted into a single shared module, so none of these are separate
  copies anymore.

### Added: cross-domain sync tests for `_zscore_threshold` and `_spike_threshold`

- **Same duplication risk as the `_stuck_window` fix above, not yet
  triggered.** `remediate.py`'s `_zscore_threshold` (finance 5.0, sensor 3.5,
  default 4.0) and `_spike_threshold` (finance 4.0, sensor 3.0, default 3.5)
  are separate hardcoded copies of the domain presets in
  `anomaly/point.py`/`anomaly/contextual.py`, and the only existing
  cross-checks (`test_outlier_nan_count_matches_detector_evidence`,
  `test_spike_nan_count_matches_detector_evidence`) exercised domain=None
  only, so a preset changed for `"finance"` or `"sensor"` in one place but
  not the other would have passed the suite silently.

  Added `test_zscore_threshold_matches_detector_across_domains` and
  `test_spike_threshold_matches_detector_across_domains`, parametrized over
  `domain in (None, "finance", "sensor")`, each built from data where the
  injected points straddle the domain-specific threshold boundaries so the
  three domains produce three different flagged counts, and asserting
  `apply_fixes`'s repaired-cell count matches the detector's own evidence
  count in every case. Mutation-verified for both: corrupting either
  function's `"finance"` or `"sensor"` branch alone makes the corresponding
  parametrized case fail, while the other two still pass, then restoring
  makes the suite pass again. No behavior changed here (the values already
  agreed); this closed the untested gap before it could drift.

  **Superseded by the entry below:** these are still useful integration
  tests (they confirm `apply_fixes` actually resolves `report.metadata`'s
  domain correctly end to end), but they no longer guard against drift
  specifically -- after the shared-module extraction, `_zscore_threshold`
  and `_spike_threshold` don't exist as separate functions to drift.

### Removed: `remediate.py`'s hand-maintained duplicate thresholds and masks

- **The root cause behind all three entries above, addressed directly
  instead of tested around.** `remediate.py` kept its own copy of every
  anomaly threshold preset and masking formula, connected to the real
  detectors in `anomaly/point.py`/`anomaly/contextual.py` only by comments
  ("Match anomaly/point.py ANO002"). That had already drifted once (the
  `_stuck_window`/`_stuck_mask` incident above) and had no cross-check at
  all for `_zscore_threshold`/`_spike_threshold` until the entry above added
  one. Testing for drift closes the gap after the fact; it does not remove
  the thing that can drift.

  Added `tsauditor/anomaly/_common.py` (mirroring the existing
  `tsauditor/leakage/_common.py` pattern) holding the single implementation
  of every domain preset (`zscore_preset`, `stuck_window_preset`,
  `spike_threshold_preset`), the `SPIKE_WINDOW` constant, and every masking
  formula (`zscore_iqr_masks`, `clip_bounds`, `stuck_run_mask`,
  `spike_stats`, `spike_bounds`). `anomaly/point.py`, `anomaly/contextual.py`,
  and `remediate.py` now all import from it; none of them define their own
  copy anymore.

  This is a pure refactor with no behaviour change: every preset and formula
  is byte-identical to what each file had before, just relocated to one
  place. The full test suite (459 pre-existing tests) passes unchanged.

  Verified structurally, not just behaviourally: added
  `test_detector_and_repair_share_the_same_threshold_and_mask_functions` in
  `tests/test_fix.py`, which asserts the detector and repair modules resolve
  every preset and mask through the *exact same function object* in
  `_common.py`, rather than merely producing the same values today.
  Mutation-tested by monkeypatching one module's reference to a decoy
  function and confirming the identity assertion catches it. This makes the
  class of bug this whole review thread started from (two places encoding
  one fact, kept in sync by memory) structurally impossible to reintroduce
  by accident: there is only one implementation left to copy-paste away
  from, and doing so now fails a test immediately instead of waiting for a
  future threshold change to expose it silently in production.

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
