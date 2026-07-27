# PNL002 — Cross-sectional lookahead detection

> **Status: IMPLEMENTED.** Shipped in `tsauditor/panel.py` as
> `audit_cross_sectional_leakage`, wired into `scan()` for panel scans with a
> target. See [Panel Data](../../wiki/Panel-Data.md#pnl002--cross-sectional-lookahead)
> for user documentation and `tests/test_panel.py` for the test suite.
>
> This document is retained as the design record — the measurements below are
> what justified the check and set its thresholds. The evidence script
> (`pnl002_evidence.py`) still reproduces the table.
>
> **Implementation notes vs. this proposal:**
> - Thresholds landed at `excess_threshold=0.15` and `min_correlation=0.15`,
>   stricter than LEK003's 0.1, validated against simulated factors up to
>   IC 0.15 (not flagged) and 0.30+ (flagged).
> - Gates: `min_entities=20` per timestamp, `min_timestamps=30`.
> - The check is vectorised (rank rows once, correlate row-wise with array
>   arithmetic). A per-timestamp loop took 4.0s for 40 entities; the shipped
>   version takes 0.05s, and 0.72s for 200 entities x 1500 timestamps.
> - It runs automatically in panel mode when a target is given, rather than
>   behind a separate opt-in flag as originally scoped — the cost turned out to
>   be low enough that an extra flag was not worth the API surface.

---

## Summary

Cross-sectional features — rank, z-score, decile, sector-neutralisation,
market-relative return — are computed **across entities at a single timestamp**.
They are ubiquitous in quantitative equity and increasingly common in multi-sensor
and multi-store forecasting.

When such a feature is computed from the cross-section at *t+1* and joined back to
*t*, the result is a leak. The existing per-entity checks (LEK002/LEK003) detect it
**only when idiosyncratic variation is large relative to the common factor**, and
their detection rate collapses as the common factor grows — while a cross-sectional
test detects it perfectly regardless.

The measurements below establish the exact condition under which the current checks
fail, and it is a condition that describes real equity markets.

---

## The problem, concretely

The legitimate feature — each entity's return rank against all others on the same
day:

```python
panel["xs_rank"] = panel.groupby("date")["ret"].rank(pct=True)         # correct
```

The bug, off by one day, in the direction nobody investigates because it improves
results:

```python
panel["xs_rank"] = panel.groupby("ticker")["xs_rank"].shift(-1)        # LEAK
```

Every row now carries **tomorrow's** cross-sectional rank.

---

## Measured behaviour of the existing checks

Simulation: 40 entities × 500 business days. Returns are a common market factor
(with time-varying volatility) plus an idiosyncratic component. The target is each
entity's own absolute return. `mkt/idio` is the ratio of common to idiosyncratic
volatility.

| mkt/idio | within-entity rho | LEK002 detection | LEK003 detection | cross-sectional rho at lag +1 |
| -------- | ----------------- | ---------------- | ---------------- | ----------------------------- |
| 0 | 0.976 | 100.0% | 100.0% | **1.000** |
| 1 | 0.537 | 100.0% | 100.0% | **1.000** |
| 5 | 0.158 | 95.0% | 95.0% | **1.000** |
| 25 | 0.033 | 32.5% | 22.5% | **1.000** |
| 100 | 0.008 | 22.5% | 12.5% | **1.000** |

Two things this shows.

**1. The existing checks are better than expected, and the honest framing is
"degrades", not "blind".** With no common factor the leaked feature retains a
within-entity rank correlation of 0.98 with the future target, and LEK002/LEK003
catch it in 100% of entities. Any proposal claiming they are useless here is wrong.

**2. Detection collapses as the common factor dominates.** A cross-sectional rank
measures *relative* performance. Once a market factor dominates absolute returns,
relative position decouples from an entity's own return: within-entity correlation
falls to 0.03 at a 25:1 ratio — below the `min_correlation = 0.1` floor — and
detection drops to roughly one entity in four. **The cross-sectional correlation
stays at exactly 1.0 throughout.** It is invariant to the ratio that destroys the
univariate signal.

Real equity markets sit in the region where this matters. Market factors routinely
explain the large majority of individual stock return variance, particularly at
daily frequency and in stressed periods.

### The failure mode is worse than a missed detection

At a 25:1 ratio the prevalence table reports:

```
LEK002   xs_rank   9/40   22.5%
```

A user reading that concludes the problem is **isolated** — a handful of odd
entities worth a look. The truth is that the leak is present in 100% of entities
and the check simply failed to score most of them.

This is a direct consequence of the panel reporting built in the previous phase:
prevalence is a very useful signal precisely because a low percentage means
"isolated" — so a check that degrades non-uniformly makes that inference actively
misleading. **A partially-sensitive detector reporting into a prevalence table is
more dangerous than no detector**, and that, rather than a claim of total blindness,
is the strongest argument for PNL002.

### Why LEK001 and LEK004 do not apply

- **LEK001** requires near-determinism (AUC ≥ 0.95 or |ρ| ≥ 0.95) within one
  entity. The leaked rank does not reach that except in the degenerate zero-common-
  factor case.
- **LEK004** requires declared `available_at` metadata. Nobody declares a
  publication lag for a feature they computed themselves from data they already
  had — there is no external release schedule to declare.

---

## Proposed detection method

The proposal deliberately reuses the **excess-over-baseline** structure from
LEK003, which is the most defensible piece of statistics in the codebase, rather
than inventing a new criterion.

### Step 1 — identify candidate cross-sectional features

A cross-sectional feature has a signature: at each timestamp, its values across
entities are strongly constrained. For a percentile rank they are approximately
uniform on [0, 1] and sum to a near-constant; for a cross-sectional z-score they
have mean ≈ 0 and standard deviation ≈ 1 at every timestamp.

Test: group by timestamp, compute a per-timestamp statistic (mean, std, min, max),
and check whether its variance **across timestamps** is far lower than would be
expected for an unconstrained column.

```
cs_score(col) = 1 - var_t( mean_e( col ) ) / var( col )
```

A column near 1.0 is cross-sectionally normalised. This step is cheap and only
narrows the search; it is not itself evidence of leakage.

### Step 2 — test cross-sectional alignment at lag +1

For a candidate feature `f` and target `y`, at each timestamp *t* compute the
**cross-sectional** Spearman correlation across entities:

```
rho_cs(t, k) = corr_e( f[e, t] , y[e, t+k] )
```

That is: on day *t*, does the feature rank entities in the order their *future*
targets will rank them?

Average over timestamps to get `rho_cs(k)`, then apply the LEK003 logic:

```
observed(k) = | mean_t rho_cs(t, k) |
expected(k) = | rho_cs(0) | × | autocorr_cs(y, k) |
excess(k)   = observed(k) - expected(k)
```

where `autocorr_cs(y, k)` is the cross-sectional autocorrelation of the target —
how much entity ordering persists from *t* to *t+k*. This is the panel analogue of
the persistence baseline, and it is essential for exactly the same reason: in a
market with persistent relative performance, an honest cross-sectional feature
will correlate with the future cross-section transitively.

Flag PNL002 when `excess(k) >= threshold` for some k >= 1.

### Step 3 — evidence

```python
evidence = {
    "metric": "cross_sectional_spearman",
    "lag": k,
    "observed_cs_corr": ...,
    "expected_from_cs_persistence": ...,
    "excess": ...,
    "threshold": ...,
    "n_timestamps_scored": ...,
    "median_entities_per_timestamp": ...,
    "cs_normalisation_score": ...,   # from step 1
}
```

Severity: **WARNING**, matching LEK002/LEK003. A genuinely predictive
cross-sectional signal — which is the entire point of a cross-sectional alpha
factor — produces the same signature as a leak, separated only by magnitude. It
must not be CRITICAL.

---

## Why this is hard, and what could go wrong

Stated plainly, because these need resolving before implementation:

**1. A good factor and a leak look alike.** This is LEK002's problem, amplified.
A real cross-sectional alpha factor is *supposed* to correlate with next-period
relative returns. Realistic factor rank-ICs are 0.02–0.08; a leak is above 0.5.
The separation is magnitude only, so threshold choice is doing all the work and
must be justified empirically, not guessed.

**2. Ragged panels break the cross-section.** If the entity set changes between
*t* and *t+k*, `corr_e` is computed over different populations. Requires an inner
join per timestamp and a minimum-entity guard — probably 20, below which a
cross-sectional correlation is noise. PNL001 already detects raggedness and should
gate this check.

**3. Cost.** Naively this is O(timestamps × lags × features) cross-sectional
correlations. For 3,000 days × 5 lags × 50 features that is 750,000 correlations
over ~500 points each. Needs vectorising over entities, and should probably be
opt-in rather than on by default.

**4. Survivorship interacts badly.** If the panel only contains entities that
survived to the end, the cross-section at *t* is already contaminated by future
information in a way this check would partly attribute to the feature.

**5. Sector/industry neutralisation is a legitimate confound.** Sector-neutralised
features are computed within groups-within-the-cross-section, and their behaviour
under this test is unclear without experimentation.

---

## Proposed scope

**In scope**

- New code PNL002 in `tsauditor/panel.py`, WARNING severity
- Opt-in via `scan(..., group_col=..., check_cross_sectional=True)`
- Requires `group_col` and `target`
- Gated on PNL001: skip, with a clear reason in the evidence, if the panel is
  too ragged to form stable cross-sections
- Minimum entities per timestamp (default 20) and minimum scored timestamps
- Documentation with a worked example showing a correct rank feature passing and
  a shifted one being caught
- Tests including **negative cases**: a legitimate cross-sectional factor with
  realistic IC must not be flagged

**Out of scope for the first version**

- Automatic repair. As with LEK004, the fix is to re-align the feature, not to
  drop it — and only the author knows the correct alignment.
- Multi-factor / combination cross-sectional leakage.
- Anything requiring a fitted model. Keeping this rank-based preserves the
  library's "no model, no hyperparameters" property.

---

## Acceptance criteria

- [ ] A correctly-computed cross-sectional rank feature is **not** flagged
- [ ] The same feature shifted by −1 **is** flagged, with `lag == 1`
- [ ] **Detection stays at 100% across the whole mkt/idio sweep above (0 → 100),**
      where LEK002 falls from 100% to 22.5%. This is the specific gap the check
      exists to close, and it is the single most important test.
- [ ] A simulated alpha factor with rank-IC ≈ 0.05 is **not** flagged
- [ ] A ragged panel causes the check to skip with a stated reason, not to produce
      a spurious finding
- [ ] Runtime on a 500-entity × 3,000-day × 20-feature panel is documented
- [ ] Threshold choice is justified against simulated data across a range of ICs,
      not asserted
- [ ] Docs include the failure modes above, per the project's
      "Limitations and false positives" convention
- [ ] Docs state plainly that LEK002/LEK003 already catch this case when the common
      factor is weak, so users understand what PNL002 adds rather than assuming the
      existing checks were useless

The simulation used to produce the table above should be committed as the basis for
these tests, so the claims are reproducible rather than asserted.

---

## Open questions

1. Should step 1 (candidate identification) gate the check, or should every
   numeric column be tested? Gating is much cheaper but will miss cross-sectional
   features that are not normalised — a raw market-relative return, for instance.
2. Is cross-sectional Spearman the right statistic, or should this use a rank-IC
   formulation that quant practitioners will recognise immediately?
3. Should the persistence baseline use cross-sectional autocorrelation of the
   target, or of the feature? LEK003 uses the target's; the panel analogue is not
   obviously identical.
4. Does this need a `min_timestamps` floor analogous to `min_obs`, and what is a
   defensible default?

---

## Prior art worth reviewing before implementing

- The rank-IC / information-coefficient literature in quantitative equity, which
  has decades of practice on exactly this statistic
- `alphalens`-style factor analysis tooling, which computes forward-return ICs
  routinely but frames them as *performance* rather than as a *leakage warning*
- The point-biserial reasoning already documented for LEK001 — the same care about
  choosing a metric that answers the actual question applies here

---

## Why this matters

The OGDC case that motivated this library was a univariate leak, and tsauditor
catches it. Cross-sectional leakage is the same class of mistake — a feature that
knows something it should not — one dimension up. It is harder to spot by eye,
more common in production quant pipelines than the OGDC bug was, and produces the
same seductive symptom: a backtest that looks excellent.

Shipping a weak version would be worse than shipping nothing, because leakage
detection is this library's credibility. Hence: proposed, scoped, and explicitly
not implemented until the threshold work is done properly.
