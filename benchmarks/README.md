# Accuracy benchmark

`accuracy_suite.py` measures tsauditor's leakage-detection accuracy
(precision, recall, false-positive rate) against real datasets whose leaky
columns are **independently documented** — not synthetic data with ground
truth by construction, and not a leak injected into otherwise-clean data
for this benchmark's own sake. Each case names its source; go verify it.

Run it:

```
python benchmarks/accuracy_suite.py
```

Prints a summary and writes `benchmarks/results/accuracy_report.md`.


## Current cases

- **ogdc_changep** / **ogdc_changep_statistical_only** : OGDC (Pakistan
  Stock Exchange) equity data, tsauditor's own original motivating case.
  Deliberately split into two runs because the five documented leaky
  columns split across two different leakage *mechanisms*: two
  (`ChangeP`, `Returns`) are statistical target-equivalence, caught by
  LEK001 from the values alone; three (`Open`, `High`, `Low`) are leaky
  only because of *when* they become available, which LEK004 can only
  catch if the caller supplies that fact via `available_at` no
  statistical test can infer publish timing from values alone. Reporting
  both runs side by side keeps the LEK001-only number (100% precision,
  40% recall) visible instead of hidden behind the combined
  LEK001+LEK004 number (100%/100%). Both are real; conflating them would
  overstate what tsauditor can do unaided.

- **alfred_gdp_revision** / **alfred_gdp_revision_statistical_only** —
  real US GDP (GDPC1) from ALFRED, the Federal Reserve's own real-time
  vintage database and the standard tool in the macro-forecasting
  literature for exactly this leak. Same before/after split as the OGDC
  pair, for the same reason: with the real vintage-release date supplied
  as `available_at`, LEK004 catches the later-revised GDP column
  cleanly (100%/100%); without it, 0% recall, because publish timing
  genuinely cannot be inferred from the values alone (the two vintages
  correlate at r=0.998 — this is a harder case than a near-duplicate
  column precisely because the numbers barely differ). Data:
  `benchmarks/data/gdpc1_vintages.csv`, six vintage snapshots
  (1995/2005/2010/2015/2020/2024) of quarterly GDP, 1947-2023 — provided
  by the project owner, since this sandbox's network is allowlisted and
  cannot reach `alfred.stlouisfed.org`/`api.stlouisfed.org` directly.

## What's not here yet, and why

One more real, independently-documented case was identified but isn't
implemented yet:

- **GEFCom (Global Energy Forecasting Competition)** — real ISO New
  England hourly demand/temperature data (2003-2017) is available via
  `github.com/camroach87/gefcom2017data`, genuinely real and downloadable.
  Not added yet because the specific leak this project was considering
  (actual vs. forecast weather used as a feature) has no third party's
  published write-up tying it to *this* dataset — only the general
  mechanism is well documented in the load-forecasting literature. Adding
  it now would mean self-constructing the leak on real data, which is a
  legitimate and useful kind of validation, but a different one than what
  this file's cases claim to be (see the module docstring's "Adding a
  case" section) — it shouldn't be quietly folded into the same set
  without saying so.
