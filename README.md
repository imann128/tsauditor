# tsauditor
[![CI](https://github.com/imann128/tsauditor/actions/workflows/ci.yml/badge.svg)](https://github.com/imann128/tsauditor/actions/workflows/ci.yml)
![Clones](https://img.shields.io/badge/clones-1000-brightgreen?logo=github)
[![codecov](https://codecov.io/github/imann128/tsauditor/graph/badge.svg)](https://codecov.io/github/imann128/tsauditor)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A data-quality auditing library for **time-series tabular data**, with a focus on
financial and sensor domains. `tsauditor` scans a `DataFrame` and returns a 
structured report of structural problems, anomalies, and — its core contribution —
**data-leakage** between features and the prediction target.

The project grew out of a real bug in a Pakistani equity (OGDC) direction-prediction
model: a same-day percentage-change feature (`ChangeP`) was mathematically near-identical
to the target it was meant to predict. With `ChangeP` included, a Random Forest
classifier reached 99.68% accuracy (AUC 0.9987); a Gradient Boosting classifier reached
the same 99.68% accuracy (AUC 0.9967). Removing it — along with same-day `Open`, `High`,
and `Low`, which are equally unavailable at prediction time — dropped accuracy to 69.81%
(RF, AUC 0.7795) and 73.70% (GBM, AUC 0.8072) on a held-out test period
(2025-01-09 to 2026-04-03). Both models still beat a 50% baseline, but the headline
accuracy had been almost entirely an artifact of the leak. `tsauditor` exists to catch
this class of mistake automatically before it reaches a model.
See [`examples/ogdc_leakage_case`](examples/ogdc_leakage_case) for the full experiment,
script, and measured results.

## Installation

```bash
pip install tsauditor
```

Requires Python ≥ 3.9. Core dependencies: `pandas`, `numpy`, `scipy`, `statsmodels`, `rich`.

### Development setup

```bash
git clone https://github.com/imann128/tsauditor.git
cd tsauditor
pip install -e ".[dev]"
```

## **Note:** Set domain="None" for domain agnostic usage. Similarly, it works well without defining a domain at all.

**For usage snippets, scroll down in the readme or check out the [examples](./examples) directory for sample scripts**

## Quickstart

```python
import tsauditor as tsa

report = tsa.scan(df, target="Direction", domain="finance")

report.summary()                 # rich-formatted CLI table
report.critical                  # list[Issue] that block modeling
report.filter(module="leakage")  # programmatic filtering
report.to_json("report.json")    # structured export
```

`scan()` returns a `GuardReport` holding `Issue` dataclasses bucketed by severity
(`critical`, `warnings`, `info`) plus dataset metadata.

### Output:
![financial_report](images/financial_report.png)


## Sensor:

###  Real-World Sensor Validation Example

Below is an example using real weather station telemetry data. To showcase how `tsauditor` behaves during typical field failures, we manually inject three classic hardware faults: a frozen sensor reading, a complete network dropout gap, and a high-voltage electrical spike.

```python
import pandas as pd
import tsauditor as tsa

print(" Fetching real-world weather station sensor dataset...")
url = "[https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv](https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv)"

try:
    df = pd.read_csv(url, parse_dates=["Date"], index_col="Date")
    df.columns = ["air_temperature"]
    print(" Dataset successfully into memory")
except Exception as e:
    print(f" Error loading dataset: {e}")

print(" Injecting typical hardware field failures for evaluation...")
# 1. Stuck sensor condition: flatlined at 12.2°C for 15 days straight
df.iloc[100:115] = 12.2

# 2. Transmission blackout: 10 days of completely missing telemetry
df.iloc[300:310] = None

# 3. Electrical surge: an impossible 75°C transient spike
df.iloc[500] = 75.0

print("\n Running `tsauditor` validation sweep")

# Execute the audit using the optimized sensor preset
report = tsa.scan(df, domain="sensor")
report.summary()
```
### Output:
![sensor_report](images/sensor_report.png)

## What it checks

| Module | Code | Severity | Detects |
|--------|------|----------|---------|
| profiler | PRF001 | warning | Irregular timestamp frequency |
| profiler | PRF002 | warning | Clustered missing values |
| profiler | PRF003 | info | Non-stationarity (Augmented Dickey-Fuller) |
| profiler | PRF004 | warning | Duplicate timestamps |
| profiler | PRF005 | warning | Clustered gaps |
| profiler | PRF006 | warning | High overall missing rate |
| anomaly | ANO001 | warning | Stuck / repeated constant values |
| anomaly | ANO002 | warning | Point outliers (z-score + IQR) |
| anomaly | ANO003 | warning | Contextual spikes (local rolling z-score) |
| leakage | LEK001 | critical | Target equivalence (feature reproduces the target) |
| leakage | LEK002 | warning | Positive-lag cross-correlation peak (future info) |
| leakage | LEK003 | warning | Rolling-window lookahead (excess over persistence) |

### Leakage detection (the research core)

Leakage checks are **rank-based**, chosen by target type:

- **LEK001 — equivalence.** Continuous targets use `|Spearman ρ|`; binary targets use
  **AUC separation** (`max(AUC, 1−AUC)`). This is deliberate: Pearson against a binary
  0/1 target is point-biserial correlation, which is capped near `√(2/π) ≈ 0.798`, so a
  feature whose sign *defines* the target scores only ~0.80 and slips under a naive
  threshold. AUC scores it 1.0.
- **LEK002 — cross-correlation.** Flags features whose peak association with the target
  falls at a *positive* lag (the feature aligns with the target's future).
- **LEK003 — temporal lookahead.** Flags features that correlate with the future target
  *beyond* what the target's own autocorrelation can explain — the signature of a
  forward-looking or centered window. The persistence baseline is what keeps a
  legitimate trailing feature from being false-flagged.

LEK002/LEK003 are WARNING-level *suspicions*: in pure cross-correlation a genuine strong
predictor and a leak are distinguishable only by magnitude. LEK001 is CRITICAL because
equivalence is near-deterministic.

## Architecture

```
tsauditor/
├── scanner.py          # scan() — orchestrates all modules into a GuardReport
├── profiler/           # structural checks: frequency, missing, stationarity
├── anomaly/            # point.py, contextual.py
├── leakage/            # equivalence.py, correlation.py, temporal.py
├── report/summary.py   # GuardReport + Issue dataclasses, rich/JSON output
└── utils/validation.py # input validation & DataFrame normalization
```

## Testing

```bash
pytest -q
```

## Contributing

Contributions are welcome. Check [open issues](https://github.com/imann128/tsauditor/issues)
for ideas, or look for the `good first issue` label. Run `pytest -q` before opening a PR —
all 93 tests must pass, and CI will verify this across Python 3.9–3.14 on Linux, Windows, and macOS.


## Status

Beta (`0.1.2`). Profiler, anomaly, and leakage modules are implemented and tested
(93 tests passing, CI across Python 3.9–3.14 on Linux, Windows, macOS).

## License

MIT — see [LICENSE](LICENSE).
