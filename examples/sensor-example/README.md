# Sensor Example

A self-contained, runnable walkthrough of `tsauditor` on the **sensor** domain —
the companion to the finance-focused [OGDC leakage case](../ogdc_leakage_case).
Where OGDC shows the leakage checks catching a feature that secretly *is* the
target, this example shows the structural and anomaly checks catching raw data
that is quietly broken before modeling ever starts.

Addresses [issue #13](https://github.com/imann128/tsauditor/issues/13).

## What it demonstrates

`sensor_example.ipynb` builds a synthetic 14-day hourly stream (336 rows) of
temperature and humidity, injects two of the most common real-world sensor
failures, and audits it with a single `tsa.scan(df, domain="sensor")` call:

| Injected fault | Column | Detected as | Severity |
|----------------|--------|-------------|----------|
| Stuck sensor — 6 h frozen on one value | `temperature` | `ANO001` (stuck values) | warning |
| Collection outage — 6 h of missing readings | `humidity` | `PRF002` (clustered missing) | warning |

Both faults are sized to the library's `domain="sensor"` thresholds
(`stuck_window=3`, `cluster_threshold=3`). The outage is kept to ~1.8 % of the
column so it surfaces as a *clustered* run rather than tripping the high
missing-rate check (`PRF006`). The notebook closes by re-scanning the
**un-faulted** stream to confirm a clean frame raises nothing — i.e. the
auditor is specific, not trigger-happy.

## Running it

From the repository root:

```bash
pip install -e ".[dev]"          # tsauditor + test deps
pip install jupyter matplotlib   # notebook runtime + plotting
jupyter notebook examples/sensor_example/sensor_example.ipynb
```

Then run all cells. No changes to any code under `tsauditor/` are required —
this is purely an example.

## Expected output

The audit on the faulted frame reports exactly two warnings and nothing else:

```
Critical: 0  Warnings: 2  Info: 0

  WARNING  ANO001  anomaly   temperature   Stuck values detected.
  WARNING  PRF002  profiler  humidity      Clustered missing value sequences ...
```

Re-scanning the clean frame reports `Critical: 0  Warnings: 0  Info: 0`.

## Notes

- **Synthetic by design.** Issue #13 calls for synthetic data that clearly
  triggers the checks; this keeps the faults unambiguous and the notebook fully
  reproducible. The generator mirrors the `sensor_df` fixture in
  `tests/conftest.py`.
- **A realistic wrinkle.** The frozen segment is placed symmetrically around a
  daily peak so the sensor "recovers" near its pre-freeze value. A sharper
  recovery jump would also trip `ANO003` (contextual spike) — correct
  behaviour, kept out of the way here for a clean two-check demo. Shift
  `stuck_start` onto a steep part of the cycle to see it appear.

## Files

- `sensor_example.ipynb` — the walkthrough (markdown + executed code).
- `README.md` — this file.