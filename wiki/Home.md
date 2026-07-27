# tsauditor

A data quality auditing library for **time-series tabular data**, with a focus on financial and sensor domains.

`tsauditor` scans a pandas (or polars) DataFrame and returns a structured report of structural problems, anomalies, and — its core contribution — **data leakage** between features and the prediction target. Since **0.2.0** it can also *repair* the flagged issues on a copy, score data health, export a formal report, and hand a clean array to a forecasting model.

```python
import tsauditor as tsa

report = tsa.scan(df, target="Direction", domain="finance")
report.summary()
```

---

## Why tsauditor exists

A direction-prediction model on OGDC (Pakistani equity) data reached **99.68% accuracy**. That number was a lie.

One feature, `ChangeP`, was the same-day percentage change in price. The target, `Direction`, was defined as `1 if ChangeP > 0 else 0`. The feature did not merely *predict* the target — it *was* the target, wearing a different name. Remove it and accuracy falls to **69.81%**, which is the honest number.

Nothing about `ChangeP` looks wrong when you inspect it. It is an ordinary float column with an ordinary distribution and no missing values. Standard profiling tools do not catch this, because they treat tabular rows as independent samples and never ask *when* each value became knowable relative to the moment of prediction.

That is the class of mistake `tsauditor` exists to catch automatically. See the [OGDC Case Study](OGDC-case-study) for the full worked example.

---

## Where to go next

**If you are new here, read these in order:**

| Page | What it covers |
| ---- | -------------- |
| [Installation](Installation) | `pip install`, extras, development setup |
| [Quickstart](Quickstart) | Your first scan, repair, and export — 5 minutes |
| [How It Works](How-it-works) | The scan pipeline, end to end |

**When you need to understand a specific finding:**

| Page | What it covers |
| ---- | -------------- |
| [Issue Code Reference](Issue-code-reference) | Lookup table for every PRF / ANO / LEK / VAL code |
| [Profiler Detectors](Detectors-Profiler) | PRF001–PRF006 — gaps, duplicates, missingness, stationarity |
| [Anomaly Detectors](Detectors-Anomaly) | ANO001–ANO003 — stuck values, outliers, contextual spikes |
| [Leakage Detectors](Detectors-Leakage) | LEK001–LEK004 — the core of the library |
| [Validity Detectors](Detectors-Validity) | VAL001–VAL002 — declared domain rules |
| [Panel Data](Panel-Data) | PNL001, PNL003 — multi-entity / long-format data |

**Reference material:**

| Page | What it covers |
| ---- | -------------- |
| [API Reference](API-Reference) | Every public function, parameter, and return type |
| [Remediation](Remediation) | What `apply_fixes` actually changes, and the health score |
| [Domain Presets](Domain-Presets) | Exactly what `domain="finance"` vs `"sensor"` changes |
| [OGDC Case Study](OGDC-case-study) | The flagship leakage example, start to finish |
| [Internals](Internals) | Private helpers, for contributors |
| [Contributing](Contributing) | How to open a PR or propose a check |

---

## At a glance

```python
import tsauditor as tsa

report = tsa.scan(df, target="Direction", domain="finance")
report.summary()          # rich CLI table
report.critical           # list of issues that block modeling
report.leaky_columns()    # the shortlist of features to review or remove
report.to_json("out.json")

# Repair on a copy and keep the audit trail (the original is untouched):
clean, report = tsa.fix(df, target="Direction", domain="finance")
```

```
────────────────── tsauditor Report ──────────────────

Critical: 2  Warnings: 39  Info: 4

 Severity   Code    Module    Column    Description
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 CRITICAL   LEK001  leakage   ChangeP   Feature near-deterministically
                                        reproduces target 'Direction'
                                        (auc score=1.0000 >= 0.95)
```

---

## Design philosophy

**Advisory by default.** `tsauditor` detects and suggests. It never edits your data *unless* you explicitly call `apply_fixes()` or `fix()` — and even then it works on a **copy**, leaving your original untouched, and it never repairs the target label.

**Time-aware.** Every check reasons about the temporal order of your data, not just its distribution. This extends to *point-in-time* availability (LEK004): a value must not be used before it was published.

**Declarative, not magical.** `tsauditor` never guesses release dates or validity rules. As-of leakage (`available_at=`) and validity constraints (`constraints=`) are opt-in — you declare them, `tsauditor` verifies them.

**Domain-aware.** Finance and sensor data have different thresholds for "normal." The `domain` parameter tunes the checks accordingly. See [Domain Presets](Domain-Presets).

**Programmatic-first.** The report is a structured Python object, not just a printed table. Filter by code, module, or severity; export to JSON or PDF; integrate into pipelines.

**Honest about limits.** Several checks are *suspicion* flags, not proofs. Every detector page has a "Limitations and false positives" section that states plainly when the check is wrong. A tool that hides its failure modes is worse than no tool.
