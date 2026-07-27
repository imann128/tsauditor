# How It Works

This page describes what `tsa.scan()` actually does, in the order it does it. If you want to know *why* a particular check exists or *how* it decides, follow the link to that detector's page.

---

## The pipeline

```
your DataFrame
      │
      ▼
1. validate_dataframe()      normalize input, resolve the time index, sort
      │
      ▼
2. build metadata            rows, columns, time range, inferred frequency
      │
      ├─────────────── group_col given? ───────────────┐
      │ no                                          yes│
      ▼                                                ▼
  run the checks once                    2a. panel structure  PNL001, PNL003
  on the whole frame                     2b. cross-sectional  PNL002
      │                                                │
      │                                  2c. for each entity: run the same
      │                                      checks below, tagging every
      │                                      Issue with .group
      │                                                │
      ├────────────────────┬───────────────────────────┘
      ▼
3. profiler   (if run_profiler)     PRF001–PRF006
      │
      ▼
4. anomaly    (if run_anomaly)      ANO001–ANO003
      │
      ▼
5. leakage    (if run_leakage and target is not None)   LEK001–LEK003, LEK005
      │
      ▼
6. as-of      (if available_at was given)               LEK004
      │
      ▼
7. validity   (if constraints was given)                VAL001–VAL002
      │
      ▼
   GuardReport
```

Each stage appends `Issue` objects to the report. Issues are routed into `critical`, `warnings`, or `info` by their severity — the stage they came from does not decide where they land.

**The panel branch is the whole of panel support.** Stages 3–7 are identical either way; they simply run once per entity instead of once. No detector knows what a panel is, which is why panel and single-series results stay consistent by construction. → [Panel Data](Panel-Data)

---

## Stage 1: Validation and normalization

Before any check runs, `validate_dataframe()` puts your input into a known-good shape. This is the only stage that can reject your data outright.

**Polars conversion.** If you passed a polars DataFrame, it is converted to pandas here. Polars has no index, so `time_col=` is mandatory in that case; without it you get a clear error rather than a confusing failure later.

**Type check.** Anything that is not a `pd.DataFrame` raises `TypeError`. An empty DataFrame raises `ValueError`.

**Copy.** `tsauditor` works on a copy from this point on. Your original object is never touched, not even during validation.

**Time index resolution**, in this order:

1. If `time_col=` was given, that column is parsed with `pd.to_datetime` and set as the index.
2. If the index is already a `DatetimeIndex`, it is used as-is.
3. If the index is **numeric**, `tsauditor` **refuses** and raises an error.
4. Otherwise (string or object labels), it attempts one coercion to datetime, and raises a clear error if that fails.

Step 3 deserves explanation, because refusing to do something looks unhelpful. `pd.to_datetime([0, 1, 2])` succeeds — it interprets those integers as nanoseconds since the epoch and returns three timestamps a few nanoseconds apart in 1970. Every gap, frequency, and clustering result computed on that index would be nonsense, and nothing would visibly fail. Refusing a numeric index is the safer behaviour.

**Sort.** The frame is sorted ascending by the time index. Every downstream check assumes chronological order.

**Target check.** If `target=` was given and the column does not exist, `ValueError`.

→ Details in [Internals](Internals#validation).

---

## Stage 2: Metadata

Cheap, dataset-level facts are recorded on the report:

```python
report.metadata
```

```python
{
    'rows': 1537,
    'columns': 24,
    'time_start': '2020-01-22',
    'time_end': '2026-04-03',
    'frequency': 'daily',
    'target': 'Direction',
    'domain': 'finance',
}
```

`frequency` is a coarse label — one of `"sub-daily"`, `"daily"`, `"weekly"`, `"monthly"`, `"irregular"`, or `"unknown"` — derived from the *median* gap between consecutive timestamps. It is intentionally rough; precise gap analysis is the profiler's job.

The metadata is not decoration. `apply_fixes` later reads `metadata["domain"]` to pick thresholds and `metadata["target"]` to know which column to protect from repair.

---

## Stage 3: Profiler — is the data structurally sound?

The profiler asks questions about the *shape* of your data, mostly ignoring the values themselves.

| Check | Code | Question it asks |
| ----- | ---- | ---------------- |
| `audit_frequency` | PRF001, PRF004, PRF005 | Are timestamps unique, evenly spaced, and gap-free? |
| `audit_stationarity` | PRF003 | Does this column have a stable mean over time? |
| `audit_missing` | PRF002, PRF006 | Are values missing, and are they missing in clumps? |

`audit_frequency` runs first because a duplicate timestamp invalidates everything downstream — it silently corrupts rolling windows, lags, and resampling. It is the only profiler check rated CRITICAL.

`audit_stationarity` is by far the most expensive check in the entire library: the Augmented Dickey-Fuller test fits many regressions per column, searching lags by AIC. Skip it with `run_stationarity=False` when you do not need it.

→ Full details: [Profiler Detectors](Detectors-Profiler)

---

## Stage 4: Anomaly — are individual values plausible?

| Check | Code | Question it asks |
| ----- | ---- | ---------------- |
| `audit_point_anomalies` | ANO002 | Is this value extreme compared to the **whole column**? |
| `audit_contextual_anomalies` | ANO001 | Is this value **frozen**, repeating unchanged? |
| `audit_contextual_anomalies` | ANO003 | Is this value extreme compared to its **neighbours**? |

The ANO002 / ANO003 distinction is the important one and is easy to miss. ANO002 takes a global view: it compares a point against the column's overall mean and quartiles. ANO003 takes a local view: it compares a point only against the surrounding window.

They catch different things. In a series that ramps steadily from 0 to 240, a value of 200 is globally unremarkable — but if its immediate neighbours are all near 58, it is obviously wrong. ANO002 finds nothing; ANO003 flags it. There is a runnable demonstration of exactly this on the [Anomaly Detectors](Detectors-Anomaly) page.

→ Full details: [Anomaly Detectors](Detectors-Anomaly)

---

## Stage 5: Leakage — does a feature know the future?

This is the reason the library exists. All four checks in this stage need `target=`; without it the stage is silently skipped.

| Check | Code | Question it asks |
| ----- | ---- | ---------------- |
| `audit_equivalence` | LEK001 | Is this feature *the target in disguise*? |
| `audit_correlation_leakage` | LEK002 | Does this feature align best with the target's **future**? |
| `audit_temporal_leakage` | LEK003 | Does it know the future better than **persistence alone** allows? |
| `audit_combination_leakage` | LEK005 | Do **several features together** rebuild the target, when none does alone? |

LEK001 and LEK005 are CRITICAL — a feature reproducing the target at AUC 1.0, or a pair reconstructing it at adjusted R² 1.0, is not a judgement call. LEK002 and LEK003 are WARNING-level *suspicion* flags: a genuinely strong predictor and a lookahead leak can produce the same signature, and the honest separator is magnitude, not kind.

LEK005 is the only check here that is **not** univariate. Everything else scores one feature at a time, which is precisely why a target defined as `x1 - x2` slips past all of them — each input correlates with it at only ~0.7.

→ Full details: [Leakage Detectors](Detectors-Leakage)

---

## Stage 6: As-of availability (opt-in)

LEK004 runs only when you pass `available_at=`. It is target-independent, so it runs even when `target=None`.

This check cannot work from values alone. Whether a CPI figure was knowable on the date it sits at depends entirely on when the statistics office published it — a fact that exists nowhere in your DataFrame. So you supply it, and `tsauditor` verifies your data respects it. It never guesses release dates.

→ Full details: [Leakage Detectors](Detectors-Leakage#lek004--as-of-availability-leakage)

---

## Stage 6b: Panel structure and cross-section (only with `group_col`)

Three checks exist only for multi-entity data, and they look at the panel *as a whole* rather than at any one entity.

| Check | Code | Question it asks |
| ----- | ---- | ---------------- |
| `audit_panel_structure` | PNL001 | Do all entities share a **common time index**? |
| `audit_panel_structure` | PNL003 | Is any entity **too short** to audit meaningfully? |
| `audit_cross_sectional_leakage` | PNL002 | Does a feature rank entities in the order their **future** targets will fall? |

PNL002 needs `target=` as well as `group_col=`. It exists because the per-entity checks degrade badly when a common market factor dominates: LEK002's detection of the same leak falls from 100% of entities to 22.5%, while the cross-sectional signal is unaffected.

→ Full details: [Panel Data](Panel-Data)

---

## Stage 7: Validity (opt-in)

VAL001 and VAL002 run only when you pass `constraints=`.

Where the anomaly module finds values that are *statistically surprising*, this stage finds values that are *definitionally impossible*: a negative traded volume, a sentiment score outside [-1, 1], a bid above the ask. `tsauditor` has no way to know that a column named `sentiment` is bounded, so you declare the rule and it enforces it.

A convenience: if you pass a flat mapping with no `"bounds"` or `"relations"` key, it is treated as `bounds`.

```python
constraints={"spread": {"min": 0}}                    # treated as bounds
constraints={"bounds": {"spread": {"min": 0}}}        # identical, explicit
```

→ Full details: [Validity Detectors](Detectors-Validity)

---

## What the report gives you back

```python
report.critical          # list[Issue]
report.warnings          # list[Issue]
report.info              # list[Issue]
report.all_issues        # all of them, sorted by severity then module
report.metadata          # the dict shown above
report.last_fixes        # populated only after apply_fixes / fix

report.filter(code=..., module=..., severity=...)
report.leaky_columns()
report.suggestions()
report.health_score(df)

report.summary()         # print a rich table
report.to_json(path)
report.to_pdf(path)      # needs the [pdf] extra
report.to_dict()
```

→ Full details: [API Reference](API-Reference)

---

## Two properties worth internalizing

**Nothing is ever mutated.** `scan()` copies at validation. `apply_fixes()` returns a new frame. `fix()` returns a new frame. There is no code path in `tsauditor` that writes to the DataFrame you passed in.

**Panel repair partitions first.** When the report came from a `group_col=` scan, `apply_fixes()` repairs each entity separately and writes back by position. Repairing an interleaved panel as one series carries values across entity boundaries — measured, a gap in a series near 10 was filled with ~1000 from the other entity.

**Detectors and repairs share formulas.** `remediate.py` recomputes the outlier, spike, and stuck masks in order to repair them. Those formulas are duplicated from the detector modules, which is a real risk — if one changed and the other did not, `apply_fixes` would repair cells the report never flagged. The test suite pins them together: `tests/test_fix.py` asserts the repaired cell count matches the detector's own evidence. Keep that test passing if you touch either side.
