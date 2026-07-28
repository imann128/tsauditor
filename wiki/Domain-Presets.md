# Domain Presets

The `domain` parameter is a threshold preset. It accepts exactly three values:

```python
tsa.scan(df, domain="finance")
tsa.scan(df, domain="sensor")
tsa.scan(df, domain=None)        # the default
```

Anything else raises `ValueError`.

It changes **five thresholds across four checks**. Everything else in the library ignores it.

## Why presets exist at all

The same number means different things in different fields.

A stock price staying flat for three consecutive trading days is unusual but not impossible — a trading halt does that. The same value repeating for three consecutive hours from a temperature sensor is almost certainly a dead instrument. One stuck-value threshold for both would either flood finance users with false positives or let real sensor failures through.

Presets exist so you get sensible behaviour without configuring every parameter by hand. They are a convenience, not a requirement — see [Choosing a domain](#choosing-a-domain).

---

## The complete table

| Check | Threshold | `"finance"` | `"sensor"` | `None` |
| ----- | --------- | ----------- | ---------- | ------ |
| PRF001 gaps | max gap | **5.0 days** (fixed) | 3 × median gap | 3 × median gap |
| PRF002 missing | cluster length | **5** consecutive | **3** consecutive | **3** consecutive |
| ANO002 outliers | z-score | **5.0** | **3.5** | **4.0** |
| ANO001 stuck | run length | **5** | **3** | **5** |
| ANO003 spikes | local z-score | **4.0** | **3.0** | **3.5** |

Note the asymmetry in the first two rows: `"sensor"` and `None` behave identically for gaps and missing clusters. Only `"finance"` differs there.

---

## Why finance is more permissive

Every finance threshold is **looser** than the default. There is a reason for each.

**Gaps of 5.0 days (PRF001).** Markets close. A daily equity series has a 3-day gap over every weekend, and 4 or 5 days around a public holiday. That is the calendar working normally, not a data defect. An adaptive threshold of `3 × median` would compute roughly 3 days on such a series and flag every single weekend. The fixed 5.0 lets ordinary closures pass while still catching a genuine multi-day outage.

**Missing clusters of 5 (PRF002).** Same reason. If a data vendor aligns a series to a full calendar rather than trading days, weekends appear as consecutive NaNs. Three in a row is routine; five is not.

**Z-score of 5.0 (ANO002).** Financial returns have famously fat tails. Market crashes, earnings surprises, and central bank announcements produce genuine five-sigma daily moves. These are real events, not corrupt data — flagging them as outliers would be both wrong and extremely noisy. The 5.0 threshold reserves the flag for things that are almost certainly errors.

**Stuck runs of 5 (ANO001).** Illiquid instruments genuinely do not trade for several sessions, leaving the last price repeated. Five allows for that; a shorter window would flood you with false positives on thin names.

**Spike threshold of 4.0 (ANO003).** Same fat-tail reasoning as ANO002, applied locally.

---

## Why sensor is stricter

Every sensor threshold is **tighter** than the default, for the mirror-image reason.

**Z-score of 3.5 and spike threshold of 3.0.** Physical measurements are usually well-behaved and close to Gaussian. Temperature, pressure, humidity, and vibration do not have the fat tails that financial returns do. A 3.5-sigma reading from a thermometer is far more likely to be a hardware fault than a real event, so it is worth flagging.

**Stuck runs of 3.** This is the most important sensor threshold. A frozen sensor — one whose hardware has failed but which is still reporting its last reading — is one of the most common and most damaging failure modes in sensor data, and it is invisible to every statistical check because the values look perfectly normal. Three identical consecutive readings from a continuously-sampling instrument already warrants suspicion, and catching it early matters.

**Gaps and missing clusters use the adaptive default.** Sensors sample at wildly different rates — 10 Hz, once a minute, once an hour. A fixed day-based threshold would be meaningless. `3 × median gap` adapts to whatever rate your instrument runs at, with no configuration from you.

---

## What domain does NOT change

This is worth stating explicitly, because the parameter appears in more signatures than it affects.

**It has no effect on any leakage check.** `audit_equivalence`, `audit_correlation_leakage`, and `audit_temporal_leakage` all accept a `domain` argument and all ignore it. The source comment is candid about this: *"Accepted for API consistency; thresholds are driven by target type, not domain."*

The reasoning is sound. Whether a feature reproduces the target is a question about the relationship between two columns. It has nothing to do with whether those columns describe a stock price or a thermometer. A leak is a leak.

**It has no effect on `audit_stationarity`.** PRF003 uses `alpha=0.05` regardless. `domain` is accepted and ignored.

**It has no effect on `audit_validity`.** VAL001 and VAL002 use only the rules you declare.

**It does not change severity.** Every issue's severity is fixed by its code, never by the domain.

**It does not enable or disable any check.** Every check that would run under `domain=None` runs under any domain.

---

## Explicit thresholds always win

`domain` is a source of **defaults**, not an override. Any threshold you pass explicitly is respected, and `domain` is consulted only for the arguments you left as `None`.

```python
audit_point_anomalies(df, zscore_threshold=2.0, domain="finance")   # uses 2.0
audit_contextual_anomalies(df, stuck_window=10, domain="sensor")    # uses 10
audit_missing(df, cluster_threshold=10, domain="finance")           # uses 10
```

This holds consistently across all three detectors that take both.

A deliberate `0` is honoured too. The resolution uses `is None` rather than the `x or default` idiom, so `stuck_window=0` means zero — not "unset":

```python
audit_contextual_anomalies(df, stuck_window=0)    # flags every run longer than 0
audit_point_anomalies(df, zscore_threshold=0.0)   # flags every non-mean point
```

> **Changed behaviour.** Before this was unified, `audit_point_anomalies` consulted `domain` *first* and silently discarded an explicit `zscore_threshold`, while `audit_contextual_anomalies` used `or` and silently swallowed a `0`. If you previously worked around either by leaving `domain=None`, that workaround is no longer needed. Domain presets with no explicit threshold are unchanged. See `tests/test_threshold_resolution.py`.

Note that `scan()` never forwards custom thresholds to the detectors — it passes only `domain` — so this distinction matters only when you call a detector function directly.

---

## `apply_fixes` reads the domain too

The repair layer resolves the same thresholds from `report.metadata["domain"]`, so the domain you scanned with automatically governs how repairs are applied.

```python
report = tsa.scan(df, domain="finance")
clean  = report.apply_fixes(df)      # uses z=5.0, stuck=5, spike=4.0
```

This matters because `apply_fixes` recomputes the outlier and stuck masks in order to repair them. If the repair layer used different thresholds than the detector, it would modify cells the report never flagged. `tests/test_fix.py` asserts these stay in lockstep.

---

## Choosing a domain

**Use `"finance"`** for daily market data — equities, FX, futures, indices. It assumes a daily-ish frequency and a fat-tailed distribution.

Do not use it for **intraday** financial data. The 5.0-day gap threshold means no overnight gap will ever be flagged, and the check becomes useless. Leave `domain=None` and let the adaptive rule work.

**Use `"sensor"`** for physical measurements from instruments at any sampling rate: IoT telemetry, industrial monitoring, environmental data, medical devices. The stricter stuck-value detection is the main benefit.

**Use `None`** — the default — for anything else, or when you are unsure. The defaults sit between the two presets and are reasonable across most data. You lose nothing critical: the leakage checks, which are the reason to use this library, are completely unaffected by the domain.

**Suitable for `"finance"`:** equity prices, indices, FX rates, daily OHLCV data, return series.

**Suitable for `"sensor"`:** IoT sensor streams, environmental monitoring, industrial equipment readings, Raspberry Pi GPIO logs.

## Proposing a new domain preset

New presets — `"crypto"`, `"iot"`, `"healthcare"` — are explicitly welcome as contributions. Each check that branches on `domain` needs one additional clause with justified thresholds.

Sketches of what each would need to argue for:

- **`"crypto"`** — 24/7 trading, so no weekend gaps at all; historically high volatility, so an even wider z-score threshold than finance
- **`"iot"`** — sub-second frequencies and variable sensor reliability; the adaptive gap rule matters more than any fixed value
- **`"healthcare"`** — irregular sampling by nature, and anomaly thresholds that have to be clinically meaningful rather than statistically convenient

To propose one:

1. Open a GitHub issue using the [Feature Request](https://github.com/imann128/tsauditor/issues/new?template=feature_request.md) template
2. Propose specific threshold values **with reasoning for each** — why is this number right for that domain's real-world behaviour?
3. Name at least one check where the difference from the existing presets matters most

Every threshold in this library has a documented justification (fat tails for finance, weekend closures for the 5-day gap). A new preset is held to the same standard. See [Contributing](Contributing).

---

**When in doubt, run all three and compare.** They are cheap, and the differences are informative:

```python
for d in ("finance", "sensor", None):
    r = tsa.scan(df, target="Direction", domain=d, run_stationarity=False)
    print(f"{str(d):8s} critical={len(r.critical):3d} warnings={len(r.warnings):3d}")
```

If a finding survives all three presets, it is robust. If it appears only under `"sensor"`, it is sitting near the threshold and deserves a closer look before you act on it.
