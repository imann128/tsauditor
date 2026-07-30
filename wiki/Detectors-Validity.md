# Validity Detectors

One function, two issue codes, in `tsauditor/validity.py`.

| Code | Severity | Detects |
| ---- | -------- | ------- |
| VAL001 | WARNING | A value outside a declared per-column range |
| VAL002 | CRITICAL | A broken ordering relationship between two columns |

**This module is opt-in.** It runs only when you pass `constraints=` to `scan()`.

---

## Statistically surprising vs. definitionally wrong

The [anomaly module](Detectors-Anomaly) finds values that are *statistically* surprising, unusual given the rest of the column. This module finds values that are *definitionally* wrong, impossible regardless of what the rest of the column looks like.

The difference is not academic:

| Value | Anomaly module | Validity module |
| ----- | -------------- | --------------- |
| A bid-ask spread of −1.0 | Might be flagged, if it is far from the mean | **Always** flagged: a spread cannot be negative |
| A sentiment score of 1.8 on a [−1, 1] scale | Probably not flagged; 1.8 is not far from 1.0 | **Always** flagged: it is outside the scale |
| A traded volume of −5,000 | Might be flagged | **Always** flagged: volume cannot be negative |
| A price of 10,000 in a series that averages 100 | Flagged as an outlier | Not flagged unless you declared a bound |

A statistical check can only ever say "this is unusual." It cannot say "this is impossible," because impossibility is a fact about the world, not about the data.

**And `tsauditor` cannot guess these rules.** It has no way to know that a column named `sentiment` is bounded to [−1, 1], or that `bid` must never exceed `ask`. Column names are not a contract. So you declare the rules once, and the check enforces them on every row.

---

## Two kinds of rule

### Bounds: per-column limits
```python
constraints={"bounds": {
    "spread":    {"min": 0, "min_exclusive": True},   # must be strictly > 0
    "sentiment": {"min": -1, "max": 1},               # must be within [-1, 1]
    "volume":    {"min": 0},                          # must be >= 0
}}
```

Recognised keys per column:

| Key | Type | Meaning |
| --- | ---- | ------- |
| `min` | number | Lower limit |
| `max` | number | Upper limit |
| `min_exclusive` | bool | If `True`, the limit is strict: `value > min` required, not `value >= min` |
| `max_exclusive` | bool | If `True`, `value < max` required |

Both limits are optional; you may set either or both. Defaults are inclusive.

The exclusive flags matter more than they look. A bid-ask spread of exactly 0 is not merely unusual, it means the bid equals the ask, which is a locked or crossed market and usually a feed glitch. `{"min": 0}` permits it; `{"min": 0, "min_exclusive": True}` catches it.

### Relations: ordering between two columns
```python
constraints={"relations": [("bid", "ask"), ("low", "high")]}
```

Each tuple is `(low_column, high_column)` and asserts `low <= high` on every row where **both** values are present.

This catches structural corruption that no per-column bound can see. A bid of 105 is perfectly valid. An ask of 104 is perfectly valid. A bid of 105 alongside an ask of 104 is a **crossed book**, an impossible market state, and a certain sign of a broken feed. Only a cross-column rule finds it, which is why VAL002 is CRITICAL while VAL001 is WARNING.

### The flat shorthand

If you pass a dict with neither a `"bounds"` nor a `"relations"` key, `scan()` treats the whole thing as bounds:

```python
constraints={"spread": {"min": 0}}                 # shorthand
constraints={"bounds": {"spread": {"min": 0}}}     # identical, explicit
```

The explicit form is clearer, and it is the only form that lets you add relations later.

---

## Signature

```python
audit_validity(
    df: pd.DataFrame,
    bounds: Optional[Dict[str, Dict[str, Any]]] = None,
    relations: Optional[Sequence[Tuple[str, str]]] = None,
) -> List[Issue]
```

Note this function takes `bounds` and `relations` as **separate arguments**. The single `constraints=` dict is a `scan()`-level convenience that is unpacked before this function is called. If you call `audit_validity` directly, pass them separately.

Unlike every other detector, this one does **not** require a `DatetimeIndex`. The checks are row-wise and have no temporal component.

---

## How it works

**Bounds**, per declared column:

1. Verify the column exists and is numeric, else `ValueError`.
2. Build a violation mask over rows where the value is present:
   - `value <= min` if `min_exclusive`, otherwise `value < min`
   - `value >= max` if `max_exclusive`, otherwise `value > max`
   - combined with OR
3. If no violations, move on. Otherwise raise one VAL001 for the column.

A column whose spec sets neither `min` nor `max` is skipped silently.

**Relations**, per declared pair:

1. Verify both columns exist and are numeric, else `ValueError`.
2. Find rows where both are present and `low > high`.
3. If any, raise one VAL002. The issue's `column` is set to the **high** column.

NaN handling is consistent throughout: a missing value is never a violation. Only rows with actual data are judged.

---

## Evidence

**VAL001, Out-of-range value.** WARNING.

| Key | Meaning |
| --- | ------- |
| `n_violations` | Number of offending rows |
| `min`, `max` | The declared limits (`None` if unset) |
| `min_exclusive`, `max_exclusive` | Whether each limit was strict |
| `observed_min` | Smallest **violating** value |
| `observed_max` | Largest **violating** value |
| `check` | `"bounds"` |

Read `observed_min` and `observed_max` carefully: they describe the *violating* values only, not the column as a whole. They are useful for diagnosis. Values of 1.8 and −2.4 against a [−1, 1] scale suggest a handful of glitches; values of 180 and −240 suggest someone forgot to divide by 100.

**VAL002, Relation violation.** CRITICAL.

| Key | Meaning |
| --- | ------- |
| `n_violations` | Number of offending rows |
| `low_col`, `high_col` | The pair as you declared it |
| `first_violation` | Timestamp of the earliest offending row |
| `check` | `"relation"` |

---

## When it does not fire

- You did not pass `constraints=`, the whole module is skipped
- A column's spec has neither `min` nor `max`
- All values satisfy the rules
- Violating rows contain NaN in the relevant column(s)

Raises `ValueError` if a declared column is missing from the DataFrame or is non-numeric. This is deliberate: a typo in a column name should fail loudly, not silently disable your check.

---

## Worked example

```python
import pandas as pd
from tsauditor.validity import audit_validity

idx = pd.date_range("2024-01-01", periods=6, freq="D")
df = pd.DataFrame({
    "bid":       [100.0, 101.0, 102.0, 103.5, 104.0, 105.0],
    "ask":       [100.5, 101.5, 101.0, 104.0, 104.5, 105.5],  # row 3: crossed
    "spread":    [0.5,   0.5,   -1.0,  0.5,   0.5,   0.5  ],  # row 3: negative
    "sentiment": [0.1,   0.5,   -0.3,  1.8,   0.0,   -2.4 ],  # rows 4, 6: off-scale
}, index=idx)

issues = audit_validity(
    df,
    bounds={
        "spread":    {"min": 0, "min_exclusive": True},
        "sentiment": {"min": -1, "max": 1},
    },
    relations=[("bid", "ask")],
)

for i in issues:
    print(i.code, i.severity, i.column, i.evidence)
```

```
VAL001 warning spread {'n_violations': 1, 'min': 0, 'max': None, 'min_exclusive': True, 'max_exclusive': False, 'observed_min': -1.0, 'observed_max': -1.0, 'check': 'bounds'}
VAL001 warning sentiment {'n_violations': 2, 'min': -1, 'max': 1, 'min_exclusive': False, 'max_exclusive': False, 'observed_min': -2.4, 'observed_max': 1.8, 'check': 'bounds'}
VAL002 critical ask {'n_violations': 1, 'low_col': 'bid', 'high_col': 'ask', 'first_violation': '2024-01-03 00:00:00', 'check': 'relation'}
```

All three planted faults were caught. The descriptions are written to be readable on their own:

```
Column 'spread' has 1 value(s) outside its declared valid range (bounds min=0,
max=None); observed [-1.0, -1.0]. These may be data-feed glitches or a scaling error.

Ordering constraint 'bid <= ask' is violated on 1 row(s) (first at 2024-01-03
00:00:00), e.g. a crossed book where 'bid' exceeds 'ask'. Inspect these
timestamps for feed glitches.
```

Note that row 3 triggered **two** issues, from two different angles: the spread went negative (VAL001) and the book crossed (VAL002). They are the same underlying corruption seen through two rules. Redundant checks like this are a feature, either rule alone might have been the one you remembered to declare.

---

## Validity issues are not leakage

`report.leaky_columns()` returns only columns flagged by the **leakage** module. A VAL001 or VAL002 finding never appears there, even though VAL002 is CRITICAL.

This is correct. A crossed book is a data error, not a modeling risk, the remedy is to fix or drop the affected rows, not to remove the feature from your model.

Note also that `apply_fixes()` does **not** repair validity issues. There is no automatic remedy: whether an off-scale sentiment score should be clipped to 1.0, set to NaN, or investigated as a scaling bug depends entirely on why it happened. `tsauditor` reports it and leaves the decision to you.

---

## Limitations

**Nothing is checked unless you declare it.** Silence from this module means only that you declared nothing, or that what you declared held. It is not evidence that your data is valid.

**Numeric columns only.** A category column with an impossible label cannot be checked here.

**Only `low <= high` relations are supported.** There is no way to express `a + b == c`, `a != b`, or any other cross-column rule.

**Bounds are constant over time.** You cannot express a limit that changes across the series, for example, a price band that widens after a known regime change.

**One issue per column or per pair**, not per row. `n_violations` gives the count; recovering the actual row positions means rebuilding the mask yourself.
