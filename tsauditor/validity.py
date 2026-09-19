"""
tsauditor.validity
-------------------
Domain validity checks: values that are structurally impossible or out of a
declared range.

Where the anomaly module finds points that are *statistically* surprising, this
module finds points that are *definitionally* wrong: a negative traded volume,
a sentiment score outside [-1, 1], a bid-ask spread of zero or below, or a
crossed order book where the bid exceeds the ask. tsauditor cannot guess these
rules (it does not know a column named "sentiment" is bounded), so the caller
declares them and this check verifies the data obeys them.

Two kinds of rule:

- **bounds**: per-column lower/upper limits (inclusive by default; set
  ``min_exclusive`` / ``max_exclusive`` for strict bounds). Example: a spread
  must be strictly positive → ``{"spread": {"min": 0, "min_exclusive": True}}``.
- **relations**: ordered ``(low, high)`` column pairs that must satisfy
  ``low <= high`` on every row. Example: ``("bid", "ask")`` catches a crossed
  book.

Issue codes raised
------------------
VAL001  Out-of-range value: a column violates its declared bound(s). WARNING.
VAL002  Relation violation: an ordering constraint between two columns is
        broken (e.g. a crossed book). CRITICAL.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from tsauditor.report.summary import Issue, WARNING, CRITICAL


def _require_numeric(df: pd.DataFrame, col: str, where: str) -> None:
    if col not in df.columns:
        raise ValueError(f"validity {where} column '{col}' not found in DataFrame.")
    if not pd.api.types.is_numeric_dtype(df[col]):
        raise ValueError(f"validity {where} column '{col}' is not numeric.")


def audit_validity(
    df: pd.DataFrame,
    bounds: Optional[Dict[str, Dict[str, Any]]] = None,
    relations: Optional[Sequence[Tuple[str, str]]] = None,
    **_: object,
) -> List[Issue]:
    """
    Check declared domain-validity rules.

    Parameters
    ----------
    df : pd.DataFrame
        The data to check.
    bounds : dict[str, dict], optional
        Per-column bounds. Recognised keys per column:
        ``min``, ``max`` (numeric limits), ``min_exclusive``, ``max_exclusive``
        (bools making the corresponding limit strict). A column may set either
        or both limits.
    relations : sequence[tuple[str, str]], optional
        Ordered ``(low, high)`` pairs; each must satisfy ``low <= high`` on every
        row where both are present.

    Returns
    -------
    List[Issue]
        VAL001 (bounds) and VAL002 (relations) issues.
    """
    issues: List[Issue] = []
    bounds = bounds or {}
    relations = relations or []

    for col, spec in bounds.items():
        _require_numeric(df, col, "bounds")
        s = df[col]
        present = s.notna()
        lo, hi = spec.get("min"), spec.get("max")
        lo_excl = bool(spec.get("min_exclusive", False))
        hi_excl = bool(spec.get("max_exclusive", False))
        if lo is None and hi is None:
            continue

        mask = pd.Series(False, index=s.index)
        if lo is not None:
            mask |= present & ((s <= lo) if lo_excl else (s < lo))
        if hi is not None:
            mask |= present & ((s >= hi) if hi_excl else (s > hi))

        n = int(mask.sum())
        if n == 0:
            continue
        bad = s[mask]
        issues.append(
            Issue(
                module="validity",
                code="VAL001",
                severity=WARNING,
                description=(
                    f"Column '{col}' has {n} value(s) outside its declared valid "
                    f"range (bounds min={lo}, max={hi}); observed [{bad.min()}, "
                    f"{bad.max()}]. These may be data-feed glitches or a scaling error."
                ),
                column=col,
                evidence={
                    "n_violations": n,
                    "min": lo,
                    "max": hi,
                    "min_exclusive": lo_excl,
                    "max_exclusive": hi_excl,
                    "observed_min": float(bad.min()),
                    "observed_max": float(bad.max()),
                    "check": "bounds",
                },
            )
        )

    for pair in relations:
        low, high = pair
        _require_numeric(df, low, "relations")
        _require_numeric(df, high, "relations")
        present = df[low].notna() & df[high].notna()
        viol = present & (df[low] > df[high])
        n = int(viol.sum())
        if n == 0:
            continue
        first = df.index[viol][0]
        issues.append(
            Issue(
                module="validity",
                code="VAL002",
                severity=CRITICAL,
                description=(
                    f"Ordering constraint '{low} <= {high}' is violated on {n} row(s) "
                    f"(first at {first}), e.g. a crossed book where '{low}' exceeds "
                    f"'{high}'. Inspect these timestamps for feed glitches."
                ),
                column=high,
                evidence={
                    "n_violations": n,
                    "low_col": low,
                    "high_col": high,
                    "first_violation": str(first),
                    "check": "relation",
                },
            )
        )

    return issues
