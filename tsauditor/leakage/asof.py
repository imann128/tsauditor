"""
tsauditor.leakage.asof
----------------------
As-of / point-in-time availability leakage (LEK004).

Many real-world series describe a *reference period* but only become
*available* some time later: macro releases (CPI, policy rates, unemployment),
company earnings, and news/social sentiment scores are all published after the
period they refer to. If such a value is aligned to its reference timestamp and
consumed at that timestamp, every decision made *before* the true release date
uses information that did not yet exist. That is lookahead leakage.

Unlike LEK002/LEK003, this cannot be detected from the values alone — it depends
entirely on *when each value was published*. So this check is explicit and
opt-in: the caller supplies the availability information, and the check verifies
the data respects it. It never guesses release dates.

Two ways to declare availability, per column:

- ``pd.Series`` aligned to ``df.index`` — the publish/availability timestamp of
  the value sitting at each row. This is the general, correct form and supports
  ragged real release schedules.
- ``pd.Timedelta`` — a fixed publication lag; availability = row timestamp + lag.
  A positive lag on a column still aligned to reference dates means the *whole*
  column is used early: the classic "forgot to shift the macro series" bug.

Issue codes raised
------------------
LEK004  A value is used before it was available (available_at > row timestamp).
        CRITICAL.
"""

from __future__ import annotations

from typing import Dict, List, Union

import pandas as pd

from tsauditor.report.summary import Issue, CRITICAL

AvailabilitySpec = Union[pd.Series, pd.Timedelta]


def _availability(
    spec: AvailabilitySpec, index: pd.DatetimeIndex, col: str
) -> pd.Series:
    """Resolve a column's availability spec into a per-row timestamp Series."""
    if isinstance(spec, pd.Timedelta):
        return index.to_series(index=index) + spec
    if isinstance(spec, pd.Series):
        avail = pd.to_datetime(spec.reindex(index), errors="coerce")
        if avail.notna().sum() == 0:
            raise ValueError(
                f"available_at['{col}'] is a Series but does not align to the "
                f"DataFrame's DatetimeIndex (all timestamps came out empty after "
                f"alignment). Index it by df.index."
            )
        # A tz-aware index compared against tz-naive availability (or vice
        # versa) raises a raw pandas TypeError deep inside `avail > idx` —
        # a confusing failure for what is usually a mundane mistake (the
        # index was tz_localize'd, the release-date metadata came from
        # somewhere that wasn't). Catch it here with a message that names
        # the actual mismatch instead.
        if index.tz != avail.dt.tz:
            raise ValueError(
                f"available_at['{col}'] timezone mismatch: the DataFrame index is "
                f"{'tz-aware (' + str(index.tz) + ')' if index.tz else 'tz-naive'}, "
                f"but the availability Series is "
                f"{'tz-aware (' + str(avail.dt.tz) + ')' if avail.dt.tz else 'tz-naive'}. "
                f"Localize or drop the timezone on one so both match before calling."
            )
        return avail
    raise ValueError(
        f"available_at['{col}'] must be a pandas Series (per-row publish "
        f"timestamps) or a pandas Timedelta (fixed publication lag); got "
        f"{type(spec).__name__}."
    )


def audit_asof_leakage(
    df: pd.DataFrame,
    available_at: Dict[str, AvailabilitySpec],
    min_violations: int = 1,
    **_: object,
) -> List[Issue]:
    """
    Detect as-of / point-in-time availability leakage (LEK004).

    Flags a column when one or more of its (non-missing) values occupies a row
    whose timestamp is *earlier* than when that value became available — i.e.
    the value would have been used before it existed.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a sorted ``DatetimeIndex`` (the decision/observation
        times at which each row would be used).
    available_at : dict[str, pd.Series | pd.Timedelta]
        Per-column availability. A ``Series`` gives each row's publish
        timestamp (must be indexed by ``df.index``); a ``Timedelta`` gives a
        fixed publication lag (availability = row timestamp + lag). Columns not
        present in this mapping are not checked.
    min_violations : int
        Minimum number of early rows required to raise the issue. Default 1 —
        a single confirmed lookahead is real leakage.

    Returns
    -------
    List[Issue]
        Zero or more LEK004 Issues (CRITICAL).

    Raises
    ------
    ValueError
        If ``df`` lacks a ``DatetimeIndex``; if a column named in
        ``available_at`` is not in ``df``; if a ``Series`` spec does not
        align with ``df.index`` at all; or if a ``Series`` spec's timezone
        awareness does not match ``df.index``'s (one tz-aware, the other
        tz-naive). The timezone case raises here with a message naming the
        actual mismatch, rather than as a raw ``TypeError`` from deep inside
        the ``avail > idx`` comparison.

    Notes
    -----
    tsauditor cannot know publication dates on its own; this check only verifies
    the availability you declare. Feeding it correct ``available_at`` metadata is
    the caller's responsibility.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(
            "as-of leakage check requires a DatetimeIndex; pass time_col to scan "
            "or set a DatetimeIndex before calling."
        )

    issues: List[Issue] = []
    idx = df.index.to_series(index=df.index)

    for col, spec in available_at.items():
        if col not in df.columns:
            raise ValueError(f"available_at column '{col}' not found in DataFrame.")

        avail = _availability(spec, df.index, col)
        present = df[col].notna() & avail.notna()
        early = present & (avail > idx)
        n_viol = int(early.sum())
        if n_viol < min_violations:
            continue

        lookahead = avail[early] - idx[early]
        max_days = round(lookahead.max().total_seconds() / 86400.0, 3)
        first = idx[early].min()
        issues.append(
            Issue(
                module="leakage",
                code="LEK004",
                severity=CRITICAL,
                description=(
                    f"Feature '{col}' is used before it was available: {n_viol} row(s) "
                    f"carry a value whose release time is later than the row's own "
                    f"timestamp (max look-ahead {max_days} days, first at {first}). "
                    f"Rows before each release consume future information — align the "
                    f"column to its release schedule."
                ),
                column=col,
                evidence={
                    "n_violations": n_viol,
                    "max_lookahead_days": max_days,
                    "first_violation": str(first),
                    "check": "as-of",
                },
            )
        )

    return issues
