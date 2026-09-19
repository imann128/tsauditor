"""
tsauditor/benchmarks/accuracy_suite.py
---------------------------------------
Measures tsauditor's leakage-detection accuracy against real datasets whose
leaky columns are independently documented -- not synthetic data with
ground truth by construction, and not a leak we injected ourselves. Each
Case's `source` field says exactly how "known leaky" was established, so a
reader can go verify the claim rather than take this file's word for it.

This is deliberately NOT part of the pytest suite. pytest asserts specific
engineering behavior (a function returns X given Y) with a pass/fail
verdict; this measures statistical detection accuracy against a small,
curated set of real cases and is meant to be *read* -- numbers on a report,
not a pass/fail gate -- the same spirit as the LEK002 threshold table in
CHANGELOG.md (measured tradeoffs, not test assertions). Run it manually:

    python benchmarks/accuracy_suite.py

Prints a summary and writes benchmarks/results/accuracy_report.md.

Adding a case
--------------
1. Get real data with a leak someone else already identified and wrote
   up (a paper, a competition post-mortem, this project's own prior
   published analysis) -- not a leak you constructed by injecting a
   feature into otherwise-clean data. That's a different, also useful,
   kind of validation (see CONTRIBUTING.md notes on synthetic sweeps
   used elsewhere in this codebase, e.g. LEK002's threshold table), but
   it answers a different question than this file is trying to answer.
2. Append a `Case(...)` to CASES below, with `source` naming exactly
   where the ground truth comes from.
3. `known_leaky` should be the *complete* set this source names --
   an incomplete set inflates precision and deflates recall from
   comparing against columns that are quietly forgotten, not comparing
   against a real audited ground truth.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

import tsauditor as tsa

HERE = Path(__file__).resolve().parent


@dataclasses.dataclass
class Case:
    name: str
    description: str
    source: str  # exactly how "known leaky" was established -- citable
    load: Callable[[], pd.DataFrame]
    target: str
    domain: Optional[str]
    known_leaky: set  # ground-truth leaky column names, from `source`
    group_col: Optional[str] = None
    scan_kwargs: dict = dataclasses.field(default_factory=dict)
    # For an `available_at` entry that needs the loaded df's own index (a
    # per-row Series of absolute publish timestamps, not a fixed Timedelta
    # offset -- see the ALFRED case) -- called with the loaded df, merged
    # into scan_kwargs["available_at"] at evaluate() time, since Case is
    # built before load() ever runs and can't know the index yet.
    available_at_builder: Optional[Callable[[pd.DataFrame], dict]] = None


def _load_ogdc() -> pd.DataFrame:
    path = (
        HERE.parent
        / "examples"
        / "ogdc_leakage_case"
        / "ogdc_with_regimes.csv"
    )
    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    return df.dropna(subset=["Direction"])


# The vintage-release date of the "as later revised" GDP column below --
# the actual Federal Reserve/ALFRED disclosure date the data was pulled
# under, not a chosen or invented cutoff. `available_at` needs this exact
# value: an absolute point every row's leaky feature became known, which is
# what makes it a real, non-injected as-of leak rather than a synthetic one.
_ALFRED_REVISED_VINTAGE = pd.Timestamp("2024-01-01")


def _load_alfred_gdp() -> pd.DataFrame:
    """
    Real Gross Domestic Product (GDPC1), from ALFRED (Archival FRED,
    Federal Reserve Bank of St. Louis) -- downloaded as "Observations by
    Vintage Date, All Observations", quarterly, 1947-2023, across six
    vintage snapshots (1995, 2005, 2010, 2015, 2020, 2024).

    Builds a genuine point-in-time leak from two vintages of the exact
    same real quantity, not from a synthetic column and not a leak
    invented for this benchmark:

    - ``gdp_level_realtime``: GDP level as known in the 1995-01-30
      vintage -- restricted to the ~191 quarters (1947-1994) that vintage
      actually covers, so this genuinely was the available number at
      analysis time for every row it appears in.
    - ``gdp_level_asrevised2024``: the *same quarters'* GDP level, but as
      later revised in the 2024-01-01 vintage -- not knowable until
      decades after the fact for any of these rows. This is the leaky
      column.
    - ``gdp_declined``: target, a simple recession-style indicator
      (quarter-over-quarter decline), computed only from the *realtime*
      series -- sound methodology (label built from the best true
      measure; feature availability is the thing under test, not the
      label).

    GDP revisions are typically small relative to level (measured here:
    Pearson r = 0.998 between the two vintages over this window), which
    is exactly why this is a harder, more realistic leak than a
    near-duplicate column: a naive correlation-threshold check could
    plausibly wave it through as "just a slightly different measurement
    of the same thing" rather than recognizing it as temporally
    unavailable.
    """
    path = HERE / "data" / "gdpc1_vintages.csv"
    raw = pd.read_csv(path, index_col="observation_date", parse_dates=True)

    realtime = raw["GDPC1_19950101"].dropna()
    df = pd.DataFrame(index=realtime.index)
    df["gdp_level_realtime"] = realtime.to_numpy()
    df["gdp_level_asrevised2024"] = raw.loc[
        realtime.index, "GDPC1_20240101"
    ].to_numpy()
    df["gdp_declined"] = (df["gdp_level_realtime"].diff() < 0).astype(float)
    return df.dropna()


CASES = [
    Case(
        name="ogdc_changep",
        description=(
            "OGDC (Oil & Gas Development Company Limited), Pakistan Stock "
            "Exchange daily OHLCV plus engineered features. tsauditor's own "
            "original motivating case.\n\n"
            "Two different leakage *mechanisms* are mixed in this one "
            "source, deliberately not smoothed over: ChangeP and Returns "
            "are statistical target-equivalence (LEK001) -- detectable "
            "from the values alone, no extra information needed. Open, "
            "High, and Low are leaky for a different reason entirely: "
            "they are same-day quantities not known until the trading "
            "session closes, a point-in-time fact about *when* data "
            "becomes available that cannot be inferred from the values "
            "themselves -- that is exactly what LEK004 exists for, and "
            "it requires the caller to supply that fact via "
            "`available_at`. Run without it, only 2/5 are caught (LEK001 "
            "alone); with it, all 5/5. This case runs with `available_at` "
            "supplied for that reason -- omitting it would not be a "
            "harder test, it would be an unfair one: it silently omits "
            "the one piece of information LEK004's contract explicitly "
            "asks the caller to provide."
        ),
        source=(
            "examples/ogdc_leakage_case/README.md, build_ogdc_notebook.py, "
            "and compare_leakage.py (this repository): Open, High, Low, "
            "ChangeP, and Returns are all same-day quantities not "
            "available at prediction time; ChangeP and Returns "
            "additionally, mathematically define Direction's sign. "
            "Measured impact of removing all five: Random Forest accuracy "
            "99.68% -> 69.81%, Gradient Boosting 99.68% -> 73.70%."
        ),
        load=_load_ogdc,
        target="Direction",
        domain="finance",
        known_leaky={"Open", "High", "Low", "ChangeP", "Returns"},
        scan_kwargs={
            "run_stationarity": False,
            # Same-day OHLC genuinely isn't known until the session
            # closes -- a domain fact, not something injected for this
            # benchmark. See the mechanism note above: without this,
            # LEK004 has no basis to flag Open/High/Low at all, and that
            # would be tsauditor being denied the one input its own
            # documented contract asks for, not tsauditor failing.
            "available_at": {
                "Open": pd.Timedelta(days=1),
                "High": pd.Timedelta(days=1),
                "Low": pd.Timedelta(days=1),
            },
        },
    ),
    Case(
        name="ogdc_changep_statistical_only",
        description=(
            "Same OGDC data and same five-column ground truth as "
            "ogdc_changep above, but *without* `available_at` -- the "
            "baseline a caller gets if they scan without supplying "
            "point-in-time metadata (the common case: most callers won't "
            "have publish-timing metadata handy for every column). "
            "Reported side by side with the case above specifically so "
            "the LEK001-alone number (statistically detectable leaks) "
            "isn't hidden behind the LEK001+LEK004 combined number."
        ),
        source="Same as ogdc_changep.",
        load=_load_ogdc,
        target="Direction",
        domain="finance",
        known_leaky={"Open", "High", "Low", "ChangeP", "Returns"},
        scan_kwargs={"run_stationarity": False},
    ),
    Case(
        name="alfred_gdp_revision",
        description=(
            "Real US GDP (GDPC1) from ALFRED (Archival FRED, Federal "
            "Reserve Bank of St. Louis) -- the standard real-time-data "
            "tool in the macro-forecasting literature (Croushore & Stark "
            "and others) for exactly this leak: a later data revision "
            "used as though it were available at the time. Two vintages "
            "of the same 191 quarters (1947-1994): as known in the "
            "1995-01-30 vintage (legitimately available then) and as "
            "later revised in the 2024-01-01 vintage (not knowable for "
            "decades). The two are highly correlated (Pearson r = 0.998 "
            "over this window) -- GDP revisions are typically modest "
            "relative to level -- which makes this a harder, more "
            "realistic leak than a near-duplicate column: it could "
            "plausibly read as 'a slightly different measurement of the "
            "same thing' rather than a temporal-availability violation, "
            "unless the check actually reasons about *when* each column "
            "became known."
        ),
        source=(
            "ALFRED download, series GDPC1, 'Observations by Vintage "
            "Date, All Observations', vintages 1995-01-01 through "
            "2024-01-01 (benchmarks/data/gdpc1_vintages.csv). The "
            "2024-01-01 column's actual vintage-release date -- a fact "
            "from the download itself, not chosen for this benchmark -- "
            "is what `available_at` below is built from."
        ),
        load=_load_alfred_gdp,
        target="gdp_declined",
        domain="finance",
        known_leaky={"gdp_level_asrevised2024"},
        scan_kwargs={"run_stationarity": False},
        available_at_builder=lambda df: {
            "gdp_level_asrevised2024": pd.Series(
                _ALFRED_REVISED_VINTAGE, index=df.index
            ),
        },
    ),
    Case(
        name="alfred_gdp_revision_statistical_only",
        description=(
            "Same ALFRED data and same ground truth as "
            "alfred_gdp_revision above, but without `available_at` -- "
            "same before/after pairing as ogdc_changep_statistical_only, "
            "for the same reason: shows the floor a caller gets without "
            "supplying the one fact (publish timing) no statistical test "
            "can infer from values alone."
        ),
        source="Same as alfred_gdp_revision.",
        load=_load_alfred_gdp,
        target="gdp_declined",
        domain="finance",
        known_leaky={"gdp_level_asrevised2024"},
        scan_kwargs={"run_stationarity": False},
    ),
]


def evaluate(case: Case) -> dict:
    """
    Score one case: precision/recall/false-positive-rate of
    `report.leaky_columns()` against `case.known_leaky`.

    "Known clean" is every other candidate column (everything except the
    target, group_col, and the documented leaky set) -- not a separately
    curated allowlist, so a column the source's write-up simply never
    mentioned is treated as clean by default. That is the honest default,
    but it means a case whose write-up only partially audited its columns
    will understate tsauditor's real false-positive rate; `known_leaky`
    should be as complete as the source actually supports (see module
    docstring).
    """
    df = case.load()
    scan_kwargs = dict(case.scan_kwargs)
    if case.available_at_builder is not None:
        # Merge rather than overwrite: a case could in principle mix a
        # static (Timedelta-based) entry in scan_kwargs with a
        # df-index-dependent one here. No current case does both, but
        # merge is the correct general behavior either way.
        scan_kwargs["available_at"] = {
            **scan_kwargs.get("available_at", {}),
            **case.available_at_builder(df),
        }
    report = tsa.scan(
        df,
        target=case.target,
        domain=case.domain,
        group_col=case.group_col,
        **scan_kwargs,
    )
    flagged = set(report.leaky_columns())

    candidates = set(df.columns) - {case.target}
    if case.group_col:
        candidates.discard(case.group_col)
    known_clean = candidates - case.known_leaky

    true_positives = flagged & case.known_leaky
    false_positives = flagged & known_clean
    false_negatives = case.known_leaky - flagged

    precision = (
        len(true_positives) / len(flagged) if flagged else float("nan")
    )
    recall = (
        len(true_positives) / len(case.known_leaky)
        if case.known_leaky
        else float("nan")
    )
    fpr = (
        len(false_positives) / len(known_clean)
        if known_clean
        else float("nan")
    )

    return {
        "case": case.name,
        "n_rows": len(df),
        "n_known_leaky": len(case.known_leaky),
        "n_known_clean": len(known_clean),
        "true_positives": sorted(true_positives),
        "false_negatives": sorted(false_negatives),
        "false_positives": sorted(false_positives),
        "precision": precision,
        "recall": recall,
        "false_positive_rate": fpr,
    }


def _pct(x) -> str:
    return "n/a" if isinstance(x, float) and pd.isna(x) else f"{x:.1%}"


def main() -> None:
    lines = [
        "# tsauditor leakage-detection accuracy benchmark",
        "",
        "Measures `report.leaky_columns()` precision/recall against real "
        "datasets whose leaky columns are independently documented (see "
        "each case's Source, not synthetic ground truth and not a leak "
        "injected for this benchmark). Regenerate with "
        "`python benchmarks/accuracy_suite.py`.",
        "",
    ]

    for case in CASES:
        result = evaluate(case)
        print(
            f"[{case.name}] recall={_pct(result['recall'])} "
            f"precision={_pct(result['precision'])} "
            f"fpr={_pct(result['false_positive_rate'])}"
        )

        lines.append(f"## {case.name}")
        lines.append("")
        lines.append(case.description)
        lines.append("")
        lines.append(f"**Source:** {case.source}")
        lines.append("")
        lines.append(f"- Rows: {result['n_rows']}")
        lines.append(
            f"- Recall: {_pct(result['recall'])} "
            f"({len(result['true_positives'])}/{result['n_known_leaky']} "
            f"known-leaky columns caught: {result['true_positives']})"
        )
        if result["false_negatives"]:
            lines.append(f"  - **Missed:** {result['false_negatives']}")
        lines.append(f"- Precision: {_pct(result['precision'])}")
        lines.append(
            f"- False positive rate: {_pct(result['false_positive_rate'])} "
            f"(of {result['n_known_clean']} known-clean columns)"
        )
        if result["false_positives"]:
            lines.append(
                f"  - **Wrongly flagged:** {result['false_positives']}"
            )
        lines.append("")

    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "accuracy_report.md"
    out_path.write_text("\n".join(lines))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
