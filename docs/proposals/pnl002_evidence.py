"""
Evidence for the PNL002 proposal (docs/proposals/pnl002-cross-sectional-leakage.md).

Reproduces the table showing that LEK002/LEK003 detection of a shifted
cross-sectional feature collapses as the common (market) factor grows, while a
cross-sectional test stays at 1.0 throughout.

Run:  python docs/proposals/pnl002_evidence.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import tsauditor as tsa

N_ENTITIES = 40
N_PERIODS = 500
RATIOS = (0, 1, 5, 25, 100)
SEED = 11


def build_panel(ratio: float, seed: int = SEED) -> pd.DataFrame:
    """
    A panel whose returns are a common market factor (with time-varying
    volatility) plus an idiosyncratic component, scaled so that ``ratio`` is the
    common-to-idiosyncratic volatility ratio.

    ``xs_rank`` is the legitimate same-day cross-sectional rank; ``leak`` is that
    same rank pulled back one period — the bug this proposal is about.
    """
    rng = np.random.default_rng(seed)
    tickers = [f"E{i:02d}" for i in range(N_ENTITIES)]
    dates = pd.date_range("2020-01-01", periods=N_PERIODS, freq="B")

    vol = np.exp(rng.normal(0, 1.0, (N_PERIODS, 1)))
    market = rng.normal(0, 0.004, (N_PERIODS, 1)) * vol * ratio
    idio = rng.normal(0, 0.004, (N_PERIODS, N_ENTITIES))
    returns = pd.DataFrame(market + idio, index=dates, columns=tickers)

    long = returns.stack().rename("ret").reset_index()
    long.columns = ["date", "ticker", "ret"]
    long["target"] = long["ret"]
    long["xs_rank"] = long.groupby("date")["ret"].rank(pct=True)
    long["leak"] = long.groupby("ticker")["xs_rank"].shift(-1)

    return long.set_index("date").sort_index().dropna().drop(columns=["xs_rank"])


def within_entity_rho(panel: pd.DataFrame) -> float:
    """Mean per-entity Spearman between the leaked feature and the future target."""
    scores = [
        sub["leak"].corr(sub["target"].shift(-1), method="spearman")
        for _, sub in panel.groupby("ticker")
    ]
    return float(np.nanmean(scores))


def cross_sectional_rho(panel: pd.DataFrame, lag: int = 1) -> float:
    """Mean over timestamps of the cross-entity Spearman at the given lag."""
    wide = panel.reset_index().pivot(index="date", columns="ticker")
    feature, target = wide["leak"], wide["target"].shift(-lag)
    scores = [
        feature.loc[d].corr(target.loc[d], method="spearman")
        for d in feature.index[:-2]
    ]
    return float(np.nanmean(scores))


def detection_rates(panel: pd.DataFrame) -> dict:
    """Percentage of entities in which each LEK check flags the leaked column."""
    report = tsa.scan(
        panel, target="target", group_col="ticker", run_stationarity=False
    )
    prevalence = {(r["code"], r["column"]): r["pct"] for r in report.prevalence()}
    return {
        "LEK002": prevalence.get(("LEK002", "leak"), 0.0),
        "LEK003": prevalence.get(("LEK003", "leak"), 0.0),
    }


def main() -> None:
    header = (
        f"{'mkt/idio':>9} | {'within-entity rho':>18} | "
        f"{'LEK002':>7} {'LEK003':>7} | {'xs rho lag+1':>12}"
    )
    print(header)
    print("-" * len(header))

    for ratio in RATIOS:
        panel = build_panel(ratio)
        rates = detection_rates(panel)
        print(
            f"{ratio:>9} | {within_entity_rho(panel):>18.3f} | "
            f"{rates['LEK002']:>6.1f}% {rates['LEK003']:>6.1f}% | "
            f"{cross_sectional_rho(panel):>12.3f}"
        )

    print(
        "\nThe univariate checks degrade as the common factor grows; the "
        "cross-sectional\nsignal does not. See "
        "docs/proposals/pnl002-cross-sectional-leakage.md"
    )


if __name__ == "__main__":
    main()
