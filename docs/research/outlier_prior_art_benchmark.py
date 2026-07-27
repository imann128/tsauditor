"""
Benchmark behind docs/research/outlier-detection-prior-art.md.

Compares tsauditor's current ANO002 rule (z-score OR IQR) against Rosner's
Generalized ESD test on planted contamination, on clean-but-awkward
distributions, and under STL detrending.

Run:  python docs/research/outlier_prior_art_benchmark.py
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.seasonal import STL

from tsauditor.anomaly.point import audit_point_anomalies

N = 1000
IDX = pd.date_range("2024-01-01", periods=N, freq="D")
SEED = 3


def generalized_esd(x, max_outliers: int, alpha: float = 0.05) -> int:
    """
    Rosner (1983) Generalized ESD. Returns the estimated number of outliers.

    Masking is avoided by recomputing the mean and standard deviation after
    removing each candidate, so contaminating points progressively stop
    inflating the scale that judges the remaining ones.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    work = x.copy()
    stats_R, criticals = [], []

    for i in range(1, max_outliers + 1):
        mu, sd = np.nanmean(work), np.nanstd(work, ddof=1)
        if sd == 0 or len(work) < 3:
            break
        deviation = np.abs(work - mu)
        j = int(np.nanargmax(deviation))
        stats_R.append(deviation[j] / sd)
        work = np.delete(work, j)

        p = 1 - alpha / (2 * (n - i + 1))
        t = stats.t.ppf(p, n - i - 1)
        criticals.append((n - i) * t / np.sqrt((n - i - 1 + t**2) * (n - i + 1)))

    k = 0
    for i in range(len(stats_R)):
        if stats_R[i] > criticals[i]:
            k = i + 1
    return k


def tsauditor_counts(values) -> tuple:
    """(zscore_outlier_count, iqr_outlier_count) from the real detector."""
    issues = audit_point_anomalies(
        pd.DataFrame({"x": values}, index=IDX[: len(values)]), zscore_threshold=5.0
    )
    if not issues:
        return 0, 0
    e = issues[0].evidence
    return e["zscore_outlier_count"], e["iqr_outlier_count"]


def contamination_sweep() -> None:
    rng = np.random.default_rng(SEED)
    base = rng.normal(0, 1, N)

    print("Contamination sweep — outliers planted at 10 sigma\n")
    print(f"{'planted':>8} {'tsa z':>7} {'tsa iqr':>8} {'ESD':>6}   verdict")
    print("-" * 50)
    for n_out in (0, 1, 5, 20, 50, 150, 300):
        v = base.copy()
        v[:n_out] = 10.0
        z, iq = tsauditor_counts(v)
        k = generalized_esd(v, max_outliers=max(10, int(0.4 * N)))
        verdict = "exact" if k == n_out else ("over" if k > n_out else "under")
        print(f"{n_out:>8} {z:>7} {iq:>8} {k:>6}   {verdict}")


def clean_data_false_positives() -> None:
    rng = np.random.default_rng(SEED)
    cases = {
        "gaussian": rng.normal(0, 1, N),
        "lognormal (skewed)": rng.lognormal(0, 1, N),
        "exponential (skewed)": rng.exponential(1, N),
        "linear trend": np.linspace(0, 100, N) + rng.normal(0, 1, N),
        "random walk": np.cumsum(rng.normal(0, 1, N)),
        "seasonal": 10 * np.sin(np.arange(N) * 2 * np.pi / 50) + rng.normal(0, 1, N),
    }

    print("\n\nClean data (0 true outliers) — false positives\n")
    print(f"{'data':26} {'ESD':>6} {'tsa iqr':>9}")
    print("-" * 44)
    for name, values in cases.items():
        _, iq = tsauditor_counts(values)
        print(f"{name:26} {generalized_esd(values, 200):>6} {iq:>9}")

    print("\n\nDoes STL detrending help? (it does not)\n")
    print(f"{'data':26} {'raw ESD':>8} {'STL-residual ESD':>18}")
    print("-" * 55)
    for name in ("linear trend", "random walk", "seasonal"):
        series = pd.Series(cases[name], index=IDX)
        resid = STL(series, period=50, robust=True).fit().resid
        print(
            f"{name:26} {generalized_esd(series.values, 200):>8} "
            f"{generalized_esd(resid.values, 200):>18}"
        )


def cost() -> None:
    rng = np.random.default_rng(SEED)
    start = time.time()
    generalized_esd(rng.normal(0, 1, 5000), 2000)
    print(f"\n\nCost: n=5000, k=2000 -> {time.time() - start:.2f}s (O(k*n))")


if __name__ == "__main__":
    contamination_sweep()
    clean_data_false_positives()
    cost()
