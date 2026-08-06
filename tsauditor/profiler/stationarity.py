import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from tsauditor.report.summary import Issue, INFO
from tsauditor.utils.validation import ensure_sorted_datetime_index


def audit_stationarity(
    df: pd.DataFrame,
    alpha: float = 0.05,
    min_obs: int = 25,
    max_lag: int = None,
    domain: str = None,
) -> list:
    """
    Audits numeric columns for stationarity using the Augmented Dickey-Fuller test.
    Validates DatetimeIndex and handles NaN/Infinite values.

    ``max_lag`` caps the ADF lag search. By default (None) statsmodels chooses
    the maximum lag and searches all of them via AIC — the bulk of scan()'s
    runtime. Passing a small cap (e.g. 4) sharply reduces the number of OLS fits
    at a slight cost in test precision.
    """
    issues = []

    # 1. Validation: the ADF test fits lagged regressions on the raw row
    # sequence -- it has no notion of the DatetimeIndex's actual timestamps,
    # only row order. An out-of-order-but-valid index would silently feed
    # adfuller a scrambled series and produce a meaningless statistic with
    # no error. See ensure_sorted_datetime_index's docstring.
    df = ensure_sorted_datetime_index(df, "audit_stationarity")

    if df.empty:
        return issues

    # Identify numeric columns
    numeric_cols = df.select_dtypes(include=["number"]).columns

    for col in numeric_cols:
        series = df[col].dropna()

        # 2. Handle inf values
        series = series.replace([np.inf, -np.inf], np.nan).dropna()
        if len(series) < min_obs:
            continue

        # A constant column has no unit root to test and makes adfuller raise
        # ("Invalid input, x is constant"). It is trivially (degenerately)
        # stationary, so skip it rather than crash the whole scan.
        if series.nunique() < 2:
            continue

        # 3. Perform ADF test
        # adfuller returns: adf_stat, p_value, used_lag, n_obs, critical_values, icbest
        try:
            result = adfuller(series, maxlag=max_lag, autolag="AIC")
        except (ValueError, np.linalg.LinAlgError):
            # adfuller can still fail on near-singular or degenerate inputs; a
            # single column's numerical quirk must not abort the entire audit.
            continue
        adf_stat, p_value, _, n_obs, _, _ = result

        # 4. Check if non-stationary
        if p_value > alpha:
            issues.append(
                Issue(
                    module="profiler",
                    code="PRF003",
                    severity=INFO,
                    description=f"Column '{col}' is non-stationary (ADF p={p_value:.4f} > {alpha}). Consider differencing before modeling.",
                    column=col,
                    evidence={
                        "adf_statistic": round(float(adf_stat), 4),
                        "p_value": round(float(p_value), 4),
                        "n_observations": int(n_obs),
                        "alpha": alpha,
                    },
                )
            )

    return issues
