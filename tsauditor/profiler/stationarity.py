import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from tsauditor.report.summary import Issue, INFO


def audit_stationarity(
    df: pd.DataFrame,
    alpha: float = 0.05,
    min_obs: int = 25,
    domain: str = None,
) -> list:
    """
    Audits numeric columns for stationarity using the Augmented Dickey-Fuller test.
    Validates DatetimeIndex and handles NaN/Infinite values.
    """
    issues = []

    # 1. Validation: Ensure index is a DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a pd.DatetimeIndex")

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

        # 3. Perform ADF test
        # adfuller returns: adf_stat, p_value, used_lag, n_obs, critical_values, icbest
        result = adfuller(series, autolag="AIC")
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
