"""
Shared pytest fixtures for tsauditor tests.
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def clean_financial_df() -> pd.DataFrame:
    """
    A clean, well-formed daily financial DataFrame with no issues.
    Mimics OGDC-style OHLCV data with correctly constructed features.
    """
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-01", periods=500, freq="B")  # business days
    price = 100 + np.cumsum(rng.normal(0, 1, 500))

    df = pd.DataFrame(
        {
            "Price": price,
            "Open": price * rng.uniform(0.99, 1.01, 500),
            "High": price * rng.uniform(1.00, 1.02, 500),
            "Low": price * rng.uniform(0.98, 1.00, 500),
            "Volume": rng.integers(1_000_000, 10_000_000, 500).astype(float),
            "Return_lag1": pd.Series(price).pct_change().shift(1).values,
            "Direction": (pd.Series(price).pct_change() > 0).astype(int).values,
        },
        index=dates,
    )

    return df.dropna()


@pytest.fixture
def leaky_financial_df(clean_financial_df: pd.DataFrame) -> pd.DataFrame:
    """
    The clean DataFrame with a deliberately injected LEK001 leakage column.
    ChangeP is the same-day percentage change — mathematically equivalent
    to the target (Direction) signal. This replicates the OGDC ChangeP case.
    """
    df = clean_financial_df.copy()
    price = df["Price"]
    # ChangeP = same-day return — equivalent to what Direction is derived from
    df["ChangeP"] = price.pct_change() * 100
    return df.dropna()


@pytest.fixture
def sensor_df() -> pd.DataFrame:
    """
    A clean sensor-style DataFrame (sub-daily frequency).
    """
    rng = np.random.default_rng(0)
    dates = pd.date_range("2023-01-01", periods=1000, freq="h")
    df = pd.DataFrame(
        {
            "temperature": 20 + np.cumsum(rng.normal(0, 0.1, 1000)),
            "pressure": 101.3 + rng.normal(0, 0.5, 1000),
            "vibration": rng.exponential(1.0, 1000),
            "label": (rng.normal(0, 1, 1000) > 0).astype(int),
        },
        index=dates,
    )
    return df
