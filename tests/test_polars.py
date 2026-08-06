import numpy as np
import pandas as pd
import pytest

import tsauditor as tsa

pl = pytest.importorskip("polars")  # polars is an optional [polars] extra


def _frame():
    n = 200
    rng = np.random.default_rng(0)
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    price = 100 + np.cumsum(rng.normal(0, 1, n))
    return pl.DataFrame(
        {
            "Date": dates,
            "Price": price,
            "ChangeP": pd.Series(price).pct_change().fillna(0).to_numpy() * 100,
            "Direction": (pd.Series(price).pct_change() > 0).astype(int).to_numpy(),
        }
    )


def test_polars_scan_runs_via_time_col():
    """A polars DataFrame is accepted when its datetime column is named."""
    report = tsa.scan(_frame(), target="Direction", time_col="Date", domain="finance")
    assert hasattr(report, "all_issues")
    assert report.metadata["rows"] == 200


def test_polars_without_time_col_raises_pointing_to_issue_28():
    """polars has no index, so time_col is mandatory — and the error says so."""
    with pytest.raises(ValueError) as exc:
        tsa.scan(_frame(), target="Direction")  # no time_col
    msg = str(exc.value)
    assert "time_col" in msg
    assert "issues/28" in msg


def test_polars_and_pandas_agree():
    """The polars path and the equivalent pandas path produce the same issues."""
    pl_df = _frame()
    pd_df = pl_df.to_pandas().set_index("Date")
    r_pl = tsa.scan(pl_df, target="Direction", time_col="Date", domain="finance")
    r_pd = tsa.scan(pd_df, target="Direction", domain="finance")
    assert {(i.code, i.column) for i in r_pl.all_issues} == {
        (i.code, i.column) for i in r_pd.all_issues
    }


def test_polars_panel_scan_with_group_col():
    """
    polars input combined with group_col. polars has no index, so the frame is
    converted at the boundary and *then* partitioned by entity — a path neither
    the polars tests nor the panel tests covered on their own.
    """
    dates = pd.date_range("2024-01-01", periods=120, freq="B")
    rows = []
    for i, ticker in enumerate(["AAA", "BBB", "CCC"]):
        rng = np.random.default_rng(i)
        price = 100 + 50 * i + np.cumsum(rng.normal(0, 1, 120))
        ret = pd.Series(price).pct_change().to_numpy()
        rows.append(
            pd.DataFrame(
                {
                    "Date": dates,
                    "ticker": ticker,
                    "price": price,
                    "ret": ret,
                    "direction": (ret > 0).astype(float),
                }
            )
        )
    flat = pd.concat(rows, ignore_index=True)

    report = tsa.scan(
        pl.from_pandas(flat),
        target="direction",
        time_col="Date",
        group_col="ticker",
        run_stationarity=False,
    )

    assert report.is_panel is True
    assert report.metadata["n_groups"] == 3
    assert report.groups() == ["AAA", "BBB", "CCC"]
    assert "ret" in report.leaky_columns()


# ── fix() / apply_fixes() with polars input ────────────────────────────────
#
# polars.DataFrame has neither .copy() nor .index, so apply_fixes's very
# first operation on a polars frame raised a raw AttributeError. scan() has
# supported polars since it was added; fix()/apply_fixes() never did, and
# nothing caught it because every test above only ever exercises scan().


def test_fix_accepts_polars_input():
    """Regression. tsa.fix(polars_df, time_col=...) used to raise AttributeError
    ('DataFrame' object has no attribute 'copy'/'index') before ever reaching
    the repair logic."""
    n = 100
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    vals = np.arange(n, dtype=float)
    vals[40:50] = np.nan
    pl_df = pl.DataFrame({"date": dates, "x": vals})

    clean, report = tsa.fix(pl_df, time_col="date", missing="interpolate")
    assert isinstance(clean, pd.DataFrame)  # polars is input-only; output stays pandas
    assert clean["x"].isna().sum() == 0
    assert list(clean.columns) == ["date", "x"]


def test_apply_fixes_accepts_polars_input_directly():
    """Same regression, called via report.apply_fixes() rather than fix()."""
    n = 100
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    vals = np.arange(n, dtype=float)
    vals[40:50] = np.nan
    pl_df = pl.DataFrame({"date": dates, "x": vals})

    report = tsa.scan(pl_df, time_col="date")
    clean = report.apply_fixes(pl_df, missing="interpolate")
    assert isinstance(clean, pd.DataFrame)
    assert clean["x"].isna().sum() == 0


def test_fix_polars_input_composes_with_shuffled_rows_and_group_col():
    """
    Regression. The polars conversion must happen before -- and compose
    correctly with -- the row-order sort-safety and panel (group_col)
    repair paths, not just work in isolation on a trivial single-series
    frame.
    """
    n = 200
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    rng = np.random.default_rng(0)
    vals = rng.normal(0, 1, n)
    vals[50:58] = 5.0  # 8-point stuck run, chronological
    pd_sorted = pd.DataFrame({"date": dates, "x": vals})
    pd_shuffled = pd_sorted.sample(frac=1.0, random_state=3).reset_index(drop=True)
    pl_shuffled = pl.from_pandas(pd_shuffled)

    clean, report = tsa.fix(
        pl_shuffled, time_col="date", missing=None, outliers=None, stuck="nan"
    )
    assert clean["x"].isna().sum() == 8

    # group_col combined with polars + time_col
    dates2 = pd.date_range("2024-01-01", periods=60, freq="D")
    panel = pd.concat(
        [
            pd.DataFrame({"date": dates2, "ticker": "AAA", "x": np.arange(60.0)}),
            pd.DataFrame({"date": dates2, "ticker": "BBB", "x": np.arange(60.0) * 2}),
        ]
    )
    clean2, report2 = tsa.fix(
        pl.from_pandas(panel), time_col="date", group_col="ticker"
    )
    assert report2.is_panel
    assert len(clean2) == len(panel)
