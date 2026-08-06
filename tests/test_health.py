import json

import numpy as np
import pandas as pd

from tsauditor.report.summary import GuardReport, Issue, WARNING, CRITICAL


def _idx(n):
    return pd.date_range("2020-01-01", periods=n, freq="D")


def test_health_score_exact_for_missing():
    n = 100
    a = np.arange(n, dtype=float)
    a[40:50] = np.nan  # 10-cell missing cluster -> re-scan flags PRF002
    df = pd.DataFrame({"a": a, "b": np.linspace(0, 5, n)}, index=_idx(n))  # 200 cells
    # health_score re-scans df, so an empty report is enough.
    assert GuardReport().health_score(df) == 95.0  # 1 - 10/200


def test_health_score_clean_is_100():
    df = pd.DataFrame({"a": np.linspace(0, 1, 50)}, index=_idx(50))
    assert GuardReport().health_score(df) == 100.0


def test_health_score_rescans_the_passed_frame():
    """The score reflects the frame passed in (a fresh scan), not the report's
    stale issues — so a repaired frame scores higher than the original."""
    import tsauditor as tsa

    n = 100
    a = np.arange(n, dtype=float)
    a[40:50] = np.nan
    df = pd.DataFrame({"a": a, "b": np.linspace(0, 5, n)}, index=_idx(n))
    report = tsa.scan(df, run_stationarity=False)
    fixed = report.apply_fixes(df, missing="interpolate")
    assert report.health_score(df) < report.health_score(fixed)
    assert report.health_score(fixed) == 100.0


def test_health_excludes_leakage():
    """A leaky column is a modeling risk, not corrupt data — score stays 100."""
    n = 200
    t = np.linspace(0, 1, n)
    df = pd.DataFrame({"target": t, "leak": t.copy()}, index=_idx(n))  # leak == target
    rep = GuardReport(metadata={"target": "target"})
    assert rep.health_score(df) == 100.0


def test_health_no_numeric_columns_is_100():
    df = pd.DataFrame({"s": ["x"] * 10}, index=_idx(10))
    assert GuardReport().health_score(df) == 100.0


def test_to_json_includes_health_block(tmp_path):
    n = 100
    a = np.arange(n, dtype=float)
    a[::10] = np.nan
    df = pd.DataFrame({"a": a, "b": np.linspace(0, 5, n)}, index=_idx(n))
    rep = GuardReport(warnings=[Issue("profiler", "PRF002", WARNING, "clustered", "a")])
    p = tmp_path / "r.json"
    rep.to_json(str(p), df=df)
    data = json.loads(p.read_text())
    assert data["health"]["score"] == 95.0
    assert data["health"]["affected_cells"] == 10
    assert data["health"]["total_cells"] == 200
    assert "leaky_columns" in data


# ── Panel awareness (full-sweep finding) ───────────────────────────────────
#
# affected_cells()/health_score() used to recompute every mask directly on
# the raw, interleaved panel `df`, with no idea `group_col` existed --
# unlike detection (scanner.py) and apply_fixes (_apply_fixes_by_group),
# both of which are correctly per-entity. A real outlier in a small-scale
# entity mixed into a global mean/std dominated by a much larger-scale
# entity can be diluted below detection and silently vanish from the score.


def test_health_score_finds_a_per_entity_outlier_invisible_globally():
    """
    Regression. AAA's own distribution is ~10; its outlier at 30 is a
    dramatic per-entity anomaly (correctly flagged by scan(group_col=...)).
    But mixed into the interleaved panel with BBB's ~1000-scale values, a
    global z-score/IQR recomputation never sees it -- affected_cells() used
    to return 0 and health_score() 100.0 even though a real, reported
    ANO002 finding for AAA was never repaired.
    """
    import tsauditor as tsa

    n = 120
    dates = _idx(n)
    rng = np.random.default_rng(0)
    aaa = rng.normal(10, 1, n)
    aaa[20] = 30.0
    bbb = rng.normal(1000, 100, n)
    panel = pd.concat(
        [
            pd.DataFrame({"ticker": "AAA", "price": aaa}, index=dates),
            pd.DataFrame({"ticker": "BBB", "price": bbb}, index=dates),
        ]
    ).sort_index()

    report = tsa.scan(
        panel, group_col="ticker", run_leakage=False, run_stationarity=False
    )
    assert any(i.code == "ANO002" and i.group == "AAA" for i in report.all_issues)

    # Deliberately leave the outlier unrepaired.
    fixed = report.apply_fixes(panel, outliers=None, missing="interpolate", stuck=None)
    # AAA and BBB share the same date index, so filter by ticker too before
    # indexing by date to avoid an ambiguous duplicate-label lookup.
    assert (
        fixed.loc[
            (fixed["ticker"] == "AAA") & (fixed.index == dates[20]), "price"
        ].iloc[0]
        == 30.0
    )

    assert report.health_score(fixed) < 100.0


def test_to_json_score_after_is_panel_aware(tmp_path):
    """
    Regression. to_json's internal `score_after` re-scan omitted
    group_col=, unlike GuardReport.health_score()'s own re-scan -- so the
    two disagreed for panel data. Both must reflect the same, correct,
    per-entity result.
    """
    import tsauditor as tsa

    n = 120
    dates = _idx(n)
    rng = np.random.default_rng(0)
    aaa = rng.normal(10, 1, n)
    aaa[20] = 30.0
    bbb = rng.normal(1000, 100, n)
    panel = pd.concat(
        [
            pd.DataFrame({"ticker": "AAA", "price": aaa}, index=dates),
            pd.DataFrame({"ticker": "BBB", "price": bbb}, index=dates),
        ]
    ).sort_index()

    report = tsa.scan(
        panel, group_col="ticker", run_leakage=False, run_stationarity=False
    )
    fixed = report.apply_fixes(panel, outliers=None, missing="interpolate", stuck=None)

    p = tmp_path / "panel_health.json"
    report.to_json(str(p), df=panel, fixed_df=fixed)
    data = json.loads(p.read_text())

    assert data["health"]["score_after"] == report.health_score(fixed)
    assert data["health"]["score_after"] < 100.0


def test_to_json_backward_compatible_without_df(tmp_path):
    rep = GuardReport(warnings=[Issue("leakage", "LEK001", CRITICAL, "eq", "x")])
    p = tmp_path / "r.json"
    rep.to_json(str(p))  # no df -> no health block, must still work
    data = json.loads(p.read_text())
    assert "health" not in data
    assert data["issues"][0]["code"] == "LEK001"
    assert data["leaky_columns"] == ["x"]
