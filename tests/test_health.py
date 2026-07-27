import json

import numpy as np
import pandas as pd

from tsauditor.report.summary import GuardReport, Issue, WARNING, CRITICAL, INFO


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


def test_to_json_backward_compatible_without_df(tmp_path):
    rep = GuardReport(warnings=[Issue("leakage", "LEK001", CRITICAL, "eq", "x")])
    p = tmp_path / "r.json"
    rep.to_json(str(p))  # no df -> no health block, must still work
    data = json.loads(p.read_text())
    assert "health" not in data
    assert data["issues"][0]["code"] == "LEK001"
    assert data["leaky_columns"] == ["x"]


def test_to_dict_counts_include_info_and_match_json(tmp_path):
    """to_dict()["counts"] must include info and agree with the to_json() payload."""
    rep = GuardReport(
        critical=[Issue("leakage", "LEK001", CRITICAL, "eq", "x")],
        warnings=[Issue("profiler", "PRF002", WARNING, "clustered", "a")],
        info=[Issue("profiler", "PRF010", INFO, "note", "b")],
    )
    p = tmp_path / "r.json"
    rep.to_json(str(p))
    payload = json.loads(p.read_text())
    counts = rep.to_dict()["counts"]
    assert counts == payload["counts"]
    assert counts == {"critical": 1, "warnings": 1, "info": 1}
