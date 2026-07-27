"""
Panel (long-format, multi-entity) support.

Covers:
- the crash that made panel data unusable before `group_col` existed
- `group_col` partitioning and `Issue.group` tagging
- prevalence reporting, which is what makes a 500-entity scan readable
- PNL001 (ragged coverage) and PNL003 (entity too short to audit)
- PNL002 (cross-sectional lookahead), including the negative case of a
  legitimate alpha factor
- panel-aware repair, which stops one entity's values filling another's gaps
"""

import numpy as np
import pandas as pd
import pytest

import tsauditor as tsa
from tsauditor.anomaly.point import audit_point_anomalies
from tsauditor.panel import audit_cross_sectional_leakage, audit_panel_structure

DATES = pd.date_range("2024-01-01", periods=200, freq="B")
TICKERS = ["AAA", "BBB", "CCC", "DDD", "EEE"]


def _entity(ticker: str, n: int = 200, level: float = 100.0, seed: int = 0):
    """One entity's slice of a panel, with `ret` deliberately defining `direction`."""
    rng = np.random.default_rng(seed)
    price = level + np.cumsum(rng.normal(0, 1, n))
    ret = pd.Series(price).pct_change().to_numpy()
    return pd.DataFrame(
        {
            "ticker": ticker,
            "price": price,
            "ret": ret,
            "direction": (ret > 0).astype(float),
        },
        index=DATES[:n],
    )


@pytest.fixture
def panel() -> pd.DataFrame:
    """A balanced 5-entity panel, interleaved by date as real panels are."""
    parts = [_entity(t, level=100 + 50 * i, seed=i) for i, t in enumerate(TICKERS)]
    return pd.concat(parts).sort_index()


# ── Phase 0: the crash ───────────────────────────────────────────────────────


def test_point_anomalies_survives_duplicate_timestamps(panel):
    """
    Regression. `series.loc[z.idxmax()]` returns a Series rather than a scalar
    when timestamps repeat — as they always do in a panel — and `float()` on it
    raised `TypeError: cannot convert the series to <class 'float'>`. The worst
    point is now located positionally.
    """
    issues = audit_point_anomalies(panel.drop(columns="ticker"))
    assert isinstance(issues, list)
    for issue in issues:
        assert isinstance(issue.evidence["worst_value"], float)
        assert isinstance(issue.evidence["worst_timestamp"], str)


def test_scan_without_group_col_does_not_crash_on_a_panel(panel):
    """Scanning a panel un-grouped is meaningless, but must not raise."""
    report = tsa.scan(panel, target="direction", run_stationarity=False)
    assert report.metadata["rows"] == len(panel)


def test_prf004_hints_at_group_col_for_panel_shaped_duplication(panel):
    """A panel should be told to use group_col, not told its data is corrupt."""
    report = tsa.scan(panel, target="direction", run_stationarity=False)
    dupes = report.filter(code="PRF004")

    assert len(dupes) == 1
    assert dupes[0].evidence["looks_like_panel"] is True
    assert dupes[0].evidence["repeats_per_timestamp"] == [5, 5]
    assert "group_col" in dupes[0].description


def test_prf004_does_not_mislabel_a_genuine_duplicate_as_a_panel():
    """One repeated timestamp among many unique ones is a bug, not a panel."""
    idx = pd.DatetimeIndex(
        ["2024-01-01", "2024-01-02", "2024-01-02", "2024-01-03", "2024-01-04"]
    )
    df = pd.DataFrame({"x": [1.0, 2, 3, 4, 5]}, index=idx)

    report = tsa.scan(df, run_stationarity=False)
    dupes = report.filter(code="PRF004")

    assert len(dupes) == 1
    assert dupes[0].evidence["looks_like_panel"] is False
    assert "group_col" not in dupes[0].description


# ── Phase 1: group_col ───────────────────────────────────────────────────────


def test_group_col_partitions_and_tags_issues(panel):
    report = tsa.scan(
        panel, target="direction", group_col="ticker", run_stationarity=False
    )

    assert report.is_panel is True
    assert report.metadata["group_col"] == "ticker"
    assert report.metadata["n_groups"] == 5
    assert report.groups() == TICKERS

    # Every per-entity issue carries its entity; panel-level ones do not.
    for issue in report.all_issues:
        if issue.module == "panel":
            assert issue.group is None
        else:
            assert issue.group in TICKERS


def test_grouping_removes_the_spurious_duplicate_timestamp_finding(panel):
    """
    Un-grouped, every date repeats 5x and PRF004 fires. Grouped, each entity has
    a unique index and the finding correctly disappears.
    """
    ungrouped = tsa.scan(panel, run_stationarity=False)
    grouped = tsa.scan(panel, group_col="ticker", run_stationarity=False)

    assert ungrouped.filter(code="PRF004")
    assert grouped.filter(code="PRF004") == []


def test_frequency_is_reinferred_from_one_entity(panel):
    """
    The interleaved panel index has 5 rows per date, so a naive median gap
    reads as sub-daily. Per entity it is business-daily.
    """
    ungrouped = tsa.scan(panel, run_stationarity=False)
    grouped = tsa.scan(panel, group_col="ticker", run_stationarity=False)

    assert ungrouped.metadata["frequency"] == "sub-daily"
    assert grouped.metadata["frequency"] == "daily"


def test_group_column_is_not_audited_as_a_feature(panel):
    """The entity column is dropped before the detectors see it."""
    report = tsa.scan(
        panel, target="direction", group_col="ticker", run_stationarity=False
    )
    assert all(i.column != "ticker" for i in report.all_issues)


def test_leakage_is_found_in_every_entity(panel):
    """`ret` defines `direction` in all five entities, so LEK001 hits 5/5."""
    report = tsa.scan(
        panel, target="direction", group_col="ticker", run_stationarity=False
    )
    assert report.groups_affected(code="LEK001", column="ret") == TICKERS


def test_missing_group_col_raises(panel):
    with pytest.raises(ValueError, match="group_col"):
        tsa.scan(panel, group_col="not_a_column", run_stationarity=False)


def test_group_col_equal_to_target_raises(panel):
    with pytest.raises(ValueError, match="cannot also be"):
        tsa.scan(panel, target="ticker", group_col="ticker", run_stationarity=False)


def test_single_series_scan_is_unchanged():
    """
    Back-compat: a non-panel scan must be untouched by all of this — no group
    tags, not a panel, and `group` absent from the JSON payload.
    """
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"price": 100 + np.cumsum(rng.normal(0, 1, 200))}, index=DATES)
    report = tsa.scan(df, run_stationarity=False)

    assert report.is_panel is False
    assert report.groups() == []
    assert all(i.group is None for i in report.all_issues)
    assert "n_groups" not in report.metadata
    for issue in report.all_issues:
        assert "group" not in issue.to_dict()


# ── Phase 2: prevalence ──────────────────────────────────────────────────────


def test_prevalence_reports_systemic_findings_at_100_percent(panel):
    report = tsa.scan(
        panel, target="direction", group_col="ticker", run_stationarity=False
    )
    rows = report.prevalence()

    lek001 = next(r for r in rows if r["code"] == "LEK001" and r["column"] == "ret")
    assert lek001["n_groups"] == 5
    assert lek001["total_groups"] == 5
    assert lek001["pct"] == 100.0
    assert lek001["severity"] == "critical"
    assert lek001["example_groups"] == TICKERS


def test_prevalence_is_sorted_by_severity_then_reach(panel):
    report = tsa.scan(
        panel, target="direction", group_col="ticker", run_stationarity=False
    )
    rows = report.prevalence()

    severities = [r["severity"] for r in rows]
    assert severities == sorted(
        severities, key=lambda s: {"critical": 0, "warning": 1, "info": 2}[s]
    )

    warnings_only = [r["n_groups"] or 0 for r in rows if r["severity"] == "warning"]
    assert warnings_only == sorted(warnings_only, reverse=True)


def test_prevalence_works_for_a_single_series_report():
    """Non-panel reports still get a prevalence table, with group counts None."""
    rng = np.random.default_rng(3)
    values = rng.normal(100, 5, 200)
    values[50] = 900.0
    df = pd.DataFrame({"x": values}, index=DATES)

    rows = tsa.scan(df, run_stationarity=False).prevalence()
    assert rows
    assert all(r["n_groups"] is None and r["total_groups"] is None for r in rows)


def test_filter_by_group_and_column(panel):
    report = tsa.scan(
        panel, target="direction", group_col="ticker", run_stationarity=False
    )

    only_aaa = report.filter(group="AAA")
    assert only_aaa
    assert {i.group for i in only_aaa} == {"AAA"}

    ret_issues = report.filter(column="ret")
    assert {i.column for i in ret_issues} == {"ret"}


def test_to_json_includes_a_panel_block(panel, tmp_path):
    import json

    report = tsa.scan(
        panel, target="direction", group_col="ticker", run_stationarity=False
    )
    path = tmp_path / "panel.json"
    report.to_json(str(path))

    payload = json.loads(path.read_text())
    assert payload["panel"]["group_col"] == "ticker"
    assert payload["panel"]["n_groups"] == 5
    assert payload["panel"]["prevalence"]
    assert any(i.get("group") == "AAA" for i in payload["issues"])


def test_summary_runs_for_a_panel(panel, capsys):
    report = tsa.scan(
        panel, target="direction", group_col="ticker", run_stationarity=False
    )
    report.summary()
    out = capsys.readouterr().out
    assert "Entities" in out
    assert "LEK001" in out


# ── Phase 3: panel structure ─────────────────────────────────────────────────


def test_ragged_panel_raises_pnl001():
    parts = [_entity("AAA", 200), _entity("BBB", 150), _entity("CCC", 100)]
    ragged = pd.concat(parts).sort_index()

    issues = audit_panel_structure(ragged, group_col="ticker")
    pnl001 = [i for i in issues if i.code == "PNL001"]

    assert len(pnl001) == 1
    assert pnl001[0].severity == "warning"
    assert pnl001[0].evidence["min_coverage"] == 100
    assert pnl001[0].evidence["max_coverage"] == 200
    assert pnl001[0].evidence["n_complete_groups"] == 1


def test_balanced_panel_raises_no_pnl001():
    parts = [_entity(t, 200) for t in ("AAA", "BBB", "CCC")]
    balanced = pd.concat(parts).sort_index()

    issues = audit_panel_structure(balanced, group_col="ticker")
    assert [i for i in issues if i.code == "PNL001"] == []


def test_short_entity_raises_pnl003():
    parts = [_entity("AAA", 200), _entity("BBB", 200), _entity("SHORT", 20)]
    df = pd.concat(parts).sort_index()

    issues = audit_panel_structure(df, group_col="ticker")
    pnl003 = [i for i in issues if i.code == "PNL003"]

    assert len(pnl003) == 1
    assert pnl003[0].severity == "info"
    assert pnl003[0].evidence["n_short_groups"] == 1
    assert pnl003[0].evidence["shortest_groups"] == [{"group": "SHORT", "rows": 20}]


def test_single_entity_is_not_a_panel():
    """One entity is just a time series — nothing panel-specific to report."""
    df = _entity("AAA", 200)
    assert audit_panel_structure(df, group_col="ticker") == []


def test_panel_checks_run_through_scan():
    parts = [_entity("AAA", 200), _entity("BBB", 150), _entity("CCC", 20)]
    df = pd.concat(parts).sort_index()

    report = tsa.scan(df, group_col="ticker", run_stationarity=False)
    codes = {i.code for i in report.filter(module="panel")}
    assert codes == {"PNL001", "PNL003"}


def test_panel_issues_have_suggestions():
    parts = [_entity("AAA", 200), _entity("BBB", 150), _entity("CCC", 20)]
    df = pd.concat(parts).sort_index()

    for issue in tsa.scan(df, group_col="ticker", run_stationarity=False).filter(
        module="panel"
    ):
        # Not the generic fallback — each panel code has its own template.
        assert "Review this issue" not in issue.suggestion
        assert len(issue.suggestion) > 40


def test_audit_panel_structure_validates_input():
    df = _entity("AAA", 50)

    with pytest.raises(ValueError, match="group_col"):
        audit_panel_structure(df, group_col="nope")

    with pytest.raises(ValueError, match="DatetimeIndex"):
        audit_panel_structure(df.reset_index(drop=True), group_col="ticker")


def test_empty_panel_returns_no_issues():
    empty = pd.DataFrame({"ticker": [], "price": []}, index=pd.DatetimeIndex([]))
    assert audit_panel_structure(empty, group_col="ticker") == []


# ── PNL002: cross-sectional lookahead ────────────────────────────────────────


def _factor_panel(
    common_ratio: float,
    n_entities: int = 40,
    n_periods: int = 400,
    seed: int = 11,
) -> pd.DataFrame:
    """
    A panel whose returns are a common market factor (with time-varying
    volatility) plus an idiosyncratic component.

    ``xs_rank`` is the legitimate same-timestamp cross-sectional rank.
    ``leak`` is that same rank pulled back one period.

    ``common_ratio`` scales the common factor. As it grows, relative position
    decouples from each entity's own absolute return, which is what destroys the
    per-entity checks' ability to see the leak.
    """
    rng = np.random.default_rng(seed)
    tickers = [f"E{i:02d}" for i in range(n_entities)]
    dates = pd.date_range("2020-01-01", periods=n_periods, freq="B")

    vol = np.exp(rng.normal(0, 1.0, (n_periods, 1)))
    market = rng.normal(0, 0.004, (n_periods, 1)) * vol * common_ratio
    idio = rng.normal(0, 0.004, (n_periods, n_entities))
    returns = pd.DataFrame(market + idio, index=dates, columns=tickers)

    long = returns.stack().rename("ret").reset_index()
    long.columns = ["date", "ticker", "ret"]
    long["target"] = long["ret"]
    long["xs_rank"] = long.groupby("date")["ret"].rank(pct=True)
    long["leak"] = long.groupby("ticker")["xs_rank"].shift(-1)

    return long.set_index("date").sort_index().dropna()


@pytest.mark.parametrize("common_ratio", [0, 1, 5, 25, 100])
def test_pnl002_detects_the_leak_at_every_common_factor_ratio(common_ratio):
    """
    The reason PNL002 exists. LEK002/LEK003 detection of this same leak falls
    from 100% of entities to ~22% as the common factor grows (measured in
    docs/proposals/pnl002-cross-sectional-leakage.md); the cross-sectional
    signal is invariant to it.
    """
    panel = _factor_panel(common_ratio)
    issues = audit_cross_sectional_leakage(panel, group_col="ticker", target="target")

    assert [i.column for i in issues] == ["leak"]
    assert issues[0].code == "PNL002"
    assert issues[0].severity == "warning"
    assert issues[0].evidence["lag"] == 1


def test_pnl002_ignores_the_legitimate_cross_sectional_feature():
    """
    `xs_rank` is computed from the cross-section at its own timestamp. It must
    never be flagged, at any ratio.
    """
    for common_ratio in (0, 5, 25, 100):
        panel = _factor_panel(common_ratio)
        flagged = {
            i.column
            for i in audit_cross_sectional_leakage(
                panel, group_col="ticker", target="target"
            )
        }
        assert "xs_rank" not in flagged


@pytest.mark.parametrize("true_ic", [0.02, 0.05, 0.08, 0.15])
def test_pnl002_does_not_flag_a_realistic_alpha_factor(true_ic):
    """
    A genuine cross-sectional factor produces the same *shape* of signal as a
    leak, separated only by magnitude. Real rank-ICs are 0.02-0.08; the
    thresholds must leave those alone or the check is unusable in production.
    """
    rng = np.random.default_rng(5)
    n_entities, n_periods = 40, 500
    tickers = [f"E{i:02d}" for i in range(n_entities)]
    dates = pd.date_range("2020-01-01", periods=n_periods, freq="B")

    signal = rng.normal(0, 1, (n_periods, n_entities))
    noise = rng.normal(0, 1, (n_periods, n_entities))
    future = true_ic * signal + np.sqrt(max(1 - true_ic**2, 0.0)) * noise

    returns = pd.DataFrame(
        np.vstack([rng.normal(0, 1, (1, n_entities)), future[:-1]]),
        index=dates,
        columns=tickers,
    )
    factor = pd.DataFrame(signal, index=dates, columns=tickers)

    long = returns.stack().rename("target").reset_index()
    long.columns = ["date", "ticker", "target"]
    long["factor"] = factor.stack().to_numpy()
    panel = long.set_index("date").sort_index()

    issues = audit_cross_sectional_leakage(panel, group_col="ticker", target="target")
    assert [i for i in issues if i.column == "factor"] == []


def test_pnl002_evidence_is_complete():
    panel = _factor_panel(25)
    issue = audit_cross_sectional_leakage(panel, group_col="ticker", target="target")[0]

    for key in (
        "metric",
        "lag",
        "observed_cs_corr",
        "expected_from_cs_persistence",
        "excess",
        "excess_threshold",
        "contemporaneous_cs_corr",
        "n_entities",
    ):
        assert key in issue.evidence, key
    assert issue.evidence["metric"] == "cross_sectional_spearman"
    assert issue.evidence["excess"] >= issue.evidence["excess_threshold"]


def test_pnl002_skips_panels_with_too_few_entities():
    """A cross-sectional correlation over a handful of entities is noise."""
    panel = _factor_panel(25, n_entities=5)
    assert (
        audit_cross_sectional_leakage(panel, group_col="ticker", target="target") == []
    )


def test_pnl002_skips_panels_with_too_few_timestamps():
    panel = _factor_panel(25, n_periods=20)
    assert (
        audit_cross_sectional_leakage(panel, group_col="ticker", target="target") == []
    )


def test_pnl002_validates_input():
    panel = _factor_panel(5, n_entities=25, n_periods=60)

    with pytest.raises(ValueError, match="group_col"):
        audit_cross_sectional_leakage(panel, group_col="nope", target="target")
    with pytest.raises(ValueError, match="target"):
        audit_cross_sectional_leakage(panel, group_col="ticker", target="nope")
    with pytest.raises(ValueError, match="DatetimeIndex"):
        audit_cross_sectional_leakage(
            panel.reset_index(drop=True), group_col="ticker", target="target"
        )


def test_pnl002_runs_through_scan():
    panel = _factor_panel(25)
    report = tsa.scan(
        panel, target="target", group_col="ticker", run_stationarity=False
    )

    found = report.filter(code="PNL002")
    assert [i.column for i in found] == ["leak"]
    assert found[0].group is None  # panel-level, not per-entity
    assert "Review this issue" not in found[0].suggestion


def test_pnl002_needs_a_target():
    panel = _factor_panel(25)
    report = tsa.scan(panel, group_col="ticker", run_stationarity=False)
    assert report.filter(code="PNL002") == []


def test_apply_fixes_does_not_leak_values_across_entities():
    """
    Regression, and the reason apply_fixes had to become panel-aware. On an
    interleaved panel, interpolation used to fill one entity's gap with another
    entity's values: a gap in a series sitting near 10 was filled with ~1000.
    """
    dates = pd.date_range("2024-01-01", periods=80, freq="D")
    rng = np.random.default_rng(0)

    low = 10 + rng.normal(0, 0.2, 80)
    low[40:46] = np.nan  # the gap is only in the low-level entity
    high = 1000 + rng.normal(0, 0.2, 80)

    panel = pd.concat(
        [
            pd.DataFrame({"ticker": "LOW", "price": low}, index=dates),
            pd.DataFrame({"ticker": "HIGH", "price": high}, index=dates),
        ]
    ).sort_index()

    report = tsa.scan(panel, group_col="ticker", run_stationarity=False)
    clean = report.apply_fixes(panel)

    filled = clean[clean["ticker"] == "LOW"]["price"].iloc[40:46].to_numpy()
    assert not np.isnan(filled).any()
    assert filled.max() < 50, "values bled in from the other entity"
    assert np.allclose(filled, 10, atol=2)


def test_apply_fixes_preserves_panel_shape_and_order():
    dates = pd.date_range("2024-01-01", periods=80, freq="D")
    rng = np.random.default_rng(1)
    frames = []
    for i, ticker in enumerate(["AAA", "BBB", "CCC"]):
        values = 100 * (i + 1) + rng.normal(0, 1, 80)
        values[20:25] = np.nan
        frames.append(pd.DataFrame({"ticker": ticker, "price": values}, index=dates))
    panel = pd.concat(frames).sort_index()

    report = tsa.scan(panel, group_col="ticker", run_stationarity=False)
    clean = report.apply_fixes(panel)

    assert clean.shape == panel.shape
    assert clean.index.equals(panel.index)
    assert clean["ticker"].equals(panel["ticker"])
    # Each entity stays near its own level — no cross-contamination anywhere.
    for i, ticker in enumerate(["AAA", "BBB", "CCC"]):
        vals = clean[clean["ticker"] == ticker]["price"]
        assert abs(vals.mean() - 100 * (i + 1)) < 10, ticker


def test_apply_fixes_never_mutates_the_panel():
    dates = pd.date_range("2024-01-01", periods=80, freq="D")
    rng = np.random.default_rng(2)
    values = 10 + rng.normal(0, 0.5, 80)
    values[30:36] = np.nan
    panel = pd.concat(
        [
            pd.DataFrame({"ticker": "AAA", "price": values}, index=dates),
            pd.DataFrame(
                {"ticker": "BBB", "price": 20 + rng.normal(0, 0.5, 80)}, index=dates
            ),
        ]
    ).sort_index()
    before = panel.copy(deep=True)

    report = tsa.scan(panel, group_col="ticker", run_stationarity=False)
    report.apply_fixes(panel)

    pd.testing.assert_frame_equal(panel, before)


def test_panel_change_log_is_tagged_by_entity():
    dates = pd.date_range("2024-01-01", periods=80, freq="D")
    rng = np.random.default_rng(3)
    frames = []
    for ticker in ("AAA", "BBB"):
        values = 10 + rng.normal(0, 0.5, 80)
        values[30:36] = np.nan
        frames.append(pd.DataFrame({"ticker": ticker, "price": values}, index=dates))
    panel = pd.concat(frames).sort_index()

    report = tsa.scan(panel, group_col="ticker", run_stationarity=False)
    report.apply_fixes(panel)

    assert report.last_fixes
    assert {entry["group"] for entry in report.last_fixes} <= {"AAA", "BBB"}
    assert all("group" in entry for entry in report.last_fixes)


def test_panel_leakage_drop_removes_the_column_once():
    """
    A leaky column either exists in the feature matrix or it does not — the drop
    is frame-wide, not per entity, and must not appear once per entity in the log.
    """
    panel = _factor_panel(0, n_entities=25, n_periods=120)

    report = tsa.scan(
        panel, target="target", group_col="ticker", run_stationarity=False
    )
    assert report.leaky_columns()

    clean = report.apply_fixes(panel, leakage="drop")
    drops = [e for e in report.last_fixes if e["action"] == "drop_column"]

    for col in {e["column"] for e in drops}:
        assert col not in clean.columns
    assert len(drops) == len({e["column"] for e in drops})  # one entry per column


def test_single_series_repair_is_unaffected():
    """Back-compat: a non-panel report must still take the original path."""
    rng = np.random.default_rng(4)
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    values = 100 + np.cumsum(rng.normal(0, 1, 100))
    values[40:46] = np.nan
    df = pd.DataFrame({"price": values}, index=dates)

    report = tsa.scan(df, run_stationarity=False)
    clean = report.apply_fixes(df)

    assert clean["price"].isna().sum() == 0
    assert all("group" not in entry for entry in report.last_fixes)


def test_pnl002_is_fast_enough_for_a_realistic_panel():
    """
    The first implementation looped per timestamp and took 4s for 40 entities,
    which would have made it unusable. The vectorised version must stay fast.
    """
    import time

    panel = _factor_panel(25, n_entities=100, n_periods=800)
    start = time.time()
    audit_cross_sectional_leakage(panel, group_col="ticker", target="target")
    assert time.time() - start < 5.0
