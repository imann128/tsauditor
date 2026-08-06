import numpy as np
import pandas as pd
import pytest

from tsauditor.report.summary import GuardReport, Issue, WARNING, CRITICAL
from tsauditor.anomaly.point import audit_point_anomalies
from tsauditor.anomaly.contextual import audit_contextual_anomalies
from tsauditor.profiler.missing import audit_missing
import tsauditor.anomaly.point as point
import tsauditor.anomaly.contextual as contextual
import tsauditor.remediate as remediate
import tsauditor.anomaly._common as anomaly_common


def test_detector_and_repair_share_the_same_threshold_and_mask_functions():
    """
    Structural guarantee, not just a behavioural one: point.py, contextual.py,
    and remediate.py must resolve every domain preset and compute every mask
    through the exact same function objects in tsauditor.anomaly._common, not
    independent copies. This is what makes the drift this file's other tests
    guard against (test_stuck_mask_matches_detector_evidence,
    test_zscore_threshold_matches_detector_across_domains,
    test_spike_threshold_matches_detector_across_domains) structurally
    impossible rather than merely tested-for: there is only one
    implementation left to call, so there is nothing left to keep in sync by
    hand. If this test ever fails, someone has reintroduced a hand-copied
    duplicate instead of importing from _common.
    """
    assert point.zscore_preset is anomaly_common.zscore_preset
    assert remediate.zscore_preset is anomaly_common.zscore_preset

    assert contextual.stuck_window_preset is anomaly_common.stuck_window_preset
    assert remediate.stuck_window_preset is anomaly_common.stuck_window_preset

    assert contextual.spike_threshold_preset is anomaly_common.spike_threshold_preset
    assert remediate.spike_threshold_preset is anomaly_common.spike_threshold_preset

    assert contextual.stuck_run_mask is anomaly_common.stuck_run_mask
    assert remediate.stuck_run_mask is anomaly_common.stuck_run_mask

    assert contextual.SPIKE_WINDOW is anomaly_common.SPIKE_WINDOW
    assert remediate.SPIKE_WINDOW is anomaly_common.SPIKE_WINDOW


def _make_df():
    """price: outlier + missing cluster; ramp: clean; stuck: a stuck run."""
    idx = pd.date_range("2020-01-01", periods=200, freq="D")
    rng = np.random.default_rng(0)
    price = rng.normal(50, 1, 200)
    price[100] = 500.0
    df = pd.DataFrame(
        {
            "price": price,
            "ramp": np.linspace(0, 10, 200),  # provably unflagged
            "stuck": rng.normal(20, 1, 200),
        },
        index=idx,
    )
    df.iloc[40:50, df.columns.get_loc("price")] = np.nan  # missing cluster
    df.iloc[60:69, df.columns.get_loc("stuck")] = 20.0  # 9-long stuck run
    return df


def _report(df, domain=None, extra=None):
    issues = (
        audit_point_anomalies(df, domain=domain)
        + audit_contextual_anomalies(df, domain=domain)
        + audit_missing(df, domain=domain)
        + (extra or [])
    )
    critical = [i for i in issues if i.severity == CRITICAL]
    warnings = [i for i in issues if i.severity != CRITICAL]
    return GuardReport(
        critical=critical, warnings=warnings, metadata={"domain": domain}
    )


# ── Non-destructiveness (the hard requirement) ────────────────────────────────


def test_original_dataframe_untouched():
    df = _make_df()
    snapshot = df.copy(deep=True)
    report = _report(df)
    report.apply_fixes(df, outliers="clip", missing="interpolate")
    pd.testing.assert_frame_equal(df, snapshot)  # byte-for-byte unchanged


def test_returns_a_new_object():
    df = _make_df()
    out = _report(df).apply_fixes(df)
    assert out is not df


def test_unflagged_column_is_untouched():
    df = _make_df()
    out = _report(df).apply_fixes(
        df, outliers="clip", missing="interpolate", stuck="nan"
    )
    pd.testing.assert_series_equal(out["ramp"], df["ramp"])


# ── Outlier handling ──────────────────────────────────────────────────────────


def test_clip_pulls_in_the_outlier():
    df = _make_df()
    out = _report(df).apply_fixes(df, outliers="clip", missing=None, stuck=None)
    assert out["price"].iloc[100] < 100  # the 500 spike is winsorized down
    assert out["price"].max() < 100  # no extreme value remains


def test_drop_is_an_alias_for_nan_and_never_deletes_rows():
    df = _make_df()
    report = _report(df)
    out = report.apply_fixes(df, outliers="drop", missing=None, stuck=None)
    out_nan = report.apply_fixes(df, outliers="nan", missing=None, stuck=None)

    assert len(out) == len(df)  # rows preserved, not deleted
    assert pd.isna(out["price"].iloc[100])  # the 500 spike actually became NaN,
    # not silently left untouched by a no-op "drop"
    pd.testing.assert_frame_equal(out, out_nan)  # truly an alias, not just similar


def test_outlier_nan_count_matches_detector_evidence():
    """The fixer's NaN-out count must equal the ANO002 combined-mask count,
    so detection and repair cannot silently diverge. Point-only report so the
    contextual (ANO003) handler does not also touch the column."""
    df = _make_df()
    report = GuardReport(warnings=audit_point_anomalies(df), metadata={"domain": None})
    ev = next(
        i for i in report.all_issues if i.code == "ANO002" and i.column == "price"
    ).evidence
    combined = (
        ev["zscore_outlier_count"] + ev["iqr_outlier_count"] - ev["agreement_count"]
    )
    before = int(df["price"].isna().sum())
    out = report.apply_fixes(df, outliers="nan", missing=None, stuck=None)
    after = int(out["price"].isna().sum())
    assert after - before == combined


@pytest.mark.parametrize("domain", [None, "finance", "sensor"])
def test_zscore_threshold_matches_detector_across_domains(domain):
    """
    remediate._zscore_threshold must equal the actual per-domain z-score
    threshold audit_point_anomalies applies (finance 5.0, sensor 3.5,
    otherwise 4.0), not just for the default domain --
    test_outlier_nan_count_matches_detector_evidence above already covers
    domain=None, but nothing exercised finance/sensor specifically. Same
    duplication-drift risk as _stuck_window/_stuck_mask, just not triggered
    yet: a changed preset in point.py could silently stop matching
    remediate.py's hardcoded copy for one domain and nothing would fail.

    Data is built so the two injected points sit on opposite sides of the
    domain split: index 0 (z ~4.2) is an outlier under the default (4.0) and
    sensor (3.5) thresholds but an inlier under finance's 5.0; index 1 (z
    ~7.4) is an outlier under every domain. Neither trips the IQR fence
    (~2.5 for this distribution), so the combined-mask formula below isolates
    the z-threshold's contribution.
    """
    rng = np.random.default_rng(3)
    n = 300
    values = (rng.random(n) < 0.3).astype(float)  # bernoulli(0.3): 0/1 core
    values[0] = 2.4
    values[1] = 4.0
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    df = pd.DataFrame({"x": values}, index=idx)

    issues = audit_point_anomalies(df, domain=domain)
    ev = next((i for i in issues if i.column == "x"), None)
    expected = 0
    if ev is not None:
        e = ev.evidence
        expected = (
            e["zscore_outlier_count"] + e["iqr_outlier_count"] - e["agreement_count"]
        )

    report = GuardReport(warnings=issues, metadata={"domain": domain})
    before = int(df["x"].isna().sum())
    out = report.apply_fixes(df, outliers="nan", missing=None, stuck=None)
    after = int(out["x"].isna().sum())
    assert after - before == expected


@pytest.mark.parametrize("domain", [None, "finance", "sensor"])
def test_spike_threshold_matches_detector_across_domains(domain):
    """
    Same drift risk as the z-score test above, for the ANO003 side:
    remediate._spike_threshold has its own hardcoded finance/sensor/default
    copy of audit_contextual_anomalies' spike_threshold (4.0/3.0/3.5), and
    test_spike_nan_count_matches_detector_evidence only ever exercised the
    default domain. The moderate spike at index 150 (local z ~3.2-3.9,
    depending on domain) and the unmistakable one at index 250 are chosen so
    finance, sensor, and the default preset each flag a different number of
    points, which a drifted threshold for any one domain would get wrong.
    """
    rng = np.random.default_rng(5)
    n = 300
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    values = rng.normal(50, 1, n)
    values[150] = 50 + 3.4  # moderate local spike
    values[250] = 50 + 20  # unmistakable local spike
    df = pd.DataFrame({"x": values}, index=idx)

    issues = audit_contextual_anomalies(df, domain=domain)
    ev = next((i for i in issues if i.code == "ANO003" and i.column == "x"), None)
    expected = ev.evidence["n_spikes"] if ev is not None else 0

    report = GuardReport(warnings=issues, metadata={"domain": domain})
    out = report.apply_fixes(df, outliers="nan", missing=None, stuck=None)
    assert int(out["x"].isna().sum()) == expected


def test_stuck_mask_matches_detector_evidence():
    """
    remediate._stuck_mask must flag exactly the rows ANO001 considers part
    of the stuck run, including the single-row-gap bridge. This is the
    missing counterpart to test_outlier_nan_count_matches_detector_evidence
    above -- that one exists for ANO002/outliers; this one didn't exist for
    ANO001/stuck values, which is exactly how the two silently drifted out
    of sync once already (the bridge was added to the ANO001 detector
    without a matching update to the repair-side mask, so scan() flagged a
    gap-split run that apply_fixes() then silently failed to touch).
    """
    idx = pd.date_range("2020-01-01", periods=20, freq="D")
    tail = [1.0, 2.0, 1.0, 3.0, 1.0, 2.0, 1.0, 3.0, 1.0]  # varying: provably not stuck
    values = np.concatenate([np.full(5, 20.0), [np.nan], np.full(5, 20.0), tail])
    df = pd.DataFrame({"x": values}, index=idx)

    report = GuardReport(
        warnings=audit_contextual_anomalies(df, stuck_window=5),
        metadata={"domain": None},
    )
    ev = next(
        i for i in report.all_issues if i.code == "ANO001" and i.column == "x"
    ).evidence
    assert ev["max_stuck_duration"] == 11  # the detector sees the bridged 11-long run

    out = report.apply_fixes(df, outliers=None, missing=None, stuck="nan")
    # Every row in the 11-long bridged run (including the gap itself) must
    # be replaced -- if repair silently used the old, unbridged mask, the
    # two 5-long halves would each sit at exactly stuck_window and never
    # get touched, leaving the original 20.0 values in place.
    assert out["x"].iloc[0:11].isna().all()
    assert out["x"].iloc[11:].tolist() == tail  # untouched: never flagged


def test_repair_steps_detect_against_the_pristine_column_not_each_others_output():
    """
    Regression. apply_fixes' outlier/spike/stuck steps used to compute their
    masks from `out[col]` -- the column as left by whichever earlier step in
    this same call already touched it -- instead of `df[col]`, the original,
    pristine column the audit actually scored. For a value that is both a
    global outlier (ANO002) and part of a stuck run (ANO001), e.g. a sensor
    stuck at an extreme constant, the outlier step ran first and NaN-ed those
    cells; by the time the stuck step recomputed its own mask on the
    already-NaN'd `out[col]`, `stuck_run_mask`'s `series.notna()` guard
    excluded them, so ANO001 silently vanished from the change log even
    though the audit had genuinely raised it -- last_fixes would show only
    outliers_to_nan, never stuck_to_nan, for a column both codes fired on.

    Detecting against `df[col]` throughout fixes that: every step now finds
    exactly what its own Issue reported, regardless of what an earlier step
    in the same call already changed. Provenance is preserved via
    already_nan (cells this step's mask covers but an earlier step got to
    first) rather than by inflating cells_changed for cells that didn't
    newly change here -- so summing cells_changed across the whole log for
    a column never double-counts a cell two detectors both flagged.
    """
    rng = np.random.default_rng(1)
    idx = pd.date_range("2020-01-01", periods=200, freq="D")
    values = rng.normal(0, 1, 200)
    values[50:58] = 999.0  # stuck AND a massive global outlier: same 8 cells
    df = pd.DataFrame({"x": values}, index=idx)

    issues = audit_point_anomalies(df) + audit_contextual_anomalies(df)
    codes = {i.code for i in issues if i.column == "x"}
    assert {"ANO001", "ANO002"} <= codes  # the fixture must actually overlap

    report = GuardReport(warnings=issues, metadata={"domain": None})
    out = report.apply_fixes(df, outliers="nan", missing="interpolate", stuck="nan")

    assert not out["x"].isna().any()  # fully repaired, regardless of overlap

    by_action = {e["action"]: e for e in report.last_fixes if e["column"] == "x"}
    assert "stuck_to_nan" in by_action  # ANO001 must still appear in the log
    assert by_action["stuck_to_nan"]["cells_changed"] == 0  # outliers got there first
    assert by_action["stuck_to_nan"]["already_nan"] == 8

    # No double-counting: summing cells_changed across every non-imputation
    # action for the column must equal the true number of distinct cells any
    # detector implicated -- not more, even though two detectors share 8 of
    # them.
    from tsauditor.remediate import affected_cells

    total_distinct = affected_cells(report, df)
    total_logged = sum(
        e["cells_changed"]
        for e in report.last_fixes
        if e["column"] == "x" and not e["action"].startswith("impute_")
    )
    assert total_logged == total_distinct


# ── Missing + stuck ───────────────────────────────────────────────────────────


def test_missing_cluster_is_imputed():
    df = _make_df()
    out = _report(df).apply_fixes(df, outliers=None, missing="interpolate", stuck=None)
    assert out["price"].iloc[40:50].isna().sum() == 0


def test_stuck_run_replaced_and_filled():
    df = _make_df()
    out = _report(df).apply_fixes(df, outliers=None, missing="interpolate", stuck="nan")
    # the formerly-flat run is no longer a single repeated value
    assert out["stuck"].iloc[60:69].nunique() > 1


def test_nan_without_imputation_leaves_nans():
    df = _make_df()
    out = _report(df).apply_fixes(df, outliers="nan", missing=None, stuck=None)
    assert out["price"].isna().sum() > df["price"].isna().sum()


# ── Leakage (opt-in) ──────────────────────────────────────────────────────────


def test_leakage_drop_is_optional():
    df = _make_df()
    df["leak"] = np.linspace(0, 1, len(df))  # ramp: no other issues to muddy the test
    extra = [Issue("leakage", "LEK001", CRITICAL, "equivalent", "leak", {})]
    report = _report(df, extra=extra)
    assert "leak" in report.apply_fixes(df, leakage=None).columns  # default keeps
    assert "leak" not in report.apply_fixes(df, leakage="drop").columns  # opt-in drops


# ── Bookkeeping & validation ──────────────────────────────────────────────────


def test_fix_log_is_recorded():
    df = _make_df()
    report = _report(df)
    report.apply_fixes(df, outliers="clip", missing="interpolate")
    actions = {entry["action"] for entry in report.last_fixes}
    assert any(a.startswith("clip") for a in actions)
    assert any(a.startswith("impute") for a in actions)


def test_invalid_option_raises():
    df = _make_df()
    with pytest.raises(ValueError):
        _report(df).apply_fixes(df, outliers="explode")


def test_target_column_is_never_fixed():
    """The label column is excluded from every repair. A binary target trips
    ANO001 (long identical runs); interpolating a label into fractions is wrong."""
    idx = pd.date_range("2020-01-01", periods=200, freq="D")
    price = np.random.default_rng(0).normal(50, 1, 200)
    price[100] = 500.0  # outlier -> ANO002 on the feature
    direction = np.array(
        ([0] * 10 + [1] * 10) * 10, dtype=float
    )  # binary runs -> ANO001
    df = pd.DataFrame({"price": price, "Direction": direction}, index=idx)
    report = GuardReport(
        warnings=[
            Issue("anomaly", "ANO001", WARNING, "stuck", "Direction"),
            Issue("anomaly", "ANO002", WARNING, "outliers", "price"),
        ],
        metadata={"domain": None, "target": "Direction"},
    )
    snapshot = df["Direction"].copy()
    out = report.apply_fixes(df, outliers="clip", stuck="nan", missing="interpolate")
    assert all(e["column"] != "Direction" for e in report.last_fixes)  # never touched
    pd.testing.assert_series_equal(out["Direction"], snapshot)  # label intact
    assert out["price"].iloc[100] < 100  # feature repaired


# ── tsa.fix() one-shot wrapper ────────────────────────────────────────────────


def test_fix_returns_clean_df_and_report():
    """fix() scans and repairs in one call, returning both the cleaned copy and
    the report so the audit trail is never discarded."""
    import tsauditor as tsa

    df = _make_df()
    snapshot = df.copy(deep=True)
    clean, report = tsa.fix(df, missing="interpolate", outliers="clip", stuck="nan")

    assert isinstance(report, GuardReport)
    assert clean is not df  # independent copy
    pd.testing.assert_frame_equal(df, snapshot)  # original untouched
    assert clean["price"].max() < 100  # outlier repaired
    assert report.last_fixes  # audit trail preserved


def test_fix_protects_the_target():
    """fix(target=...) must not repair the label column."""
    import tsauditor as tsa

    idx = pd.date_range("2020-01-01", periods=200, freq="D")
    price = np.random.default_rng(0).normal(50, 1, 200)
    price[100] = 500.0
    direction = np.array(([0] * 10 + [1] * 10) * 10, dtype=float)
    df = pd.DataFrame({"price": price, "Direction": direction}, index=idx)
    snapshot = df["Direction"].copy()

    clean, report = tsa.fix(df, target="Direction")
    pd.testing.assert_series_equal(clean["Direction"], snapshot)  # label intact
    assert all(e["column"] != "Direction" for e in report.last_fixes)


# ── available_at / constraints (#42) ───────────────────────────────────────────


def test_fix_accepts_available_at_and_runs_lek004():
    """
    Before this, fix() had no way to pass available_at, so LEK004 silently
    never ran under the one-shot wrapper, not because the data was clean, but
    because the check was never given the release-schedule metadata it needs.
    """
    import tsauditor as tsa

    idx = pd.date_range("2020-01-01", periods=60, freq="D")
    df = pd.DataFrame(
        {"cpi": np.linspace(1, 5, 60), "price": np.linspace(10, 20, 60)}, index=idx
    )

    clean, report = tsa.fix(
        df, available_at={"cpi": pd.Timedelta(days=30)}, leakage=None
    )

    assert any(i.code == "LEK004" for i in report.critical)
    assert "cpi" in report.leaky_columns()
    assert clean is not df


def test_fix_accepts_constraints_and_runs_validity():
    """Same gap for VAL001/VAL002: constraints= had no path into fix()."""
    import tsauditor as tsa

    idx = pd.date_range("2020-01-01", periods=60, freq="D")
    bid = np.full(60, 100.0)
    ask = np.full(60, 100.2)
    ask[30] = 99.0  # a crossed book: bid > ask
    df = pd.DataFrame({"bid": bid, "ask": ask}, index=idx)

    clean, report = tsa.fix(df, constraints={"relations": [("bid", "ask")]})

    assert any(i.code == "VAL002" for i in report.critical)
    assert clean is not df


def test_fix_without_available_at_or_constraints_is_unaffected():
    """Back-compat: omitting the new parameters must behave exactly as before."""
    import tsauditor as tsa

    df = _make_df()
    clean, report = tsa.fix(df, missing="interpolate", outliers="clip", stuck="nan")

    assert isinstance(report, GuardReport)
    assert clean["price"].max() < 100


# ── Contextual spikes (ANO003) folded into the outlier handler ────────────────


def _regime_df():
    """A local spike hidden from global stats: tight flat regime + one spike,
    then a global blow-out that inflates global variance so the spike is normal
    globally (ANO002 misses it) but extreme locally (ANO003 catches it)."""
    idx = pd.date_range("2020-01-01", periods=300, freq="D")
    rng = np.random.default_rng(0)
    col = rng.normal(50, 1, 300)
    col[0:40] = 100.0
    col[20] = 105.0
    col[200:] = col[200:] * 50
    return pd.DataFrame({"regime": col, "ramp": np.linspace(0, 5, 300)}, index=idx)


def test_spike_is_global_clean_but_locally_flagged():
    df = _regime_df()
    assert "regime" not in {
        i.column for i in audit_point_anomalies(df)
    }  # ANO002 misses
    assert "regime" in {
        i.column for i in audit_contextual_anomalies(df)
    }  # ANO003 catches


def test_contextual_spike_naned_and_imputed():
    df = _regime_df()
    out = _report(df).apply_fixes(df, outliers="nan", missing="interpolate", stuck=None)
    assert out["regime"].iloc[20] != 105.0  # the spike is gone
    assert out["regime"].isna().sum() == 0  # and imputed


def test_contextual_spike_clipped_to_local_band():
    df = _regime_df()
    out = _report(df).apply_fixes(df, outliers="clip", missing=None, stuck=None)
    assert out["regime"].iloc[20] < 105.0  # pulled toward the local ~100
    assert out["ramp"].equals(df["ramp"])  # unflagged column untouched


def test_spike_nan_count_matches_detector_evidence():
    """The spike NaN-out count must equal ANO003's own n_spikes."""
    df = _regime_df()
    report = GuardReport(
        warnings=audit_contextual_anomalies(df), metadata={"domain": None}
    )
    n_spikes = next(
        i for i in report.all_issues if i.code == "ANO003" and i.column == "regime"
    ).evidence["n_spikes"]
    out = report.apply_fixes(df, outliers="nan", missing=None, stuck=None)
    assert int(out["regime"].isna().sum()) == n_spikes
