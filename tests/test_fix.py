import numpy as np
import pandas as pd
import pytest

from tsauditor.report.summary import GuardReport, Issue, WARNING, CRITICAL
from tsauditor.anomaly.point import audit_point_anomalies
from tsauditor.anomaly.contextual import audit_contextual_anomalies
from tsauditor.profiler.missing import audit_missing


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
