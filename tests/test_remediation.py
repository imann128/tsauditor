import re

import numpy as np
import pandas as pd

from tsauditor.report.summary import Issue, GuardReport, CRITICAL, WARNING, INFO
from tsauditor.report.remediation import _REMEDIATIONS


# ── apply_fixes / detector index-order alignment (full-sweep finding) ─────────
#
# scan() validates and sorts its own working copy before running any
# detector. apply_fixes(report, df) receives a *separate* `df` argument --
# whatever the caller passed to fix()/apply_fixes() -- completely
# independent of the sorted frame scan() actually saw. Before this was
# fixed, if that caller-supplied `df` had a valid but out-of-order
# DatetimeIndex, every repair mask (stuck_run_mask's consecutive-run walk,
# spike_bounds' rolling window) was computed directly on the unsorted
# order and could find nothing at all -- even for a column the report says
# was flagged. The result was a "repaired" DataFrame that silently still
# contained the original, unfixed anomaly: worse than a missed detection,
# because report.last_fixes and the caller both believe the data was
# cleaned.


def test_apply_fixes_repairs_a_stuck_run_even_when_input_rows_are_shuffled():
    """
    Regression. Same underlying bug class as the detector-side fixes in
    contextual.py et al., but on the repair side: a stuck run that scan()
    correctly finds (thanks to the detector now sorting internally) must
    still actually get repaired when apply_fixes is handed the caller's
    original, unsorted DataFrame -- not silently skipped.
    """
    import tsauditor as tsa

    dates = pd.date_range("2024-01-01", periods=200, freq="D")
    rng = np.random.default_rng(0)
    vals = rng.normal(0, 1, 200)
    vals[50:58] = 5.0  # 8-point stuck run, in chronological order
    df_sorted = pd.DataFrame({"x": vals}, index=dates)
    df_shuffled = df_sorted.sample(frac=1.0, random_state=3)

    report = tsa.scan(df_shuffled, run_leakage=False, run_stationarity=False)
    assert any(i.code == "ANO001" for i in report.all_issues)

    clean = report.apply_fixes(df_shuffled, outliers=None, missing=None, stuck="nan")
    assert clean["x"].isna().sum() == 8
    assert set(clean.index[clean["x"].isna()]) == set(dates[50:58])


def test_apply_fixes_preserves_the_callers_original_row_order():
    """
    Regression. apply_fixes must internally sort to compute correct masks,
    but the returned DataFrame's row order must still match the caller's
    input exactly -- both because callers may rely on positional alignment
    with their own df, and because _apply_fixes_by_group writes each
    entity's repaired result back by raw position and requires the row
    order it gets back to match what it passed in.
    """
    import tsauditor as tsa

    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    rng = np.random.default_rng(1)
    vals = rng.normal(0, 1, 100)
    df_shuffled = pd.DataFrame({"x": vals}, index=dates).sample(
        frac=1.0, random_state=5
    )

    report = tsa.scan(df_shuffled, run_leakage=False, run_stationarity=False)
    clean = report.apply_fixes(df_shuffled)
    assert clean.index.equals(df_shuffled.index)


def test_fix_accepts_group_col():
    """
    Regression. fix() -- the one-shot scan-and-repair convenience wrapper --
    had no group_col parameter at all, even though scan(), apply_fixes(),
    health_score(), to_json() and to_pdf() are all panel-aware. Calling
    tsa.fix(panel_df, group_col=...) raised TypeError, forcing panel users
    to always fall back to the two-call scan() + apply_fixes() form.
    """
    import tsauditor as tsa

    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    panel = pd.concat(
        [
            pd.DataFrame({"ticker": "AAA", "x": np.arange(60.0)}, index=dates),
            pd.DataFrame({"ticker": "BBB", "x": np.arange(60.0) * 2}, index=dates),
        ]
    ).sort_index()

    clean, report = tsa.fix(panel, group_col="ticker")
    assert report.is_panel
    assert len(clean) == len(panel)


def test_apply_fixes_resolves_time_col_from_report_metadata():
    """
    Regression. apply_fixes(report, df) receives whatever `df` the caller
    passed to fix()/apply_fixes() -- if that call used
    scan(df, time_col=...), the df it's given still has time_col as a plain
    column and a meaningless RangeIndex, not the DatetimeIndex scan()
    resolved internally. Every mask this function computes is positional
    (stuck_run_mask's consecutive-run walk), so on a DataFrame whose rows
    are out of chronological order by time_col (but happen to already carry
    a RangeIndex, not a DatetimeIndex), the existing DatetimeIndex
    sort-safety never engaged at all -- the same "found it, repaired zero
    cells" failure as an out-of-order DatetimeIndex, reached through
    time_col instead.
    """
    import tsauditor as tsa

    n = 200
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    rng = np.random.default_rng(0)
    vals = rng.normal(0, 1, n)
    vals[50:58] = 5.0  # 8-point stuck run, chronological
    df_sorted = pd.DataFrame({"date": dates, "x": vals})
    df_shuffled = df_sorted.sample(frac=1.0, random_state=3).reset_index(drop=True)

    clean, report = tsa.fix(
        df_shuffled, time_col="date", missing=None, outliers=None, stuck="nan"
    )
    assert any(i.code == "ANO001" for i in report.all_issues)
    assert clean["x"].isna().sum() == 8
    # Shape and row order restored to the caller's original.
    assert list(clean.columns) == ["date", "x"]
    assert clean["date"].tolist() == df_shuffled["date"].tolist()


def test_apply_fixes_panel_path_repairs_correctly_when_whole_panel_is_shuffled():
    """
    Regression, panel variant. The panel repair path writes each entity's
    result back by position (out.iloc[positions, ...] = repaired.to_numpy()),
    which depends on the recursive apply_fixes() call preserving the
    per-entity row order it was given. A fully shuffled two-entity panel
    (rows from both entities interleaved out of chronological order) must
    still repair each entity's own stuck run at the correct timestamps.
    """
    import tsauditor as tsa

    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 100)
    a[20:26] = 5.0
    b = rng.normal(0, 1, 100)
    b[40:47] = -5.0
    df = pd.concat(
        [
            pd.DataFrame({"ticker": "AAA", "x": a}, index=dates),
            pd.DataFrame({"ticker": "BBB", "x": b}, index=dates),
        ]
    ).sample(frac=1.0, random_state=7)

    report = tsa.scan(df, group_col="ticker", run_leakage=False, run_stationarity=False)
    clean = report.apply_fixes(df, outliers=None, missing=None, stuck="nan")

    assert clean.index.equals(df.index)
    assert (clean["ticker"] == df["ticker"]).all()

    aaa_nan = set(clean.index[(df["ticker"] == "AAA") & clean["x"].isna()])
    bbb_nan = set(clean.index[(df["ticker"] == "BBB") & clean["x"].isna()])
    assert aaa_nan == set(dates[20:26])
    assert bbb_nan == set(dates[40:47])


def test_issue_suggestion_mentions_column_and_action():
    i = Issue("leakage", "LEK001", CRITICAL, "eq", "ChangeP", {"separation": 1.0})
    s = i.suggestion
    assert "ChangeP" in s and ("Remove" in s or "reconstruct" in s)


def test_lek002_suggestion_fills_peak_lag():
    i = Issue("leakage", "LEK002", WARNING, "x", "leak", {"peak_lag": 1})
    assert "+1" in i.suggestion


def test_dataset_level_issue_says_dataset():
    i = Issue("profiler", "PRF001", WARNING, "irregular", None, {})
    assert "the dataset" in i.suggestion


def test_unknown_code_falls_back():
    i = Issue("x", "ZZZ999", INFO, "?", None, {})
    assert i.suggestion and "Review this issue" in i.suggestion


def test_missing_placeholder_does_not_crash():
    # LEK002's template references {peak_lag}; if evidence lacks it the
    # suggestion must still render rather than raising KeyError.
    i = Issue("leakage", "LEK002", WARNING, "x", "leak", {})
    assert isinstance(i.suggestion, str) and i.suggestion


def test_to_dict_includes_suggestion():
    d = Issue("leakage", "LEK001", CRITICAL, "eq", "ChangeP", {}).to_dict()
    assert "suggestion" in d and d["code"] == "LEK001" and d["column"] == "ChangeP"


def test_leaky_columns_lists_only_leakage_columns():
    r = GuardReport(
        critical=[Issue("leakage", "LEK001", CRITICAL, "eq", "ChangeP")],
        warnings=[
            Issue("leakage", "LEK002", WARNING, "x", "RSI"),
            Issue("profiler", "PRF001", WARNING, "gap", "Price"),
        ],
    )
    assert r.leaky_columns() == ["ChangeP", "RSI"]


def test_suggestions_structure_and_severity_order():
    r = GuardReport(
        critical=[Issue("leakage", "LEK001", CRITICAL, "eq", "ChangeP")],
        warnings=[Issue("profiler", "PRF001", WARNING, "gap", "Price")],
    )
    sg = r.suggestions()
    assert [s["severity"] for s in sg] == ["critical", "warning"]
    assert all({"code", "column", "severity", "suggestion"} <= set(s) for s in sg)


def test_empty_report_has_no_suggestions_or_leaky_columns():
    assert GuardReport().leaky_columns() == []
    assert GuardReport().suggestions() == []


# ── Evidence-key drift guard ────────────────────────────────────────────────
#
# suggest() renders each _REMEDIATIONS template via _SafeDict, which leaves an
# unresolved placeholder untouched instead of raising -- deliberately, so a
# missing key never crashes report.summary(). That safety has a cost: if a
# detector's evidence dict key is ever renamed (ANO002's evidence has already
# been reshaped once in this codebase's history) without updating the matching
# template here, nothing fails. The suggestion text just silently starts
# showing a literal "{old_key_name}" to the user instead of the real value.
#
# _KNOWN_EVIDENCE_KEYS is the evidence schema each code's detector actually
# produces today, transcribed from the detector source itself (anomaly/*.py,
# leakage/*.py, profiler/*.py, panel.py, validity.py). Keep it in sync with
# the detectors when their evidence dicts change -- that's the whole point:
# a code whose template references a key not in this set below is caught
# here, in a fast unit test, rather than discovered as broken user-facing
# text.
_KNOWN_EVIDENCE_KEYS = {
    "LEK001": {
        "metric",
        "auc",
        "separation",
        "spearman_rho",
        "threshold",
        "target_type",
        "n_obs",
    },
    "LEK002": {"peak_lag", "peak_correlation", "min_correlation", "max_lag", "metric"},
    "LEK003": {
        "lag",
        "observed_future_corr",
        "excess_over_persistence",
        "excess_threshold",
        "metric",
    },
    "LEK004": {"n_violations", "max_lookahead_days", "first_violation", "check"},
    "LEK005": {
        "metric",
        "form",
        "group",
        "group_size",
        "group_adjusted_r2",
        "best_single_adjusted_r2",
        "threshold",
        "n_obs",
    },
    "VAL001": {
        "n_violations",
        "min",
        "max",
        "min_exclusive",
        "max_exclusive",
        "observed_min",
        "observed_max",
        "check",
    },
    "VAL002": {"n_violations", "low_col", "high_col", "first_violation", "check"},
    "PNL001": {
        "n_groups",
        "n_timestamps",
        "min_coverage",
        "max_coverage",
        "n_complete_groups",
        "worst_groups",
        "group_col",
    },
    "PNL002": {
        "metric",
        "lag",
        "observed_cs_corr",
        "expected_from_cs_persistence",
        "excess",
        "excess_threshold",
        "contemporaneous_cs_corr",
        "n_entities",
        "group_col",
    },
    "PNL003": {
        "n_short_groups",
        "n_groups",
        "min_rows",
        "shortest_groups",
        "group_col",
    },
    "PNL004": {"n_null_rows", "n_total_rows", "pct_null", "group_col"},
    "ANO001": {"max_stuck_duration"},
    "ANO002": {
        "zscore_outlier_count",
        "iqr_outlier_count",
        "agreement_count",
        "esd_outlier_count",
        "masking_suspected",
        "max_zscore",
        "worst_value",
        "worst_timestamp",
    },
    "ANO003": {"n_spikes", "max_spike_zscore", "zero_variance_context"},
    "PRF001": {"gap_count", "maximum_gap_days", "locations"},
    "PRF002": {
        "missing_percentage",
        "longest_consecutive_run",
        "cluster_count",
        "first_occurrence",
        "cluster_threshold",
    },
    "PRF003": {"adf_statistic", "p_value", "n_observations", "alpha"},
    "PRF004": {
        "duplicate_count",
        "examples",
        "looks_like_panel",
        "repeats_per_timestamp",
    },
    "PRF005": {"cluster_count", "max_consecutive_gaps", "cluster_start_locations"},
    "PRF006": {"missing_count", "missing_percentage", "threshold_percentage"},
    "PRF007": {
        "non_finite_count",
        "positive_inf_count",
        "negative_inf_count",
        "non_finite_percentage",
        "n_finite_remaining",
        "below_leakage_min_obs",
        "leakage_min_obs",
        "first_occurrence",
    },
}

# suggest() always injects these itself, regardless of evidence -- a
# template may reference them even though no detector's evidence dict
# carries them.
_ALWAYS_AVAILABLE = {"target", "column"}


def test_every_template_placeholder_is_a_real_evidence_key():
    """
    Static drift guard, complementing test_remediation_matches_real_scan_evidence
    below (which only exercises the codes cheap to trigger in one scan). For
    every code with a template, every {placeholder} it references must be a
    key that code's own detector actually puts in Issue.evidence -- otherwise
    suggest() silently renders a literal, unfilled "{placeholder}" into
    user-facing text instead of raising anything a test would catch.
    """
    placeholder_re = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")

    for code, template in _REMEDIATIONS.items():
        referenced = set(placeholder_re.findall(template)) - _ALWAYS_AVAILABLE
        if not referenced:
            continue
        known = _KNOWN_EVIDENCE_KEYS.get(code)
        assert known is not None, (
            f"{code} references {referenced} but has no known-evidence-keys entry"
        )
        unknown = referenced - known
        assert not unknown, (
            f"{code}'s template references {unknown}, not in its detector's evidence"
        )


def test_remediation_matches_real_scan_evidence():
    """
    Integration counterpart to the static check above: run real detectors
    (not hand-built Issue objects) and confirm the resulting suggestions
    never contain an unfilled "{placeholder}" -- the actual, user-visible
    failure mode if a detector's evidence schema and this module's templates
    ever drift apart. Constructed to trip as many codes as practical in one
    dataset rather than one fixture per code.
    """
    import numpy as np
    import pandas as pd
    import tsauditor as tsa

    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-01-01", periods=300, freq="D")

    target = np.cumsum(rng.normal(0, 1, 300))  # non-stationary -> PRF003
    leak = target.copy()  # near-identical -> LEK001
    stuck = rng.normal(0, 1, 300)
    stuck[100:108] = 5.0  # ANO001
    stuck[150] = 999.0  # ANO002
    stuck[200] = stuck[199] + 40  # ANO003 (local spike)
    sparse = rng.normal(0, 1, 300)
    sparse[50:56] = np.nan  # PRF002 clustered missing
    sparse[:120] = np.nan  # push overall rate high -> PRF006
    infy = rng.normal(0, 1, 300)
    infy[10] = np.inf  # PRF007
    spread = np.full(300, 0.5)
    spread[20] = -1.0  # VAL001

    df = pd.DataFrame(
        {
            "target": target,
            "leak": leak,
            "stuck": stuck,
            "sparse": sparse,
            "infy": infy,
            "spread": spread,
        },
        index=idx,
    )
    # PRF004/PRF001/PRF005: a duplicate timestamp and a large gap.
    df = pd.concat([df, df.iloc[[0]]]).sort_index()  # duplicate
    gap_df = df.drop(df.index[280:290])  # a gap

    report = tsa.scan(
        gap_df,
        target="target",
        constraints={"spread": {"min": 0}},
        run_stationarity=True,
    )

    assert len(report.all_issues) > 5, (
        "fixture should trigger a meaningful spread of codes"
    )

    placeholder_re = re.compile(r"\{[a-zA-Z_][a-zA-Z0-9_]*\}")
    for issue in report.all_issues:
        leftover = placeholder_re.findall(issue.suggestion)
        assert not leftover, (
            f"{issue.code} suggestion has unfilled placeholder(s): {leftover}"
        )
