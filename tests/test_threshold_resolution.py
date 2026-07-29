"""
Threshold resolution: an explicitly passed threshold must always win over the
`domain` preset, and `domain` must only supply defaults for arguments the
caller left as None.

Regression tests. `audit_point_anomalies` previously consulted `domain` first
and silently discarded an explicit `zscore_threshold`, while `audit_missing`
and `audit_contextual_anomalies` did the opposite — three detectors, two
conventions. These tests pin the single convention across all three.

They also cover the `or`-vs-`is None` distinction: `x or default` treats a
deliberate 0 as "unset", so a caller passing 0 silently got the default.
"""

import numpy as np
import pandas as pd
import pytest

from tsauditor.anomaly.contextual import audit_contextual_anomalies
from tsauditor.anomaly.point import audit_point_anomalies
from tsauditor.profiler.missing import audit_missing


@pytest.fixture
def one_moderate_outlier() -> pd.DataFrame:
    """
    60 points of N(100, 5) with a single ~3.4-sigma outlier at position 30.

    Deliberately sits *between* the sensor (3.5) and finance (5.0) presets so
    the choice of threshold visibly changes the outcome.
    """
    rng = np.random.default_rng(42)
    values = rng.normal(100, 5, 60)
    values[30] = 118.0
    return pd.DataFrame(
        {"x": values}, index=pd.date_range("2024-01-01", periods=60, freq="D")
    )


def _zscore_count(issues):
    """Total z-score outliers reported across ANO002 issues."""
    return sum(i.evidence["zscore_outlier_count"] for i in issues if i.code == "ANO002")


# ── audit_point_anomalies ────────────────────────────────────────────────────


def test_explicit_zscore_threshold_beats_domain(one_moderate_outlier):
    """An explicit zscore_threshold must not be discarded when domain is set."""
    # domain="finance" alone would use 5.0 and find nothing.
    assert (
        _zscore_count(audit_point_anomalies(one_moderate_outlier, domain="finance"))
        == 0
    )

    # Passing 3.0 explicitly must win, even though domain would say 5.0.
    strict = audit_point_anomalies(
        one_moderate_outlier, zscore_threshold=3.0, domain="finance"
    )
    assert _zscore_count(strict) == 1

    # And the reverse: a loose explicit threshold must suppress a sensor find.
    assert (
        _zscore_count(audit_point_anomalies(one_moderate_outlier, domain="sensor")) == 1
    )
    loose = audit_point_anomalies(
        one_moderate_outlier, zscore_threshold=5.0, domain="sensor"
    )
    assert _zscore_count(loose) == 0


@pytest.mark.parametrize(
    "domain, expected_z_outliers",
    [("finance", 0), ("sensor", 1), (None, 1)],
)
def test_domain_presets_unchanged(one_moderate_outlier, domain, expected_z_outliers):
    """
    With no explicit threshold the domain presets must behave exactly as before
    (finance 5.0, sensor 3.5, None 4.0). This is the guard against the fix
    having changed default behaviour.
    """
    issues = audit_point_anomalies(one_moderate_outlier, domain=domain)
    assert _zscore_count(issues) == expected_z_outliers


def test_zero_zscore_threshold_is_honoured(one_moderate_outlier):
    """
    0.0 is falsy. It must still be treated as a real threshold, not as "unset".
    A threshold of 0 flags every point more than 0 std from the mean.
    """
    issues = audit_point_anomalies(one_moderate_outlier, zscore_threshold=0.0)
    assert _zscore_count(issues) == 60


# ── audit_contextual_anomalies ───────────────────────────────────────────────


@pytest.fixture
def short_stuck_run() -> pd.DataFrame:
    """40 noisy points with a 4-long stuck run — between the sensor (3) and
    finance/None (5) windows."""
    rng = np.random.default_rng(7)
    values = list(rng.normal(20, 1, 40))
    values[10:14] = [20.0] * 4
    return pd.DataFrame(
        {"r": values}, index=pd.date_range("2024-01-01", periods=40, freq="D")
    )


def _stuck(issues):
    return [i for i in issues if i.code == "ANO001"]


def test_explicit_stuck_window_beats_domain(short_stuck_run):
    """A 4-long run: flagged by sensor (window 3), not by finance (window 5)."""
    assert (
        len(_stuck(audit_contextual_anomalies(short_stuck_run, domain="sensor"))) == 1
    )
    assert (
        len(_stuck(audit_contextual_anomalies(short_stuck_run, domain="finance"))) == 0
    )

    # Explicit window must override the domain in both directions.
    tight = audit_contextual_anomalies(
        short_stuck_run, stuck_window=3, domain="finance"
    )
    assert len(_stuck(tight)) == 1

    loose = audit_contextual_anomalies(short_stuck_run, stuck_window=5, domain="sensor")
    assert len(_stuck(loose)) == 0


def test_default_stuck_window_is_five():
    """
    Pins the undocumented-by-test default of 5 for domain=None. A run of
    exactly 4 must not fire; 6 must.

    Mutation-checked: lowering the default to 3 left every existing test in
    this file passing, because the only fixture exercising the default
    (short_stuck_run) has a 4-long run, which is under both 3 and 5.
    """
    rng = np.random.default_rng(9)

    def _with_run(run_len):
        values = list(rng.normal(20, 1, 40))
        values[10 : 10 + run_len] = [7.0] * run_len
        return pd.DataFrame(
            {"x": values}, index=pd.date_range("2024-01-01", periods=40, freq="D")
        )

    assert audit_contextual_anomalies(_with_run(4)) == []
    flagged = audit_contextual_anomalies(_with_run(6))
    assert len(flagged) == 1
    assert flagged[0].code == "ANO001"


def test_sensor_spike_threshold_is_three():
    """
    Pins the sensor spike_threshold default of 3.0. A local z-score of ~3.4
    sits between the sensor (3.0) and finance/None (3.5/4.0) defaults, so it
    must be flagged under domain="sensor" and distinguishes 3.0 from anything
    higher.

    Mutation-checked: raising the sensor default to 5.0 left every existing
    test in this file passing.
    """
    rng = np.random.default_rng(9)
    values = rng.normal(50, 2, 60)
    values[30] = 50 + 3.4 * 2  # local z roughly 3.4
    df = pd.DataFrame(
        {"x": values}, index=pd.date_range("2024-01-01", periods=60, freq="D")
    )
    flagged = [
        i for i in audit_contextual_anomalies(df, domain="sensor") if i.code == "ANO003"
    ]
    assert len(flagged) == 1


def test_zero_stuck_window_is_honoured(short_stuck_run):
    """
    stuck_window=0 is falsy but meaningful: every run longer than 0 is flagged.
    Under the old `or` idiom this silently became the domain default of 5.
    """
    issues = _stuck(audit_contextual_anomalies(short_stuck_run, stuck_window=0))
    assert len(issues) == 1
    assert issues[0].evidence["max_stuck_duration"] == 4


def test_explicit_spike_window_is_honoured():
    """
    spike_window must reach the detector rather than falling back to 21.

    The previous version of this test asserted only that both calls returned a
    list, which would have passed with the parameter ignored entirely. It needs
    a fixture where the window size changes the verdict.

    A 10-point bump of +6 sigma sits inside 200 points of N(50, 1). With a
    narrow window the bump dominates its own local context and its edges read as
    spikes; with a window wide enough to contain the bump and a lot of
    surrounding data, the local mean absorbs it and nothing is flagged.
    """
    rng = np.random.default_rng(3)
    values = rng.normal(50, 1, 200)
    values[95:105] += 6
    df = pd.DataFrame(
        {"r": values}, index=pd.date_range("2024-01-01", periods=200, freq="D")
    )

    narrow = [
        i for i in audit_contextual_anomalies(df, spike_window=7) if i.code == "ANO003"
    ]
    wide = [
        i
        for i in audit_contextual_anomalies(df, spike_window=101)
        if i.code == "ANO003"
    ]

    assert len(narrow) == 1
    assert narrow[0].evidence["n_spikes"] == 7
    assert wide == []


# ── audit_missing (already correct; pinned so it stays that way) ─────────────


def test_explicit_cluster_threshold_beats_domain():
    """A 4-long NaN run: caught at cluster_threshold=3, not at 5."""
    values = np.arange(60.0)
    values[10:14] = np.nan
    df = pd.DataFrame(
        {"x": values}, index=pd.date_range("2024-01-01", periods=60, freq="D")
    )

    # finance preset is 5 -> a 4-long run is not a cluster
    assert [i for i in audit_missing(df, domain="finance") if i.code == "PRF002"] == []

    # explicit 3 must win over the finance preset
    explicit = [
        i
        for i in audit_missing(df, cluster_threshold=3, domain="finance")
        if i.code == "PRF002"
    ]
    assert len(explicit) == 1
    assert explicit[0].evidence["cluster_threshold"] == 3
    assert explicit[0].evidence["longest_consecutive_run"] == 4
