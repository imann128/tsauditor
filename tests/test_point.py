import pytest
import pandas as pd
import numpy as np
from tsauditor.anomaly.point import audit_point_anomalies


@pytest.fixture
def base_date_index():
    return pd.date_range("2026-01-01", periods=100, freq="D")


def test_audit_point_anomalies_cases(base_date_index):
    rng = np.random.default_rng(123)

    # 1. Clean df
    df_clean = pd.DataFrame({"val": rng.normal(0, 1, 100)}, index=base_date_index)
    assert len(audit_point_anomalies(df_clean)) == 0

    # 2. Extreme Z-score outlier
    df_z = pd.DataFrame({"val": rng.normal(0, 1, 100)}, index=base_date_index)
    df_z.iloc[0, 0] = 10.0
    issues_z = audit_point_anomalies(df_z)
    assert len(issues_z) == 1
    assert issues_z[0].evidence["zscore_outlier_count"] >= 1

    # 3. IQR outlier (but not Z-score)
    data = np.concatenate([rng.normal(0, 0.1, 97), [0.35, 0.36, 0.37]])
    df_iqr = pd.DataFrame({"val": data}, index=base_date_index)
    issues_iqr = audit_point_anomalies(df_iqr)

    assert len(issues_iqr) == 1
    assert issues_iqr[0].evidence["iqr_outlier_count"] >= 3
    assert issues_iqr[0].evidence["zscore_outlier_count"] == 0


def test_audit_point_anomalies_constant_col(base_date_index):
    df = pd.DataFrame({"val": [1.0] * 100}, index=base_date_index)
    assert len(audit_point_anomalies(df)) == 0


def test_audit_point_anomalies_invalid_index():
    df = pd.DataFrame({"a": [1, 2]}, index=[1, 2])
    with pytest.raises(ValueError, match="DataFrame index must be a pd.DatetimeIndex"):
        audit_point_anomalies(df)


def test_audit_point_anomalies_finance_threshold(base_date_index):
    rng = np.random.default_rng(123)
    data = rng.normal(0, 2, 100)
    data[0] = 4.5
    df = pd.DataFrame({"val": data}, index=base_date_index)

    # In finance (5.0), 4.5 is not an outlier
    issues = audit_point_anomalies(df, domain="finance")
    assert len(issues) == 0


def test_iqr_fence_is_one_and_a_half():
    """
    Pins the 1.5 multiplier on the IQR fence.

    Mutation-checked: widening it to 3.0 left every other test in this file
    passing, so the constant was unpinned. 1.5 is Tukey's convention and the
    docs state it; a change to it is a behaviour change and should fail here.

    The fixture places a point between the two fences: outside 1.5*IQR, inside
    3.0*IQR, and under the z-score threshold so only the IQR rule can flag it.
    """
    base = list(np.linspace(10.0, 20.0, 60))
    q25, q75 = np.percentile(base, [25, 75])
    iqr = q75 - q25
    planted = q75 + 2.0 * iqr  # beyond 1.5, inside 3.0
    values = base + [planted]
    df = pd.DataFrame(
        {"x": values}, index=pd.date_range("2024-01-01", periods=len(values), freq="D")
    )

    issues = [i for i in audit_point_anomalies(df) if i.code == "ANO002"]
    assert len(issues) == 1
    ev = issues[0].evidence
    assert ev["iqr_outlier_count"] >= 1
    assert ev["zscore_outlier_count"] == 0, "must be the IQR rule doing the work"


def test_iqr_fence_is_symmetric():
    """
    Same as test_iqr_fence_is_one_and_a_half, but for the lower fence, which
    was not covered by that test.

    Mutation-checked: widening the lower fence to 3.0*IQR left every existing
    test in this file passing, including the upper-fence test above, because
    it changes a different comparison.
    """
    base = list(np.linspace(10.0, 20.0, 60))
    q25, q75 = np.percentile(base, [25, 75])
    iqr = q75 - q25
    planted = q25 - 2.0 * iqr  # beyond 1.5 on the low side, inside 3.0
    values = base + [planted]
    df = pd.DataFrame(
        {"x": values}, index=pd.date_range("2024-01-01", periods=len(values), freq="D")
    )

    issues = [i for i in audit_point_anomalies(df) if i.code == "ANO002"]
    assert len(issues) == 1
    ev = issues[0].evidence
    assert ev["iqr_outlier_count"] >= 1
    assert ev["zscore_outlier_count"] == 0


def test_esd_diagnostic_matches_planted_count():
    """
    Pins the ESD diagnostic's own docstring claim: exact recovery at 1000
    points with outliers planted at 10 sigma, tested at 50 and 150 planted
    (the levels where the z-score rule goes blind and ESD is actually
    computed; below that the z-score rule already agrees with IQR and ESD is
    never invoked).

    Mutation-checked: _ESD_MAX_FRACTION 0.4->0.1 and _ESD_MIN_OBS 15->2 both
    left every existing test in this file passing, because nothing exercised
    the ESD path with checkable numbers before.
    """
    for n_planted in (50, 150):
        rng = np.random.default_rng(42)
        values = rng.normal(0, 1, 1000)
        idxs = rng.choice(1000, n_planted, replace=False)
        values[idxs] = 10.0
        df = pd.DataFrame(
            {"x": values}, index=pd.date_range("2024-01-01", periods=1000, freq="D")
        )
        issues = [i for i in audit_point_anomalies(df) if i.code == "ANO002"]
        assert len(issues) == 1
        ev = issues[0].evidence
        assert ev["zscore_outlier_count"] == 0, "fixture must land in the blind regime"
        assert ev["esd_outlier_count"] == n_planted
        assert ev["masking_suspected"] is True


def test_masking_suspected_ratio_boundary():
    """
    masking_suspected requires esd_outlier_count > iqr_outlier_count * 0.5.
    Pins the 0.5 specifically.

    No natural fixture found lands the ratio strictly between 0.1 and 0.5: on
    real contaminated data, ESD and IQR counts move together (both near-equal,
    or ESD near 0), so a search across contamination levels and magnitudes
    never produced a case distinguishing a 0.5 threshold from a 0.1 one. Rather
    than force brittle synthetic data, this monkeypatches `_generalized_esd` to
    return a fixed count against a real `iqr_outlier_count`, isolating the
    ratio comparison itself from what ESD actually computes.

    Mutation-checked: relaxing the ratio to 0.1 flips this fixture's
    masking_suspected from False to True.
    """
    import tsauditor.anomaly.point as point_module

    rng = np.random.default_rng(42)
    values = rng.normal(0, 1, 1000)
    idxs = rng.choice(1000, 50, replace=False)
    values[idxs] = 10.0
    df = pd.DataFrame(
        {"x": values}, index=pd.date_range("2024-01-01", periods=1000, freq="D")
    )

    # iqr_outlier_count on this fixture is 56 (verified below); a mocked ESD
    # count of 20 gives a ratio of ~0.357: below 0.5, above 0.1.
    original = point_module._generalized_esd
    point_module._generalized_esd = lambda values, alpha=0.05: 20
    try:
        issues = [
            i for i in point_module.audit_point_anomalies(df) if i.code == "ANO002"
        ]
    finally:
        point_module._generalized_esd = original

    ev = issues[0].evidence
    assert ev["iqr_outlier_count"] == 56
    assert 0.1 < ev["esd_outlier_count"] / ev["iqr_outlier_count"] < 0.5
    assert ev["masking_suspected"] is False
