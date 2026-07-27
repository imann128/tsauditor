"""
tsauditor.panel
---------------
Structural checks that only exist for panel (long-format, multi-entity) data.

Every other module in tsauditor audits a *single* time series. When
``scan(..., group_col=...)`` is used the frame is partitioned and those modules
run once per entity, unchanged. The checks in this module are different: they
look at the panel *as a whole* and ask questions that are meaningless for a
single series.

Issue codes raised
------------------
PNL001  Ragged panel: entities do not share a common time index.  WARNING.
PNL002  Cross-sectional lookahead in a feature.                   WARNING.
PNL003  Entity too short to audit meaningfully.                   INFO.
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd

import numpy as np

from tsauditor.report.summary import Issue, WARNING, INFO

# Below this many rows an entity cannot support the library's own minimums:
# audit_equivalence and the correlation checks require min_obs=30 pairwise
# observations, and the ADF test requires 25.
_MIN_USEFUL_ROWS = 30


def audit_panel_structure(
    df: pd.DataFrame,
    group_col: str,
    min_rows: int = _MIN_USEFUL_ROWS,
    max_ragged_examples: int = 5,
) -> List[Issue]:
    """
    Check the structural integrity of a panel.

    Parameters
    ----------
    df : pd.DataFrame
        The full panel, with a DatetimeIndex and an entity column.
    group_col : str
        Name of the entity column.
    min_rows : int
        Entities with fewer rows than this raise PNL003. Default 30, matching
        the ``min_obs`` floor used by the leakage detectors.
    max_ragged_examples : int
        How many example entities to name in the PNL001 evidence.

    Returns
    -------
    List[Issue]
        Zero or more PNL001 / PNL003 issues.
    """
    issues: List[Issue] = []

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("panel checks require a DatetimeIndex")
    if group_col not in df.columns:
        raise ValueError(f"group_col '{group_col}' not found in DataFrame columns.")
    if df.empty:
        return issues

    grouped = df.groupby(group_col, sort=True)
    coverage = {str(key): sub.index for key, sub in grouped}
    if len(coverage) < 2:
        # A "panel" of one entity is just a time series; nothing panel-specific
        # to say about it.
        return issues

    all_timestamps = df.index.unique()
    n_all = len(all_timestamps)

    # ── PNL001: ragged coverage ──────────────────────────────────────────────
    counts = {key: len(idx.unique()) for key, idx in coverage.items()}
    distinct_counts = set(counts.values())

    if len(distinct_counts) > 1:
        shortfall = {
            key: n_all - n for key, n in sorted(counts.items(), key=lambda kv: kv[1])
        }
        worst = [k for k, v in shortfall.items() if v > 0][:max_ragged_examples]
        complete = [k for k, v in shortfall.items() if v == 0]

        issues.append(
            Issue(
                module="panel",
                code="PNL001",
                severity=WARNING,
                description=(
                    f"Ragged panel: entities do not share a common time index. "
                    f"The panel spans {n_all} distinct timestamps, but coverage "
                    f"per entity ranges from {min(counts.values())} to "
                    f"{max(counts.values())}. Only {len(complete)} of "
                    f"{len(counts)} entities are complete. Cross-sectional "
                    f"operations (ranks, market-wide aggregates, pivots) will "
                    f"silently compare different sets of entities at different "
                    f"timestamps."
                ),
                column=None,
                evidence={
                    "n_groups": len(counts),
                    "n_timestamps": int(n_all),
                    "min_coverage": int(min(counts.values())),
                    "max_coverage": int(max(counts.values())),
                    "n_complete_groups": len(complete),
                    "worst_groups": worst,
                    "group_col": group_col,
                },
            )
        )

    # ── PNL003: entities too short to audit ──────────────────────────────────
    short = {key: len(idx) for key, idx in coverage.items() if len(idx) < min_rows}
    if short:
        ordered = sorted(short.items(), key=lambda kv: kv[1])
        issues.append(
            Issue(
                module="panel",
                code="PNL003",
                severity=INFO,
                description=(
                    f"{len(short)} of {len(coverage)} entities have fewer than "
                    f"{min_rows} rows, which is below the minimum the leakage "
                    f"and stationarity checks need to produce a trustworthy "
                    f"score. Findings for these entities are unreliable, and "
                    f"their absence of findings is not evidence of health."
                ),
                column=None,
                evidence={
                    "n_short_groups": len(short),
                    "n_groups": len(coverage),
                    "min_rows": int(min_rows),
                    "shortest_groups": [
                        {"group": k, "rows": int(v)} for k, v in ordered[:5]
                    ],
                    "group_col": group_col,
                },
            )
        )

    return issues


# ── PNL002: cross-sectional lookahead ────────────────────────────────────────


def _cross_sectional_corr(
    feature_wide, target_wide, lag: int, min_entities: int
) -> Optional[float]:
    """
    Mean over timestamps of the Spearman correlation *across entities* between
    ``feature[:, t]`` and ``target[:, t + lag]``.

    Returns None if fewer than two timestamps had enough co-present entities.

    Vectorised. Spearman is Pearson on ranks, so both frames are masked to their
    co-present entries, ranked row-wise once, and then correlated row-wise with
    array arithmetic. Looping timestamp by timestamp and calling ``Series.corr``
    was ~80x slower and made the check unusable on a realistic panel.
    """
    shifted = target_wide.shift(-lag)

    # Mask both sides to entities present in both, so the ranks below are
    # computed over exactly the entities being compared.
    both = feature_wide.notna() & shifted.notna()
    usable_rows = both.sum(axis=1) >= min_entities
    if usable_rows.sum() < 2:
        return None

    f_masked = feature_wide.where(both).loc[usable_rows]
    y_masked = shifted.where(both).loc[usable_rows]

    # Average ranks within each row; NaNs stay NaN and are excluded below.
    f_rank = f_masked.rank(axis=1).to_numpy(dtype=float)
    y_rank = y_masked.rank(axis=1).to_numpy(dtype=float)

    valid = ~(np.isnan(f_rank) | np.isnan(y_rank))
    counts = valid.sum(axis=1)

    f_rank = np.where(valid, f_rank, 0.0)
    y_rank = np.where(valid, y_rank, 0.0)

    with np.errstate(invalid="ignore", divide="ignore"):
        f_centred = f_rank - (f_rank.sum(axis=1) / counts)[:, None]
        y_centred = y_rank - (y_rank.sum(axis=1) / counts)[:, None]
        f_centred = np.where(valid, f_centred, 0.0)
        y_centred = np.where(valid, y_centred, 0.0)

        numerator = (f_centred * y_centred).sum(axis=1)
        denominator = np.sqrt((f_centred**2).sum(axis=1) * (y_centred**2).sum(axis=1))
        rho = np.where(denominator > 0, numerator / denominator, np.nan)

    rho = rho[np.isfinite(rho)]
    if len(rho) < 2:
        return None
    return float(rho.mean())


def audit_cross_sectional_leakage(
    df: pd.DataFrame,
    group_col: str,
    target: str,
    max_lag: int = 3,
    excess_threshold: float = 0.15,
    min_correlation: float = 0.15,
    min_entities: int = 20,
    min_timestamps: int = 30,
) -> List[Issue]:
    """
    Detect cross-sectional lookahead (PNL002).

    A cross-sectional feature — a rank, z-score, decile or sector-neutralised
    value computed *across entities at one timestamp* — is legitimate when built
    from the cross-section at time t. Computed from the cross-section at t+1 and
    joined back to t, it is a leak.

    Why a dedicated check
    ---------------------
    The per-entity checks (LEK002/LEK003) do detect this, but only while
    idiosyncratic variation is large relative to any common factor. As a market
    factor comes to dominate, a *relative* measure decouples from each entity's
    own absolute outcome and the within-entity signal collapses, while the
    cross-sectional signal is unaffected. Measured on simulated panels (see
    ``docs/proposals/pnl002-cross-sectional-leakage.md``), LEK002 detection fell
    from 100% of entities to 22.5% as the common-factor ratio rose, while the
    cross-sectional correlation stayed at 1.0 throughout.

    That degradation is worse than a plain miss, because it feeds
    ``report.prevalence()``: a leak present in every entity would be reported as
    affecting 22% of them, which reads as "isolated" rather than "systemic".

    Method
    ------
    At each timestamp, correlate the feature against the target *across
    entities*, then average over timestamps::

        observed(k) = | mean_t  corr_e( f[e, t], y[e, t+k] ) |

    A feature can reach some of this legitimately, because entity ordering
    persists: a genuinely good cross-sectional signal today still ranks entities
    similarly tomorrow. So the same persistence baseline used by LEK003 is
    applied, in cross-sectional form::

        expected(k) = | observed(0) | * | cs_autocorr(y, k) |
        excess(k)   = observed(k) - expected(k)

    Flagged when ``excess(k) >= excess_threshold`` and
    ``observed(k) >= min_correlation`` for some k in 1..max_lag.

    Parameters
    ----------
    df : pd.DataFrame
        The full panel with a DatetimeIndex and an entity column.
    group_col : str
        Entity column name.
    target : str
        Target column name.
    max_lag : int
        Forward lags to test. Default 3.
    excess_threshold : float
        How far above the persistence baseline counts as leakage. Default 0.15 —
        deliberately stricter than LEK003's 0.1, because a real cross-sectional
        alpha factor legitimately carries some forward signal.
    min_correlation : float
        The observed cross-sectional correlation must itself reach this. Default
        0.15. Realistic factor rank-ICs are 0.02-0.08, so this leaves genuine
        factors alone.
    min_entities : int
        Minimum co-present entities at a timestamp for it to be scored. Below
        about 20, a cross-sectional correlation is mostly noise. Default 20.
    min_timestamps : int
        Minimum scored timestamps required to trust the average. Default 30.

    Returns
    -------
    List[Issue]
        Zero or more PNL002 Issues (WARNING).

    Notes
    -----
    This is a *suspicion* flag, not proof — exactly like LEK002/LEK003. A
    genuinely predictive cross-sectional factor produces the same signature as a
    leak, separated only by magnitude. Read ``excess`` before acting.
    """
    issues: List[Issue] = []

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("panel checks require a DatetimeIndex")
    for name, col in (("group_col", group_col), ("target", target)):
        if col not in df.columns:
            raise ValueError(f"{name} '{col}' not found in DataFrame columns.")
    if df.empty:
        return issues

    frame = df.copy()
    frame[group_col] = frame[group_col].astype(str)

    numeric = frame.select_dtypes(include=["number"]).replace([np.inf, -np.inf], np.nan)
    if target not in numeric.columns:
        return issues

    features = [c for c in numeric.columns if c != target]
    if not features:
        return issues

    stamps = frame.index
    target_wide = (
        pd.DataFrame(
            {"g": frame[group_col], "v": numeric[target]}, index=stamps
        ).pivot_table(
            index=stamps.name or None, columns="g", values="v", aggfunc="first"
        )
        if stamps.name
        else numeric[target].groupby([stamps, frame[group_col]]).first().unstack()
    )
    target_wide = target_wide.sort_index()
    if target_wide.shape[1] < min_entities or len(target_wide) < min_timestamps:
        return issues

    # Cross-sectional persistence of the target: how much does entity ordering
    # carry forward on its own? This is the baseline a feature may reach without
    # knowing anything it should not.
    persistence = {
        k: _cross_sectional_corr(target_wide, target_wide, k, min_entities)
        for k in range(1, max_lag + 1)
    }

    for col in features:
        feature_wide = (
            numeric[col]
            .groupby([stamps, frame[group_col]])
            .first()
            .unstack()
            .sort_index()
        )
        feature_wide = feature_wide.reindex(
            index=target_wide.index, columns=target_wide.columns
        )

        contemporaneous = _cross_sectional_corr(
            feature_wide, target_wide, 0, min_entities
        )
        if contemporaneous is None:
            continue

        best = None
        for k in range(1, max_lag + 1):
            if persistence[k] is None:
                continue
            observed = _cross_sectional_corr(feature_wide, target_wide, k, min_entities)
            if observed is None:
                continue
            expected = abs(contemporaneous) * abs(persistence[k])
            excess = abs(observed) - expected
            if best is None or excess > best[0]:
                best = (excess, k, observed, expected)

        if best is None:
            continue

        excess, lag, observed, expected = best
        if excess >= excess_threshold and abs(observed) >= min_correlation:
            issues.append(
                Issue(
                    module="panel",
                    code="PNL002",
                    severity=WARNING,
                    description=(
                        f"Feature '{col}' ranks entities in the order their future "
                        f"target values will fall, at lag +{lag} (cross-sectional "
                        f"Spearman={observed:.3f}) — more strongly than the target's "
                        f"own cross-sectional persistence explains (excess="
                        f"{excess:.3f}). This is the signature of a cross-sectional "
                        f"feature computed from a later timestamp and joined back. "
                        f"Verify it is built from the cross-section at each row's own "
                        f"timestamp."
                    ),
                    column=col,
                    evidence={
                        "metric": "cross_sectional_spearman",
                        "lag": int(lag),
                        "observed_cs_corr": round(float(observed), 4),
                        "expected_from_cs_persistence": round(float(expected), 4),
                        "excess": round(float(excess), 4),
                        "excess_threshold": excess_threshold,
                        "contemporaneous_cs_corr": round(float(contemporaneous), 4),
                        "n_entities": int(target_wide.shape[1]),
                        "group_col": group_col,
                    },
                )
            )

    return issues
