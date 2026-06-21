import pandas as pd
import numpy as np
from tsauditor.report.summary import Issue, WARNING


def audit_contextual_anomalies(
    df: pd.DataFrame,
    stuck_window: int = None,
    spike_threshold: float = None,
    spike_window: int = None,
    domain: str = None,
    handle_missing: str = "strict",
) -> list:
    issues = []

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a pd.DatetimeIndex")
    if df.empty:
        return issues

    # Domain defaults
    if domain == "finance":
        stuck_window = stuck_window or 5
        spike_threshold = spike_threshold or 4.0
    elif domain == "sensor":
        stuck_window = stuck_window or 3
        spike_threshold = spike_threshold or 3.0
    else:
        stuck_window = stuck_window or 5
        spike_threshold = spike_threshold or 3.5

    # Local context window for ANO003. Must be wide enough to estimate the
    # local spread reliably: a 4-5 point window gives a noisy std and floods
    # the result with false positives once the current point is excluded.
    spike_window = spike_window or 21

    for col in df.select_dtypes(include=["number"]).columns:
        series = df[col].copy()

        if handle_missing == "interpolate":
            series = series.interpolate(method="linear", limit=1)

        # We need a copy that doesn't drop NaNs for spike calc,
        # but ANO001 needs to break on NaNs in strict mode
        series_clean = series.dropna()
        if series_clean.empty:
            continue

        # --- ANO001 ---
        # Group by consecutive values. Note: diff() on NaN results in NaN,
        # so this correctly breaks the group when a NaN is present.
        diffs = series.diff().ne(0).cumsum()
        counts = series.groupby(diffs).transform("count")
        stuck_mask = (counts > stuck_window) & series.notna()

        if stuck_mask.any():
            issues.append(
                Issue(
                    module="anomaly",
                    code="ANO001",
                    severity=WARNING,
                    description="Stuck values detected.",
                    column=col,
                    evidence={"max_stuck_duration": int(counts[stuck_mask].max())},
                )
            )

        # --- ANO003: contextual spike detection ---
        # Compare each point to its LOCAL context (the surrounding window),
        # EXCLUDING the point itself. If the point stays in its own window an
        # extreme spike inflates the window mean and std and masks itself, so
        # |z| never crosses the threshold (this was the original bug: a 50x
        # spike scored only z ~= 1.8 in a centered 5-window).
        sq = series_clean.pow(2)
        mp = max(3, spike_window // 2)
        roll = series_clean.rolling(window=spike_window, center=True, min_periods=mp)
        roll_sq = sq.rolling(window=spike_window, center=True, min_periods=mp)

        n_excl = roll.count() - 1  # neighbours, excluding self
        sum_excl = roll.sum() - series_clean
        sumsq_excl = roll_sq.sum() - sq

        local_mean = sum_excl / n_excl
        local_var = (sumsq_excl / n_excl) - local_mean.pow(2)
        local_std = np.sqrt(local_var.clip(lower=0))  # clip kills tiny fp negatives
        deviation = (series_clean - local_mean).abs()

        with np.errstate(divide="ignore", invalid="ignore"):
            z_scores = deviation / local_std

        # A point that differs from a perfectly flat local context (std == 0)
        # is a definite spike, but its z-score is undefined (x / 0). Flag it
        # explicitly instead of silently dropping it via NaN.
        flat_context_spike = (local_std == 0) & (deviation > 0) & (n_excl >= 2)

        spike_mask = ((z_scores > spike_threshold) | flat_context_spike).fillna(False)

        if spike_mask.any():
            finite_z = z_scores[spike_mask].replace([np.inf, -np.inf], np.nan)
            max_z = finite_z.max()
            issues.append(
                Issue(
                    module="anomaly",
                    code="ANO003",
                    severity=WARNING,
                    description="Contextual spikes detected.",
                    column=col,
                    evidence={
                        "n_spikes": int(spike_mask.sum()),
                        "max_spike_zscore": round(float(max_z), 4)
                        if pd.notna(max_z)
                        else None,
                        "zero_variance_context": bool(flat_context_spike.any()),
                    },
                )
            )
    return issues
