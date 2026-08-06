import pandas as pd
import numpy as np
from tsauditor.report.summary import Issue, WARNING
from tsauditor.anomaly._common import (
    stuck_window_preset,
    spike_threshold_preset,
    stuck_run_mask,
    spike_stats,
    SPIKE_WINDOW,
)
from tsauditor.utils.validation import ensure_sorted_datetime_index


def audit_contextual_anomalies(
    df: pd.DataFrame,
    stuck_window: int = None,
    spike_threshold: float = None,
    spike_window: int = None,
    domain: str = None,
    handle_missing: str = "strict",
) -> list:
    """
    Audits numeric columns for stuck values (ANO001) and contextual spikes
    (ANO003).

    Parameters
    ----------
    df : pd.DataFrame
        Time-series DataFrame with a DatetimeIndex.
    stuck_window : int, optional
        A run longer than this is flagged as stuck. Derived from ``domain``
        when None ('sensor' -> 3, otherwise 5).
    spike_threshold : float, optional
        Local z-score above which a point is flagged as a spike. Derived from
        ``domain`` when None ('finance' -> 4.0, 'sensor' -> 3.0, None -> 3.5).
    spike_window : int, optional
        Width of the local context window for ANO003. Defaults to 21.
    domain : str, optional
        Domain context ('finance', 'sensor', or None). Only consulted for
        parameters the caller left as None.
    handle_missing : str
        "interpolate" fills single-row gaps before auditing; anything else
        leaves NaNs in place.

    Returns
    -------
    list
        List of Issue objects (ANO001 and/or ANO003).
    """
    issues = []

    # Both ANO001 (consecutive-run detection) and ANO003 (rolling local
    # z-score) depend on row *position*, not just on the index being a
    # DatetimeIndex -- a valid-but-unsorted index previously made this
    # function silently miss real stuck runs and spikes. See
    # ensure_sorted_datetime_index's docstring.
    df = ensure_sorted_datetime_index(df, "audit_contextual_anomalies")
    if df.empty:
        return issues

    # Domain defaults. An explicitly passed argument always wins; `domain` is a
    # preset consulted only for parameters left as None. `is None` (not `or`)
    # so a deliberate 0 is honoured rather than treated as "unset". Presets
    # live in tsauditor.anomaly._common, shared with remediate.py's repair
    # step so the two cannot drift apart.
    if stuck_window is None:
        stuck_window = stuck_window_preset(domain)
    if spike_threshold is None:
        spike_threshold = spike_threshold_preset(domain)

    # Local context window for ANO003. Must be wide enough to estimate the
    # local spread reliably: a 4-5 point window gives a noisy std and floods
    # the result with false positives once the current point is excluded.
    if spike_window is None:
        spike_window = SPIKE_WINDOW

    for col in df.select_dtypes(include=["number"]).columns:
        series = df[col].copy()

        if handle_missing == "interpolate":
            series = series.interpolate(method="linear", limit=1)

        # ANO003's rolling window tolerates NaNs via min_periods, so it uses
        # a NaN-dropped view; ANO001 bridges gaps internally via
        # stuck_run_mask below, using `series` (not this dropped view).
        series_clean = series.dropna()
        if series_clean.empty:
            continue

        # --- ANO001 ---
        # Group by consecutive values, bridging a lone missing reading inside
        # an otherwise-flat run (still a stuck run) -- this is about what
        # "stuck" means, not a general missing-data preference, so it applies
        # regardless of handle_missing (which governs the series used by
        # ANO003 above, a separate concern). Bridging via `series`/
        # `series_clean` directly is deliberately avoided: interpolating
        # unconditionally would also change what ANO003 sees.
        #
        # Shared with remediate.py's repair step (tsauditor.anomaly._common)
        # so detection and repair cannot silently disagree about which rows
        # are part of a stuck run -- see stuck_run_mask's docstring for the
        # bridging rationale in full.
        stuck_mask, counts = stuck_run_mask(series, stuck_window)

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
        # spike scored only z ~= 1.8 in a centered 5-window). Shared with
        # remediate.py's repair step (tsauditor.anomaly._common) so the
        # detection z-scores and the repair clip band come from the same
        # local-context computation.
        spike_mask, z_scores, flat_context_spike, _, _ = spike_stats(
            series_clean, spike_window, spike_threshold
        )

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
