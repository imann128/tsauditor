"""
tsauditor.profiler._common
----------------------------
Shared run-length-encoding helper for the profiler checks.

``frequency.py`` (PRF001/PRF005, run lengths of large gaps) and
``missing.py`` (PRF002, run lengths of consecutive NaNs) each computed run
starts, ends, and lengths from a 0/1 int array using the same three-step
numpy pattern, independently -- two copies of one algorithm with no shared
implementation, the same category of drift risk as the anomaly presets in
``tsauditor/anomaly/_common.py`` (see CHANGELOG [0.5.0]). Centralizing
it here means a fix to the run-length boundary logic only needs to be made
once and cannot silently diverge between the two callers.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def consecutive_run_lengths(
    flags: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run starts, ends, and lengths for a 0/1 int array marking a per-row
    condition (e.g. "is this a large gap" or "is this row missing").

    Parameters
    ----------
    flags : np.ndarray
        1-D array of 0s and 1s, in row order.

    Returns
    -------
    run_starts, run_ends, run_lengths : np.ndarray
        Positions where each run of 1s begins, ends (exclusive), and its
        length, in order of appearance. All three are empty int arrays if
        ``flags`` is empty or has no runs of 1s.
    """
    if len(flags) == 0:
        empty = np.array([], dtype=int)
        return empty, empty, empty

    run_starts = np.where((flags[:-1] == 0) & (flags[1:] == 1))[0] + 1
    if flags[0] == 1:
        run_starts = np.insert(run_starts, 0, 0)

    run_ends = np.where((flags[:-1] == 1) & (flags[1:] == 0))[0] + 1
    if flags[-1] == 1:
        run_ends = np.append(run_ends, len(flags))

    run_lengths = run_ends - run_starts
    return run_starts, run_ends, run_lengths
