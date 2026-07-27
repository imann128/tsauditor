"""
tsauditor.report.remediation
-----------------------------
The advisory layer. tsauditor detects and reports problems but never edits the
user's data — dropping or rewriting a feature is a modeling decision only the
user can make. This module maps each issue code to a concrete *suggested
action* so the report tells the user what to consider doing, while leaving the
decision (and the data) in their hands.

Suggestions are derived from the issue code (plus its column and evidence), so
detectors do not each need to carry remediation text. Add a new code here when
you add a new check.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


# code -> suggested-action template. Templates may reference {target} (a phrase
# like "column 'X'" or "the dataset") and any key present in the issue evidence.
_REMEDIATIONS: Dict[str, str] = {
    # ── Leakage ──────────────────────────────────────────────────────────────
    "LEK001": (
        "Remove or reconstruct {target}: it near-deterministically reproduces the "
        "target variable and will leak. Keep it only if you can confirm it is "
        "genuinely available at prediction time."
    ),
    "LEK002": (
        "Inspect how {target} is built. Its strongest correlation with the target "
        "falls at a future lag (+{peak_lag}), which means it aligns with future "
        "target values — a sign it encodes information not available at prediction time."
    ),
    "LEK003": (
        "Confirm {target} is computed only from past data. It tracks the future "
        "target beyond what the target's own persistence explains, the signature of "
        "a centered or forward-looking window."
    ),
    "LEK004": (
        "Align {target} to its release schedule: some values sit at a timestamp "
        "earlier than when they were published, so rows before each release use "
        "future information. Shift the column forward to its availability dates "
        "(not reference dates) rather than dropping it."
    ),
    # ── Validity ─────────────────────────────────────────────────────────────
    "VAL001": (
        "Inspect the out-of-range values in {target}: they fall outside the "
        "declared valid range and are likely data-feed glitches or a scaling "
        "error. Correct or drop them before modeling."
    ),
    "VAL002": (
        "Rows in {target} break an ordering constraint (e.g. a crossed book where "
        "bid exceeds ask). Inspect these timestamps for feed glitches and correct "
        "or remove them before modeling."
    ),
    "LEK005": (
        "Check how the target was defined. {target} and its partner column "
        "together reconstruct it almost exactly while neither does alone, which "
        "usually means the target is an arithmetic function of the two (a "
        "difference, a mean, a spread). Remove or rebuild whichever inputs are "
        "not genuinely available at prediction time."
    ),
    # ── Panel structure ──────────────────────────────────────────────────────
    "PNL002": (
        "Verify {target} is built from the cross-section at each row's own "
        "timestamp, not a later one. It currently ranks entities in the order "
        "their future target values fall, beyond what the target's own "
        "cross-sectional persistence explains. If it is a genuine predictive "
        "factor, confirm the magnitude is plausible before keeping it."
    ),
    "PNL001": (
        "Align the panel to a common time index before any cross-sectional "
        "work: reindex each entity onto the full timestamp set (and decide "
        "explicitly whether to forward-fill or leave gaps), or restrict the "
        "panel to the {n_complete_groups} entities with complete coverage."
    ),
    "PNL003": (
        "Treat findings for the short entities as provisional — they fall below "
        "the {min_rows}-row minimum the leakage and stationarity checks need. "
        "Either gather more history for them or exclude them from the audit, "
        "rather than reading their silence as a clean result."
    ),
    # ── Anomaly ──────────────────────────────────────────────────────────────
    "ANO001": (
        "Investigate {target} for a stuck sensor or a forward-filled gap: the value "
        "repeats unchanged for an unusually long run."
    ),
    "ANO002": (
        "Review the point outliers in {target} and decide whether to winsorize, "
        "transform, or treat them as data errors before modeling."
    ),
    "ANO003": (
        "Examine the local spikes in {target}: they deviate sharply from their "
        "neighbours and may be data-entry or data-feed errors."
    ),
    # ── Profiler ─────────────────────────────────────────────────────────────
    "PRF001": (
        "Resample {target} to a regular frequency, or document why the timestamps "
        "are irregular, before time-based modeling."
    ),
    "PRF002": (
        "Handle the clustered missing values in {target} (careful interpolation, "
        "limited forward-fill, or dropping the affected span) before modeling."
    ),
    "PRF003": (
        "Difference or otherwise transform {target} to achieve stationarity before "
        "modeling; non-stationary inputs bias many time-series models."
    ),
    "PRF004": "Remove or aggregate the duplicate timestamps in {target}.",
    "PRF005": (
        "Review the clustered gaps in {target}; they may indicate a collection "
        "outage or missing period that should be handled explicitly."
    ),
    "PRF006": (
        "{target} has a high overall missing rate — consider dropping it or imputing "
        "with care, and check whether the missingness is informative."
    ),
}

_FALLBACK = "Review this issue before using the data for modeling."


class _SafeDict(dict):
    """dict for str.format_map that leaves unknown placeholders untouched."""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def suggest(
    code: str,
    column: Optional[str] = None,
    evidence: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Return a human-readable suggested action for an issue.

    Parameters
    ----------
    code : str
        Issue code, e.g. "LEK001".
    column : Optional[str]
        Affected column, or None for a dataset-level issue.
    evidence : Optional[Dict[str, Any]]
        The issue's evidence dict; its keys may fill template placeholders.

    Returns
    -------
    str
        The suggested action, or a generic fallback for unknown codes.
    """
    template = _REMEDIATIONS.get(code, _FALLBACK)
    target = f"column '{column}'" if column else "the dataset"
    fields = _SafeDict(evidence or {})
    fields["target"] = target
    fields.setdefault("column", column or "this column")
    return template.format_map(fields)
