import numpy as np
import pandas as pd

from tsauditor.report.summary import Issue, WARNING
from tsauditor.anomaly._common import (
    zscore_preset,
    zscore_iqr_masks,
    esd_masking_recovery,
    # Re-exported for backward compatibility (this is where it used to live)
    # and because tests target it directly; the real definition and its full
    # docstring are in _common.py, alongside esd_masking_recovery, which is
    # also what remediate.py's repair step now shares this logic through.
    _generalized_esd,
)


def audit_point_anomalies(
    df: pd.DataFrame,
    zscore_threshold: float = None,
    domain: str = None,
) -> list:
    """
    Audits numeric columns for point anomalies using Z-score and IQR methods.

    Parameters
    ----------
    df : pd.DataFrame
        Time-series DataFrame with a DatetimeIndex.
    zscore_threshold : float, optional
        Absolute z-score above which a point is flagged. An explicitly passed
        value always wins over ``domain``; when None (the default) the
        threshold is derived from ``domain``.
    domain : str, optional
        Domain context ('finance' -> 5.0, 'sensor' -> 3.5, None -> 4.0).
        Only consulted when ``zscore_threshold`` is None.

    Returns
    -------
    list
        List of Issue objects describing point anomalies (ANO002).

    Notes
    -----
    When the z-score and IQR rules disagree (z-score finds nothing, IQR
    finds something), ``evidence["masking_suspected"]`` reports whether a
    generalized ESD re-scan suggests the raw z-score was blinded by heavy
    contamination, computed as ``n_esd > n_iqr * 0.5``. That ``0.5``
    multiplier is a heuristic, not a value derived from the ESD/Rosner
    literature or validated against a labeled contamination benchmark; it
    was chosen because it seemed reasonable, in the same spirit as
    CONTRIBUTING.md's policy on float thresholds.

    When ``masking_suspected`` is True, the points ESD itself identified as
    outliers (``_generalized_esd``'s Rosner procedure, which recomputes
    mean/std after each removal and so cannot be blinded by the
    contamination it is scoring) are folded into the points this function
    flags, not just reported as a diagnostic count. Before this, ESD's
    result never changed what was flagged -- only IQR's own points were
    reported, so a point beyond IQR's fence but still masked from the raw
    z-score (the exact case ``masking_suspected`` exists to name) was named
    in evidence as *suspected* but never itself surfaced as a flagged
    anomaly. ``evidence["esd_recovered_count"]`` reports how many flagged
    points came from ESD specifically (0 when ``masking_suspected`` is
    False, since ESD's result isn't used for flagging in that case).

    ``apply_fixes(outliers=...)`` stays consistent with this: ``"nan"``/
    ``"drop"`` repair the ESD-recovered points along with everything else
    (``remediate.py`` shares ``esd_masking_recovery``, the same function this
    detector uses), and ``"clip"`` now also reaches them -- not by clipping
    to a data-derived band (no such band is guaranteed to actually be
    outside every point Rosner's test flags; see ``_generalized_esd``'s
    docstring), but by NaN-ing just those specific rows and letting the
    ``missing`` imputation step fill them, separately from the ordinary
    z-band/IQR-fence clip applied to the rest of the column. See
    ``remediate._outlier_mask``, ``remediate._esd_masking_repair``, and the
    "clip" branch of ``apply_fixes`` for the mechanics and why a single
    whole-column clip bound cannot reach these points at all.

    This still does not cover every masking scenario: if contamination is
    heavy enough that the IQR rule *itself* finds nothing (Tukey's fence has
    roughly a 25% breakdown point, well above what the z-score's ~0%
    breakdown point tolerates but not unlimited), ``combined_mask`` is empty
    before ESD is even consulted, and the column is skipped with no ANO002
    Issue at all. Recovering that case would mean running ESD unconditionally
    on every column rather than only when IQR has already found something to
    disagree with the z-score about, which is a real added cost (ESD is
    O(k*n) per column) for a substantially rarer failure mode; it is left as
    a known, separate limitation rather than folded into this fix.
    """
    issues = []

    # 1. Validation
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a pd.DatetimeIndex")

    if df.empty:
        return issues

    # 2. Resolve the threshold. An explicit argument always wins; `domain` is a
    #    preset consulted only when the caller did not specify one. `is None`
    #    (not `or`) so that a deliberate 0.0 is honoured rather than treated as
    #    "unset". Mirrors audit_missing and audit_contextual_anomalies.
    z_thresh = zscore_preset(domain) if zscore_threshold is None else zscore_threshold

    numeric_cols = df.select_dtypes(include=["number"]).columns

    for col in numeric_cols:
        # Treat inf as missing, as every other detector does. Left in, an inf
        # makes mean inf and std NaN, so the comparisons below silently
        # evaluate to False and the whole column is skipped, including any
        # genuine outliers among its finite values.
        series = df[col].replace([np.inf, -np.inf], np.nan).dropna()
        if series.empty:
            continue

        # 3-4. Z-score + IQR methods, shared with remediate.py's repair step
        # (tsauditor.anomaly._common) so the two cannot drift apart.
        z_mask, iqr_mask, z_scores, degenerate = zscore_iqr_masks(series, z_thresh)
        # A zero-variance column has no outliers; a NaN std (fewer than two
        # observations) cannot be compared against.
        if degenerate:
            continue

        # 5. Consolidate and flag
        combined_mask = z_mask | iqr_mask
        if combined_mask.any():
            agreement_mask = z_mask & iqr_mask

            # Diagnostic first, detection second: ESD's result (see
            # _generalized_esd's docstring) can now change combined_mask
            # itself, so it must run before worst_pos/masked_abs_z are
            # computed from combined_mask below -- not after, as it did
            # when it was evidence-only.
            #
            # Resolves the otherwise ambiguous case where agreement_count is
            # 0: that happens both for a harmlessly skewed column and for
            # contamination heavy enough to blind the z-score, and the counts
            # alone cannot tell them apart. ESD can, because it recomputes
            # the scale after each removal.
            #
            # Only computed when the answer is actually needed. ESD is O(k*n)
            # (about 27ms on 1,000 points), and when the z-score rule agrees
            # with the IQR rule there is nothing to disambiguate.
            #
            # esd_masking_recovery (tsauditor.anomaly._common) is the single
            # place that decides "ambiguous", "masking_suspected", which
            # points ESD flags, and the ESD-consistent clip bound -- shared
            # with remediate.py's repair step (both the nan/drop mask and the
            # clip band for masked points) so detection and repair cannot
            # disagree about what a masked column's ANO002 finding covers.
            n_zscore = int(z_mask.sum())
            n_iqr = int(iqr_mask.sum())
            recovery = esd_masking_recovery(series, z_mask, iqr_mask)
            n_esd = recovery.n_esd
            masking_suspected = recovery.masking_suspected

            # Fold ESD's own flagged points into detection, not just into
            # evidence -- see this function's Notes for why a diagnostic-only
            # count left the exact case masking_suspected exists to name
            # (a point beyond IQR's fence but still masked from the raw
            # z-score) undetected. esd_masking_recovery already gates
            # esd_positions on masking_suspected (empty otherwise), so no
            # separate check is needed here.
            esd_recovered_count = 0
            if recovery.esd_positions:
                recovered = np.zeros(len(series), dtype=bool)
                recovered[recovery.esd_positions] = True
                already_flagged = combined_mask.to_numpy()
                esd_recovered_count = int((recovered & ~already_flagged).sum())
                combined_mask = combined_mask | pd.Series(
                    recovered, index=series.index
                )

            # Locate the worst *flagged* point, positionally. Label-based
            # lookup (series.loc[idxmax()]) returns a Series rather than a
            # scalar when the index has duplicate timestamps (as
            # panel/long-format data always does), and float() on that
            # raises TypeError. Positional access is unambiguous regardless
            # of index duplication.
            #
            # Restricted to combined_mask, not a column-wide argmax: the
            # single highest |z-score| point in the whole column is not
            # necessarily one of the points this call actually flagged. That
            # happens whenever z_mask and IQR disagree (the "ambiguous"
            # branch just above) and Tukey's fence (which can be asymmetric
            # on skewed data) picks a different point than the raw z-score
            # ranking would. An unrestricted argmax can then report
            # worst_value/worst_timestamp for a row nothing actually flagged,
            # which misdirects anyone reading the evidence to investigate the
            # wrong timestamp. Confirmed reproducible on plain Gaussian noise
            # (no injected contamination needed): seed=243, n=111.
            masked_abs_z = z_scores.abs().to_numpy()
            masked_abs_z = np.where(combined_mask.to_numpy(), masked_abs_z, -np.inf)
            worst_pos = int(masked_abs_z.argmax())

            issues.append(
                Issue(
                    module="anomaly",
                    code="ANO002",
                    severity=WARNING,
                    description=f"Point anomalies detected in column '{col}'.",
                    column=col,
                    evidence={
                        "zscore_outlier_count": n_zscore,
                        "iqr_outlier_count": n_iqr,
                        "agreement_count": int(agreement_mask.sum()),
                        # None when the z-score and IQR rules already agree, so
                        # there is nothing ambiguous to resolve.
                        "esd_outlier_count": n_esd,
                        "masking_suspected": masking_suspected,
                        # How many of the points in this Issue came from ESD
                        # rather than z-score/IQR -- 0 unless masking_suspected.
                        "esd_recovered_count": esd_recovered_count,
                        # Same masked-argmax as worst_pos, not a column-wide
                        # z_scores.abs().max() -- max_zscore is presented
                        # alongside worst_value/worst_timestamp as describing
                        # the same point, so it must be consistent with them
                        # rather than silently describing a different,
                        # possibly-unflagged row.
                        "max_zscore": round(float(masked_abs_z[worst_pos]), 4),
                        "worst_value": float(series.iloc[worst_pos]),
                        "worst_timestamp": str(series.index[worst_pos]),
                    },
                )
            )

    return issues
