import numpy as np
import pandas as pd
from scipy import stats

from tsauditor.report.summary import Issue, WARNING
from tsauditor.anomaly._common import zscore_preset, zscore_iqr_masks

# Cap on how many outliers the ESD diagnostic will look for, as a fraction of
# the column length. ESD is O(k*n); beyond ~40% contamination the "outliers"
# are a second population rather than anomalies.
_ESD_MAX_FRACTION = 0.4
_ESD_MIN_OBS = 15


def _generalized_esd(values: np.ndarray, alpha: float = 0.05) -> int:
    """
    Rosner's Generalized ESD test — estimated number of outliers.

    Reported as *evidence only*; it does not affect what ANO002 flags.

    Why it is here: the z-score half of ANO002 goes blind under heavy
    contamination, because the outliers inflate the standard deviation that
    judges them. That makes ``agreement_count`` drop to zero exactly when
    contamination is worst — indistinguishable from a harmlessly skewed column.
    ESD removes the most extreme point and *recomputes* the mean and standard
    deviation before testing the next, so masking cannot occur by construction.

    Measured against 1,000 clean points with outliers planted at 10 sigma: exact
    at every level (1, 5, 20, 50, 150, 300), and 0 on clean Gaussian data where
    the IQR rule reports 10 false positives.

    Reference: Rosner, B. (1983), "Percentage Points for a Generalized ESD
    Many-Outlier Procedure", Technometrics 25(2), 165-172.
    """
    n = len(values)
    if n < _ESD_MIN_OBS:
        return 0

    max_outliers = max(1, int(_ESD_MAX_FRACTION * n))
    work = values.astype(float, copy=True)
    test_statistics = []
    criticals = []

    for i in range(1, max_outliers + 1):
        if len(work) < 3:
            break
        mean, std = work.mean(), work.std(ddof=1)
        if std == 0 or not np.isfinite(std):
            break

        deviation = np.abs(work - mean)
        worst = int(deviation.argmax())
        test_statistics.append(deviation[worst] / std)
        work = np.delete(work, worst)

        p = 1.0 - alpha / (2.0 * (n - i + 1))
        t = stats.t.ppf(p, n - i - 1)
        criticals.append((n - i) * t / np.sqrt((n - i - 1 + t**2) * (n - i + 1)))

    estimated = 0
    for i in range(len(test_statistics)):
        if test_statistics[i] > criticals[i]:
            estimated = i + 1
    return estimated


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
    finds something), ``evidence["masking_suspected"]`` flags whether a
    generalized ESD re-scan suggests the raw z-score was blinded by heavy
    contamination, computed as ``n_esd > n_iqr * 0.5``. That ``0.5``
    multiplier is a heuristic, not a value derived from the ESD/Rosner
    literature or validated against a labeled contamination benchmark; it
    was chosen because it seemed reasonable, in the same spirit as
    CONTRIBUTING.md's policy on float thresholds. It only affects this
    diagnostic field, never which points get flagged, so a wrong call here
    doesn't change what's reported as an anomaly, only how confidently the
    evidence explains itself.
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
        # evaluate to False and the whole column is skipped — including any
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

            # Locate the worst point *positionally*. Label-based lookup
            # (series.loc[idxmax()]) returns a Series rather than a scalar when
            # the index has duplicate timestamps — as panel/long-format data
            # always does — and float() on that raises TypeError. Positional
            # access is unambiguous regardless of index duplication.
            worst_pos = int(z_scores.abs().to_numpy().argmax())

            # Diagnostic only — never changes what is flagged. Resolves the
            # otherwise ambiguous case where agreement_count is 0: that happens
            # both for a harmlessly skewed column and for contamination heavy
            # enough to blind the z-score, and the counts alone cannot tell them
            # apart. ESD can, because it recomputes the scale after each removal.
            #
            # Only computed when the answer is actually needed. ESD is O(k*n) —
            # about 27ms on 1,000 points — and when the z-score rule agrees with
            # the IQR rule there is nothing to disambiguate.
            n_zscore = int(z_mask.sum())
            n_iqr = int(iqr_mask.sum())
            ambiguous = n_zscore == 0 and n_iqr > 0

            n_esd = (
                _generalized_esd(series.to_numpy(dtype=float)) if ambiguous else None
            )
            masking_suspected = bool(
                ambiguous and n_esd is not None and n_esd > n_iqr * 0.5
            )

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
                        "max_zscore": round(float(z_scores.abs().max()), 4),
                        "worst_value": float(series.iloc[worst_pos]),
                        "worst_timestamp": str(series.index[worst_pos]),
                    },
                )
            )

    return issues
