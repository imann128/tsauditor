import pandas as pd
from tsauditor.report.summary import Issue, WARNING


def audit_point_anomalies(
    df: pd.DataFrame,
    zscore_threshold: float = 4.0,
    domain: str = None,
) -> list:
    """
    Audits numeric columns for point anomalies using Z-score and IQR methods.
    """
    issues = []

    # 1. Validation
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a pd.DatetimeIndex")

    if df.empty:
        return issues

    # 2. Resolve Domain-specific thresholds
    if domain == "finance":
        z_thresh = 5.0
    elif domain == "sensor":
        z_thresh = 3.5
    else:
        z_thresh = zscore_threshold

    numeric_cols = df.select_dtypes(include=["number"]).columns

    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            continue

        # 3. Z-score Method
        mean, std = series.mean(), series.std()
        if std == 0:
            continue
        z_scores = (series - mean) / std
        z_mask = abs(z_scores) > z_thresh

        # 4. IQR Method
        q25, q75 = series.quantile([0.25, 0.75])
        iqr = q75 - q25
        iqr_mask = (series < q25 - 1.5 * iqr) | (series > q75 + 1.5 * iqr)

        # 5. Consolidate and flag
        combined_mask = z_mask | iqr_mask
        if combined_mask.any():
            agreement_mask = z_mask & iqr_mask
            worst_idx = z_scores.abs().idxmax()

            issues.append(
                Issue(
                    module="anomaly",
                    code="ANO002",
                    severity=WARNING,
                    description=f"Point anomalies detected in column '{col}'.",
                    column=col,
                    evidence={
                        "zscore_outlier_count": int(z_mask.sum()),
                        "iqr_outlier_count": int(iqr_mask.sum()),
                        "agreement_count": int(agreement_mask.sum()),
                        "max_zscore": round(float(z_scores.abs().max()), 4),
                        "worst_value": float(series.loc[worst_idx]),
                        "worst_timestamp": str(worst_idx),
                    },
                )
            )

    return issues
