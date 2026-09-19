"""
tsauditor.report.pdf
--------------------
PDF export for a GuardReport. This is the only module that imports matplotlib,
gated behind the optional ``[pdf]`` extra.

The report is a formal document: serif (Times New Roman) text, black
throughout, clear section headings, and tables where the content is tabular.
The output is vector and text-selectable, so it copies and OCRs cleanly (e.g.
AWS Textract). The machine-readable companion is ``GuardReport.to_json``.

One deliberate exception to "no charts, no colour": the lead/lag
cross-correlation heatmap (see ``_correlation_heatmap_fig``). A grid of
signed correlations is not usefully readable as a table at any width that
fits a page, and colour is what makes a positive-vs-negative lag peak visible
at a glance. Everything else in the report stays black-and-white tabular by
design; this page is the one place a chart earns its keep.

Layout
------
Scorecard (health score, dataset overview, before/after, leakage callout,
executive summary) and the Detected Issues table share the first page when the
issue list is short; long lists spill onto continuation pages. The lead/lag
heatmap (when applicable, see ``export_pdf``'s ``include_correlation_heatmap``)
is its own page at the end.
"""

from __future__ import annotations

import textwrap
from typing import List, Optional

import numpy as np
import pandas as pd

from tsauditor.remediate import health_score

_A4 = (8.27, 11.69)
_ROW_H = 0.05  # page-fraction height of one issues-table row
_BOTTOM = 0.05  # bottom margin

_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "text.color": "black",
    "axes.edgecolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
}


def _require_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages

        return plt, PdfPages
    except ImportError as exc:  # pragma: no cover - exercised only without mpl
        raise ImportError(
            "PDF export requires matplotlib, which is an optional dependency.\n"
            "Install it with:  pip install 'tsauditor[pdf]'"
        ) from exc


def _status(score: float) -> str:
    if score >= 90:
        return "Excellent"
    if score >= 70:
        return "Needs Review"
    return "At Risk"


def _executive_summary(report, df) -> List[str]:
    codes = [i.code for i in report.all_issues]
    miss_cols = sorted(
        {
            i.column
            for i in report.all_issues
            if i.code in ("PRF002", "PRF006") and i.column
        }
    )
    miss_cells = (
        int(df[miss_cols].isna().sum().sum()) if (df is not None and miss_cols) else 0
    )
    n_stuck = codes.count("ANO001")
    n_outlier = codes.count("ANO002")
    n_spike = codes.count("ANO003")
    n_leak = len(report.leaky_columns())

    lines: List[str] = []
    if miss_cells:
        lines.append(
            f"{miss_cells} missing data cells across {len(miss_cols)} column(s)."
        )
    if n_stuck:
        lines.append(f"{n_stuck} stuck-value segment(s) detected.")
    if n_outlier or n_spike:
        lines.append(
            f"{n_outlier} column(s) with point outliers; "
            f"{n_spike} with contextual spikes."
        )
    if n_leak:
        lines.append(f"{n_leak} target-leakage column(s) - exclude before modeling.")
    if not lines:
        lines.append("No data-quality or leakage issues detected. Data is model-ready.")
    return lines


def _style(table, fontsize, header_rows=1):
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    for (r, _c), cell in table.get_celld().items():
        cell.set_edgecolor("black")
        cell.set_linewidth(0.5)
        txt = cell.get_text()
        txt.set_color("black")
        txt.set_fontfamily("serif")
        txt.set_verticalalignment("center")
        if r < header_rows:
            txt.set_fontweight("bold")


def _kv_table(fig, rect, col_labels, rows, col_widths, fontsize=8.5):
    ax = fig.add_axes(rect)
    ax.axis("off")
    if not rows:
        rows = [["(none)"] + [""] * (len(col_labels) - 1)]
    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc="left",
        colWidths=col_widths,
        loc="upper left",
    )
    table.scale(1, 1.4)
    _style(table, fontsize)


def _wrap_desc(text: str, width: int = 50, max_lines: int = 2) -> str:
    lines = textwrap.wrap(text, width=width) or ["-"]
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip(".") + " ..."
    return "\n".join(lines)


def _issues_table(fig, left, top, width, issues, fontsize=8) -> None:
    """Draw the issues table with a fixed bbox so row heights are deterministic."""
    rows = [
        [i.code, i.severity.upper(), i.column or "-", _wrap_desc(i.description)]
        for i in issues
    ] or [["-", "-", "-", "-"]]
    n = len(rows) + 1  # + header
    height = n * _ROW_H
    ax = fig.add_axes([left, top - height, width, height])
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["Code", "Severity", "Column", "Description"],
        colWidths=[0.12, 0.15, 0.20, 0.53],
        cellLoc="left",
        bbox=[0, 0, 1, 1],
    )
    _style(table, fontsize)


def _prevalence_table(fig, left, top, width, rows, fontsize=8) -> None:
    """
    Draw the panel prevalence table: one row per (code, column) finding, with
    how many entities it hits -- the panel counterpart to _issues_table.

    Without this, a panel PDF rendered report.all_issues directly: one row
    per entity per finding, with no entity label on the row at all (the row
    tuple never included Issue.group), so a systemic bug present in every
    entity produced hundreds or thousands of visually-identical rows with no
    way to tell them apart, spread across as many continuation pages as it
    took to fit them. report.summary()'s CLI output already avoided this via
    report.prevalence(); this mirrors that for the PDF.
    """
    table_rows = [
        [
            r["severity"].upper(),
            r["code"],
            r["column"] or "-",
            f"{r['n_groups']}/{r['total_groups']}"
            if r["n_groups"] is not None
            else "panel-level",
            f"{r['pct']}%" if r["pct"] is not None else "-",
            ", ".join(r["example_groups"]) or "-",
        ]
        for r in rows
    ] or [["-", "-", "-", "-", "-", "-"]]
    n = len(table_rows) + 1  # + header
    height = n * _ROW_H
    ax = fig.add_axes([left, top - height, width, height])
    ax.axis("off")
    table = ax.table(
        cellText=table_rows,
        colLabels=["Severity", "Code", "Column", "Entities", "%", "Examples"],
        colWidths=[0.12, 0.10, 0.18, 0.14, 0.08, 0.38],
        cellLoc="left",
        bbox=[0, 0, 1, 1],
    )
    _style(table, fontsize)


def _correlation_heatmap_fig(plt, matrix_df: pd.DataFrame, target: str, top_n: int):
    """
    Build the lead/lag cross-correlation heatmap page.

    ``matrix_df`` is expected to come straight from
    ``tsauditor.leakage.correlation.lag_correlation_matrix``: the same grid
    LEK002 searches for a positive-lag peak. Rows are ranked by peak |r|
    across lags and truncated to ``top_n`` so a wide frame doesn't produce an
    unreadable (or un-renderable) page; the truncation is stated on the page
    itself rather than silently dropping features.

    Cells with no value (fewer than ``min_obs`` overlapping observations at
    that lag, or a constant subset) are left blank rather than shown as 0:
    a blank cell and a measured-zero correlation are not the same claim.
    """
    peak_abs = matrix_df.abs().max(axis=1, skipna=True)
    ranked = peak_abs.sort_values(ascending=False)
    selected = ranked.index[:top_n]
    sub = matrix_df.loc[selected]
    truncated = len(selected) < len(matrix_df)

    lags = list(sub.columns)
    zero_col = lags.index(0) if 0 in lags else None

    fig = plt.figure(figsize=_A4)
    fig.text(0.08, 0.955, "Lead/Lag Cross-Correlation Heatmap", fontsize=14, weight="bold")
    subtitle = (
        f"Spearman correlation of each feature with target '{target}' across "
        f"lags {lags[0]} to {lags[-1]}. Positive lag = feature aligns with "
        f"future target values (LEK002 territory)."
    )
    y_note = 0.935
    for line in textwrap.wrap(subtitle, width=95):
        fig.text(0.08, y_note, line, fontsize=8.5)
        y_note -= 0.017
    y_note -= 0.001
    if truncated:
        fig.text(
            0.08,
            y_note,
            f"Showing top {len(selected)} of {len(matrix_df)} features by peak |correlation|.",
            fontsize=8,
            style="italic",
        )
        y_note -= 0.016
    fig.text(
        0.08,
        y_note,
        "Blank cells: fewer than min_obs overlapping observations at that lag, "
        "or a constant subset (undefined, not zero).",
        fontsize=8,
        style="italic",
    )

    # Top-anchor the plot right below the header text instead of pinning it
    # to the bottom margin: with few rows, a bottom-anchored axes leaves a
    # large dead gap between the notes and the chart.
    n_rows = max(len(sub), 1)
    top = y_note - 0.03
    height = min(0.03 * n_rows + 0.08, top - _BOTTOM)
    bottom = top - height

    ax = fig.add_axes([0.30, bottom, 0.55, height])
    masked = np.ma.masked_invalid(sub.to_numpy(dtype=float))
    im = ax.imshow(masked, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
    im.cmap.set_bad(color="0.85")  # blank cells render as light grey, not white-on-white

    ax.set_yticks(range(len(sub)))
    ax.set_yticklabels(list(sub.index), fontsize=7)
    step = max(1, len(lags) // 15)  # thin x labels so they don't overlap on wide max_lag
    ax.set_xticks(range(0, len(lags), step))
    ax.set_xticklabels([str(lags[i]) for i in range(0, len(lags), step)], fontsize=7)
    ax.set_xlabel("Lag (periods)", fontsize=9)
    if zero_col is not None:
        ax.axvline(zero_col, color="black", lw=0.8, linestyle="--")

    cax = fig.add_axes([0.87, bottom, 0.02, height])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Spearman r", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    return fig


def _capacity(top: float) -> int:
    """How many table rows (issues or prevalence rows) fit between ``top`` and
    the bottom margin. Both tables share the same fixed row height (_ROW_H),
    so one capacity function serves either."""
    return max(1, int((top - _BOTTOM) / _ROW_H) - 1)  # -1 for the header row


def export_pdf(
    report,
    path: str,
    df: Optional[pd.DataFrame] = None,
    fixed_df: Optional[pd.DataFrame] = None,
    title: Optional[str] = None,
    include_correlation_heatmap: bool = True,
    heatmap_max_lag: int = 10,
    heatmap_top_n: int = 30,
) -> str:
    """
    Render ``report`` (and optionally ``df``/``fixed_df``) as a PDF at ``path``.

    ``include_correlation_heatmap``, ``heatmap_max_lag``, ``heatmap_top_n``
    control the lead/lag cross-correlation heatmap page (see
    ``_correlation_heatmap_fig``). It only renders when all of the following
    hold: ``include_correlation_heatmap`` is True, ``df`` is given,
    ``report.metadata['target']`` is set and present in ``df``'s columns, and
    the report is not a panel scan: cross-sectional panel data needs its own
    aggregation story (per-entity vs. pooled) that this page does not yet
    provide, so a panel report gets a one-line note instead of a
    silently-wrong pooled heatmap.
    """
    plt, PdfPages = _require_matplotlib()
    meta = report.metadata
    title = title or "Time-Series Data Health Report"

    score = health_score(report, df) if df is not None else None
    after = None
    if fixed_df is not None:
        from tsauditor import scan

        # group_col must be threaded through here exactly as
        # GuardReport.health_score() does: without it, a panel scan treats
        # every entity as one interleaved series (wrong detection, and even
        # where detection is incidentally still right, affected_cells()
        # would recompute masks on values mixed across entities of very
        # different scale). This is the same fix as summary.py's
        # GuardReport.to_json() -- a separate, independent copy of the same
        # re-scan that had the same gap. The five detector-tuning settings
        # are threaded through for the same reason: without them this
        # re-scan silently falls back to the domain-only preset regardless
        # of any explicit override the original scan() call used.
        after_report = scan(
            fixed_df,
            target=meta.get("target"),
            domain=meta.get("domain"),
            group_col=meta.get("group_col"),
            zscore_threshold=meta.get("zscore_threshold"),
            stuck_window=meta.get("stuck_window"),
            spike_threshold=meta.get("spike_threshold"),
            spike_window=meta.get("spike_window"),
            handle_missing=meta.get("handle_missing") or "strict",
            run_leakage=False,
            run_stationarity=False,
        )
        after = health_score(after_report, fixed_df)
    leak_cols = report.leaky_columns()

    with plt.rc_context(_RC), PdfPages(path) as pdf:
        fig = plt.figure(figsize=_A4)
        fig.text(0.08, 0.955, title, fontsize=17, weight="bold")
        fig.text(0.08, 0.935, "Generated by tsauditor", fontsize=8.5)
        fig.lines.append(
            plt.Line2D(
                [0.08, 0.92],
                [0.928, 0.928],
                transform=fig.transFigure,
                color="black",
                lw=0.8,
            )
        )

        # Top band: dataset overview (left) and health score (right)
        fig.text(0.08, 0.90, "Dataset Overview", fontsize=12, weight="bold")
        meta_rows = [
            [k.replace("_", " ").title(), str(meta[k])]
            for k in (
                "rows",
                "columns",
                "time_start",
                "time_end",
                "frequency",
                "domain",
                "target",
            )
            if meta.get(k) is not None
        ]
        if report.is_panel:
            # group_col/n_groups otherwise never appear anywhere in the PDF,
            # unlike report.summary()'s CLI output, which always states the
            # entity count and grouping column for a panel scan.
            meta_rows.append(
                [
                    "Entities",
                    f"{meta.get('n_groups')} (grouped by '{meta.get('group_col')}')",
                ]
            )
        _kv_table(
            fig,
            [0.08, 0.74, 0.46, 0.14],
            ["Field", "Value"],
            meta_rows,
            col_widths=[0.45, 0.55],
        )

        if score is not None:
            fig.text(0.60, 0.90, "Data Health Score", fontsize=12, weight="bold")
            fig.text(0.60, 0.855, f"{score:.0f}% Clean", fontsize=22, weight="bold")
            fig.text(0.60, 0.832, f"Status: {_status(score)}", fontsize=10)
            if after is not None:
                delta = after - score
                sign = "+" if delta >= 0 else ""
                _kv_table(
                    fig,
                    [0.60, 0.745, 0.32, 0.06],
                    ["Metric", "Value"],
                    [
                        ["Before", f"{score:.0f}%"],
                        ["After fixes", f"{after:.0f}%"],
                        ["Change", f"{sign}{delta:.1f} pts"],
                    ],
                    col_widths=[0.6, 0.4],
                )

        # Flowing sections
        y = 0.70
        if leak_cols:
            fig.text(0.08, y, "Critical: Target Leakage", fontsize=12, weight="bold")
            y -= 0.026
            fig.text(
                0.08,
                y,
                "Exclude these columns before modeling: " + ", ".join(leak_cols),
                fontsize=9.5,
            )
            y -= 0.038

        fig.text(0.08, y, "Executive Summary", fontsize=12, weight="bold")
        y -= 0.026
        for line in _executive_summary(report, df):
            fig.text(0.10, y, f"- {line}", fontsize=9.5)
            y -= 0.022

        fig.text(
            0.08,
            y,
            f"Critical: {len(report.critical)}    "
            f"Warnings: {len(report.warnings)}    "
            f"Info: {len(report.info)}",
            fontsize=9.5,
            weight="bold",
        )
        y -= 0.04

        # Issues table: on this page if it fits, otherwise continuation pages.
        #
        # Panel scans use the prevalence view (one row per finding, with how
        # many entities it hits) instead of report.all_issues. A 500-entity
        # panel can raise tens of thousands of issues -- report.prevalence()'s
        # own docstring cites exactly that -- and report.all_issues has no
        # entity label on each row at all, so dumping it here would produce a
        # PDF with hundreds of continuation pages of visually-identical,
        # unlabeled rows for a single systemic finding. report.summary()'s
        # CLI output already avoids this via the same prevalence data.
        if report.is_panel:
            heading = "Findings by Prevalence"
            continued_heading = "Findings by Prevalence (continued)"
            table_fn = _prevalence_table
            rows_to_render = report.prevalence()
        else:
            heading = "Detected Issues"
            continued_heading = "Detected Issues (continued)"
            table_fn = _issues_table
            rows_to_render = report.all_issues

        fig.text(0.08, y, heading, fontsize=14, weight="bold")
        y -= 0.022
        if report.is_panel:
            fig.text(
                0.08,
                y,
                "A finding at 100% is systemic - suspect the pipeline, not the "
                "entities. Full per-entity detail is in the JSON export.",
                fontsize=8,
                style="italic",
            )
            y -= 0.018
            if include_correlation_heatmap and meta.get("target") is not None:
                fig.text(
                    0.08,
                    y,
                    "Lead/lag correlation heatmap is not yet available for panel "
                    "scans (per-entity vs. pooled aggregation is undecided).",
                    fontsize=8,
                    style="italic",
                )
                y -= 0.018
        cap = _capacity(y)
        table_fn(fig, 0.08, y, 0.84, rows_to_render[:cap])
        pdf.savefig(fig)
        plt.close(fig)

        rest = rows_to_render[cap:]
        per_page = _capacity(0.92)
        for start in range(0, len(rest), per_page):
            chunk = rest[start : start + per_page]
            fig = plt.figure(figsize=_A4)
            fig.text(0.08, 0.95, continued_heading, fontsize=15, weight="bold")
            table_fn(fig, 0.08, 0.92, 0.84, chunk)
            pdf.savefig(fig)
            plt.close(fig)

        # Lead/lag cross-correlation heatmap: its own page, at the end, and
        # only when there's a target to correlate against, a df to compute it
        # from, and this isn't a panel report (see export_pdf's docstring).
        target = meta.get("target")
        if (
            include_correlation_heatmap
            and df is not None
            and target is not None
            and not report.is_panel
            and target in df.columns
        ):
            from tsauditor.leakage.correlation import lag_correlation_matrix

            try:
                matrix_df = lag_correlation_matrix(
                    df, target=target, max_lag=heatmap_max_lag
                )
            except ValueError:
                # Same failure modes lag_correlation_matrix's own validation
                # already raises for (e.g. a non-binary categorical target
                # slipping through): a PDF export must not crash over a
                # bonus page when the rest of the report rendered fine.
                matrix_df = None

            if matrix_df is not None and not matrix_df.empty and matrix_df.notna().any().any():
                heat_fig = _correlation_heatmap_fig(
                    plt, matrix_df, target, heatmap_top_n
                )
                pdf.savefig(heat_fig)
                plt.close(heat_fig)

        info = pdf.infodict()
        info["Title"] = title
        info["Author"] = "tsauditor"
        info["Subject"] = "Time-series data health and leakage audit"
        info["Keywords"] = "time-series data-quality leakage audit health-score"

    return path
