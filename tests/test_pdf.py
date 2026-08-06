import numpy as np
import pandas as pd
import pytest

pytest.importorskip("matplotlib")  # PDF export is gated behind the [pdf] extra

from tsauditor.report.summary import GuardReport, Issue, WARNING


def _idx(n):
    return pd.date_range("2020-01-01", periods=n, freq="D")


def _df_and_report():
    n = 120
    rng = np.random.default_rng(0)
    price = rng.normal(50, 1, n)
    price[60] = 500.0
    df = pd.DataFrame({"price": price, "clean": np.linspace(0, 5, n)}, index=_idx(n))
    df.iloc[20:30, df.columns.get_loc("price")] = np.nan
    rep = GuardReport(
        warnings=[
            Issue("anomaly", "ANO002", WARNING, "Point anomalies in 'price'.", "price"),
            Issue("profiler", "PRF002", WARNING, "Clustered NaNs in 'price'.", "price"),
        ],
        metadata={"rows": n, "columns": 2, "domain": None, "target": None},
    )
    return df, rep


def test_to_pdf_creates_valid_pdf(tmp_path):
    df, rep = _df_and_report()
    out = tmp_path / "report.pdf"
    rep.to_pdf(str(out), df=df)
    assert out.exists()
    assert out.stat().st_size > 0
    assert out.read_bytes()[:4] == b"%PDF"  # real PDF header


def test_to_pdf_with_before_after(tmp_path):
    df, rep = _df_and_report()
    fixed = rep.apply_fixes(df, outliers="clip", missing="interpolate")
    out = tmp_path / "before_after.pdf"
    rep.to_pdf(str(out), df=df, fixed_df=fixed)
    assert out.read_bytes()[:4] == b"%PDF"


def test_to_pdf_without_df_still_renders(tmp_path):
    _, rep = _df_and_report()
    out = tmp_path / "nodf.pdf"
    rep.to_pdf(str(out))  # no df -> scorecard + issues, no charts
    assert out.read_bytes()[:4] == b"%PDF"


# ── Panel reports (0.3.0) ────────────────────────────────────────────────────


def _panel_report():
    """
    A panel scan, which produces Issues carrying `.group` plus panel-level PNL
    codes. Neither existed before 0.3.0, so nothing else in this file covers
    them.
    """
    import numpy as np
    import pandas as pd

    import tsauditor as tsa

    dates = pd.date_range("2024-01-01", periods=120, freq="B")
    parts = []
    for i, ticker in enumerate(["AAA", "BBB", "CCC"]):
        rng = np.random.default_rng(i)
        price = 100 + 50 * i + np.cumsum(rng.normal(0, 1, 120))
        ret = pd.Series(price).pct_change().to_numpy()
        parts.append(
            pd.DataFrame(
                {
                    "ticker": ticker,
                    "price": price,
                    "ret": ret,
                    "direction": (ret > 0).astype(float),
                },
                index=dates,
            )
        )
    panel = pd.concat(parts).sort_index()
    report = tsa.scan(
        panel, target="direction", group_col="ticker", run_stationarity=False
    )
    return panel, report


def test_to_pdf_renders_a_panel_report(tmp_path):
    """
    Panel reports carry Issue.group and PNL* codes. The PDF exporter predates
    both, so this pins that it still renders rather than raising on the new
    fields.
    """
    panel, report = _panel_report()
    assert report.is_panel
    assert any(i.group is not None for i in report.all_issues)

    out = tmp_path / "panel.pdf"
    report.to_pdf(str(out), df=panel)
    assert out.read_bytes()[:4] == b"%PDF"
    assert out.stat().st_size > 1000


def test_to_pdf_after_score_is_panel_aware(tmp_path):
    """
    Regression (full-sweep finding). export_pdf's "after fixes" health score
    re-scanned fixed_df without group_col -- an independent copy of the same
    gap fixed in GuardReport.to_json()'s score_after. For a panel, that
    treats every entity as one interleaved series: a real outlier in a
    small-scale entity gets diluted into a global mean/std dominated by a
    much larger-scale entity and is never detected, so an intentionally
    unrepaired anomaly was misreported as "After fixes 100%".
    """
    import numpy as np
    import pandas as pd
    import tsauditor as tsa

    dates = pd.date_range("2024-01-01", periods=120, freq="B")
    rng = np.random.default_rng(0)
    aaa = rng.normal(10, 1, 120)
    aaa[20] = 30.0  # dramatic per-entity outlier, invisible against BBB's scale
    bbb = rng.normal(1000, 100, 120)
    panel = pd.concat(
        [
            pd.DataFrame({"ticker": "AAA", "price": aaa}, index=dates),
            pd.DataFrame({"ticker": "BBB", "price": bbb}, index=dates),
        ]
    ).sort_index()

    report = tsa.scan(
        panel, group_col="ticker", run_leakage=False, run_stationarity=False
    )
    fixed = report.apply_fixes(panel, outliers=None, missing="interpolate", stuck=None)
    assert report.health_score(fixed) < 100.0  # the outlier is still there

    out = tmp_path / "panel_after_score.pdf"
    report.to_pdf(str(out), df=panel, fixed_df=fixed)

    pypdf = pytest.importorskip("pypdf")
    reader = pypdf.PdfReader(str(out))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    assert "After fixes 100%" not in text


def _pdf_text(path) -> str:
    pypdf = pytest.importorskip("pypdf")
    reader = pypdf.PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def test_to_pdf_panel_overview_states_entity_count(tmp_path):
    """
    Regression. Dataset Overview only ever showed rows/columns/time
    range/frequency/domain/target -- group_col and n_groups never appeared
    anywhere in the PDF, even though report.summary()'s CLI output always
    states the entity count for a panel scan.
    """
    panel, report = _panel_report()
    out = tmp_path / "panel.pdf"
    report.to_pdf(str(out), df=panel)

    text = _pdf_text(out)
    assert "Entities" in text
    assert str(report.metadata["n_groups"]) in text
    assert report.metadata["group_col"] in text


def test_to_pdf_panel_uses_prevalence_not_raw_issue_dump(tmp_path):
    """
    Regression. A panel report's PDF used to render report.all_issues
    directly -- one row per entity per finding, with no entity label on the
    row at all (Issue.group was never included in the table columns). A
    systemic bug present in every entity of a large panel produced hundreds
    or thousands of visually-identical, unlabeled rows spread across as many
    continuation pages as it took to fit them.

    Built with a bug that hits every one of 40 entities: previously this
    meant 40+ near-identical "ANO001" rows with no way to tell them apart.
    Now it must render as a small number of prevalence rows instead
    ("Findings by Prevalence", not "Detected Issues"), one per (code,
    column), each carrying its own entity count -- so the page count stays
    small and the systemic nature of the bug is visible rather than buried
    in repetition.
    """
    import numpy as np
    import pandas as pd
    import tsauditor as tsa

    rng = np.random.default_rng(0)
    n_entities = 40
    dates = pd.date_range("2024-01-01", periods=200, freq="D")
    rows = []
    for e in range(n_entities):
        vals = rng.normal(0, 1, 200)
        vals[10:16] = 5.0  # every entity gets the same stuck run
        rows.append(pd.DataFrame({"ticker": f"E{e:02d}", "x": vals}, index=dates))
    df = pd.concat(rows).sort_index()

    report = tsa.scan(df, group_col="ticker", run_leakage=False, run_stationarity=False)
    assert len(report.all_issues) >= n_entities  # the raw, unlabeled-dump scenario

    out = tmp_path / "prevalence.pdf"
    report.to_pdf(str(out), df=df)

    pypdf = pytest.importorskip("pypdf")
    reader = pypdf.PdfReader(str(out))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)

    assert "Findings by Prevalence" in text
    assert "Detected Issues" not in text
    assert f"{n_entities}/{n_entities}" in text  # the ANO001 prevalence row
    # The old per-issue dump would have needed several continuation pages for
    # 40+ rows; the prevalence view collapses to a handful of (code, column)
    # rows and must fit in very few pages.
    assert len(reader.pages) <= 3
