"""
Builder for whats_new_0_4_0/whats_new_0_4_0.ipynb.

0.4.0 added two new issue codes (PRF007, PNL004), let fix() accept
available_at= and constraints= so LEK004/VAL001/VAL002 can run in a one-shot
repair, and changed LEK002's default threshold in a way that silently changes
results for anyone already relying on the old default. None of that is
covered by the other notebooks in examples/, which predate this release, so
someone upgrading has no single place that shows what's new and how to use
it. This notebook is that place.

Run:  python examples/whats_new_0_4_0/build_whats_new_notebook.py
It writes the .ipynb next to this file and executes it to confirm it is clean.
"""

from __future__ import annotations

import pathlib

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "whats_new_0_4_0.ipynb"


def md(text: str):
    return new_markdown_cell(text)


def code(text: str):
    return new_code_cell(text)


cells = [
    md(
        "# What's new in tsauditor 0.4.0\n"
        "\n"
        "Four changes since 0.3.0, each with its own section below:\n"
        "\n"
        "1. **PRF007** — infinite values (`inf` / `-inf`) are now detected and repaired\n"
        "2. **PNL004** — panel rows with a null entity id are now reported instead of\n"
        "   silently receiving zero checks\n"
        "3. **`fix()` accepts `available_at=` and `constraints=`** — LEK004 and\n"
        "   VAL001/VAL002 can now run as part of a one-shot repair\n"
        "4. **LEK002's default threshold changed** — this is a **behaviour change**:\n"
        "   some columns flagged under 0.3.0 will no longer be flagged\n"
        "\n"
        "If you're new to tsauditor entirely, start with\n"
        "[`examples/getting_started/`](../getting_started) instead — this notebook\n"
        "assumes you already know `scan()` / `apply_fixes()` / `fix()`.\n"
        "\n"
        "Install or upgrade: `pip install -U tsauditor`"
    ),
    md(
        "---\n"
        "\n"
        "## 1. PRF007 — infinite values\n"
        "\n"
        "Before 0.4.0, `inf` and `-inf` were invisible to the whole library.\n"
        "`isna()` is `False` for an infinity, so the missing-data checks never saw one,\n"
        "and every anomaly/leakage detector quietly replaced it with `NaN` on its own\n"
        "working copy so its arithmetic wouldn't break. You could run `scan()`, see\n"
        "nothing relevant, run `fix()`, and still hand infinities to your model.\n"
        "\n"
        "A feature built from a ratio is the realistic way this happens — a\n"
        "denominator of zero on some rows is common in ordinary feature engineering,\n"
        "not a data-entry mistake."
    ),
    code(
        "import numpy as np\n"
        "import pandas as pd\n"
        "import tsauditor as tsa\n"
        "\n"
        "idx = pd.bdate_range('2024-01-02', periods=100)\n"
        "rng = np.random.default_rng(0)\n"
        "\n"
        "price = pd.Series(100 + np.cumsum(rng.normal(0, 1, 100)), index=idx)\n"
        "change_pct = price.pct_change()\n"
        "\n"
        "# A ratio feature whose denominator hits exactly zero a few times -- this is\n"
        "# how infinities actually show up in real feature pipelines, not by accident.\n"
        "volume = rng.integers(1000, 5000, 100).astype(float)\n"
        "volume[[10, 40, 70]] = 0.0\n"
        "turnover_ratio = price / volume     # divide-by-zero -> +inf on those 3 rows\n"
        "\n"
        "df = pd.DataFrame({\n"
        "    'price': price,\n"
        "    'change_pct': change_pct,\n"
        "    'turnover_ratio': turnover_ratio,\n"
        "})\n"
        "print('inf count in turnover_ratio:', np.isinf(df['turnover_ratio']).sum())"
    ),
    code(
        "report = tsa.scan(df, run_stationarity=False)\n"
        "pd.DataFrame(report.filter(code='PRF007')[0].to_dict(), index=[0])[\n"
        "    ['code', 'severity', 'column', 'description']\n"
        "]"
    ),
    md(
        "The evidence tells you exactly how much of the column is still usable, and\n"
        "whether that's enough for the leakage checks to trust it:"
    ),
    code("report.filter(code='PRF007')[0].evidence"),
    md(
        "`below_leakage_min_obs` is the field to actually read — below 30 finite\n"
        "observations, `turnover_ratio` wouldn't just be noisier for LEK001/LEK002/\n"
        "LEK003/LEK005, it would be **skipped by them entirely**, with no message.\n"
        "\n"
        "`apply_fixes()` converts the infinities to `NaN` and imputes them, and — unlike\n"
        "every other repair — this step is unconditional. There is no reading under\n"
        "which keeping an infinity is correct, so it isn't gated by `outliers=` or\n"
        "`missing=` the way clipping and interpolation are."
    ),
    code(
        "clean = report.apply_fixes(df)\n"
        "print('inf before:', np.isinf(df['turnover_ratio']).sum())\n"
        "print('inf after :', np.isinf(clean['turnover_ratio']).sum())\n"
        "print([f for f in report.last_fixes if f['column'] == 'turnover_ratio'])"
    ),
    md(
        "---\n"
        "\n"
        "## 2. PNL004 — rows with a null entity id\n"
        "\n"
        "`scan(..., group_col=...)` partitions a panel by entity and audits each one as\n"
        "its own time series. `df.groupby(group_col)` drops rows with a null entity id\n"
        "by default — the only sound choice, since there's no entity identity to\n"
        "compare coverage or short-history against.\n"
        "\n"
        "But before 0.4.0 that meant those rows received **zero** checks and nothing\n"
        "said so: not the panel-level checks, not any per-entity check either, since\n"
        "the same `groupby` drives the per-entity loop. A null-id row looked identical\n"
        "to a clean one. PNL004 reports the count and percentage up front."
    ),
    code(
        "rng = np.random.default_rng(1)\n"
        "dates = pd.date_range('2024-01-01', periods=100, freq='D')\n"
        "parts = [\n"
        "    pd.DataFrame(\n"
        "        {'ticker': t, 'price': 100 + np.cumsum(rng.normal(0, 1, 100))},\n"
        "        index=dates,\n"
        "    )\n"
        "    for t in ('AAA', 'BBB')\n"
        "]\n"
        "panel = pd.concat(parts).sort_index()\n"
        "panel.iloc[0:20, panel.columns.get_loc('ticker')] = None   # 20 rows, no entity id\n"
        "\n"
        "panel_report = tsa.scan(panel, group_col='ticker', run_stationarity=False)\n"
        "pnl004 = panel_report.filter(code='PNL004')[0]\n"
        "print(pnl004.description)\n"
        "print()\n"
        "print(pnl004.evidence)"
    ),
    md(
        "These rows are still left unrepaired by `apply_fixes()` — there is no single\n"
        "entity's distribution to repair them from — but that skip is now explicit and\n"
        "logged, instead of an accidental side effect of `NaN != NaN`:"
    ),
    code(
        "panel_clean = panel_report.apply_fixes(panel)\n"
        "print([f for f in panel_report.last_fixes if f['action'] == 'skip_null_group_rows'])"
    ),
    md(
        "---\n"
        "\n"
        "## 3. `fix()` now accepts `available_at=` and `constraints=`\n"
        "\n"
        "LEK004 (as-of leakage) and VAL001/VAL002 (validity) are opt-in on `scan()`\n"
        "because tsauditor can't infer a release schedule or a validity bound on its\n"
        "own. Before 0.4.0, `fix()` had no parameter to pass either one through, so a\n"
        'one-shot repair silently skipped them — which reads as "nothing wrong" rather\n'
        'than "not checked." The only way to exercise them together with a repair was\n'
        "`scan()` followed by `apply_fixes()` in two separate calls."
    ),
    code(
        "cpi = pd.Series(50.0, index=idx)          # a macro series used as a feature\n"
        "spread = price * 0.001                     # bid/ask spread, must stay >= 0\n"
        "spread.iloc[10:15] = -0.5                   # a real violation: negative spread\n"
        "\n"
        "df2 = pd.DataFrame({'price': price, 'cpi': cpi, 'spread': spread})\n"
        "\n"
        "clean2, report2 = tsa.fix(\n"
        "    df2,\n"
        "    available_at={'cpi': pd.Timedelta(days=30)},   # cpi is published ~30d late\n"
        "    constraints={'bounds': {'spread': {'min': 0}}},\n"
        ")\n"
        "print('codes found:', sorted({i.code for i in report2.all_issues}))"
    ),
    md(
        "Both `LEK004` and `VAL001` are reachable now in a single call — before 0.4.0,\n"
        "`fix()` alone would never have produced either code, regardless of what was\n"
        "passed in. `spread` was deliberately given 5 negative rows above so `VAL001`\n"
        "has something real to flag rather than staying silent because nothing violated\n"
        "the bound."
    ),
    md(
        "---\n"
        "\n"
        "## 4. LEK002's default threshold changed — read this if you're upgrading\n"
        "\n"
        "`audit_correlation_leakage`'s `min_correlation` default moved from **0.1 to\n"
        "0.5**. This is not a bugfix in the sense of the other three; it's a deliberate,\n"
        "**silent behaviour change** that will make LEK002 quieter on existing code.\n"
        "\n"
        "Why: LEK002 fires when a feature's peak cross-correlation with the target lands\n"
        "at a *positive* lag. For two persistent series (a price level, a random walk,\n"
        "any slow-moving process), spurious correlation is large by construction while\n"
        "*which* lag happens to win the argmax is close to a coin flip. Measured over\n"
        "100 trials per cell on 400-point series, two **independently generated** series\n"
        "(a true negative by construction) were flagged 37-51% of the time under the old\n"
        "0.1 gate. Under 0.5, that drops to 8-13%, with no genuine leak lost in 200\n"
        "trials."
    ),
    code(
        "from tsauditor.leakage import audit_correlation_leakage\n"
        "\n"
        "# Two independently generated random walks -- no real relationship by\n"
        "# construction, so any finding here is a false positive. seed=2 is used\n"
        "# because it happens to land in the 0.1-0.5 band this section is about;\n"
        "# most seeds don't, which is itself the point (see the measured false-positive\n"
        "# rates below).\n"
        "rng2 = np.random.default_rng(2)\n"
        "idx3 = pd.bdate_range('2023-01-02', periods=400)\n"
        "independent_a = pd.Series(np.cumsum(rng2.normal(0, 1, 400)), index=idx3)\n"
        "independent_b = pd.Series(np.cumsum(rng2.normal(0, 1, 400)), index=idx3)\n"
        "df3 = pd.DataFrame({'target': independent_a, 'other_series': independent_b})\n"
        "\n"
        "old_default = audit_correlation_leakage(df3, target='target', min_correlation=0.1)\n"
        "new_default = audit_correlation_leakage(df3, target='target', min_correlation=0.5)\n"
        "print('flagged under old default (0.1):', [(i.column, i.evidence['peak_correlation']) for i in old_default])\n"
        "print('flagged under new default (0.5):', [(i.column, i.evidence['peak_correlation']) for i in new_default])"
    ),
    md(
        "The old 0.1 default flagged `other_series` as leaking, a false positive: these\n"
        "two series were generated completely independently, there is nothing for\n"
        "either to have leaked. The new 0.5 default correctly stays quiet. Whether any\n"
        "*particular* pair of independent random walks trips the old gate depends on\n"
        "the random seed, which is exactly the point: at 0.1 it was close to a coin\n"
        "flip (37-51% false-positive rate measured across seeds, see below), not a real\n"
        "threshold.\n"
        "\n"
        "**If you need the old behaviour** (e.g. you've tuned a downstream pipeline\n"
        "around it, or you'd rather over-flag than under-flag for now), pass\n"
        "`min_correlation=0.1` explicitly — the parameter didn't go away, only the\n"
        "default did:\n"
        "\n"
        "```python\n"
        "report = tsa.scan(df, target='y', run_leakage=True)   # uses the new 0.5 default\n"
        "\n"
        "# to restore 0.3.0 behaviour for this one check, call it directly:\n"
        "from tsauditor.leakage import audit_correlation_leakage\n"
        "old_style_issues = audit_correlation_leakage(df, target='y', min_correlation=0.1)\n"
        "```\n"
        "\n"
        'This affects `domain="finance"` scans most, since random-walk-like price series\n'
        "are exactly the case the old gate handled worst."
    ),
    md(
        "---\n"
        "\n"
        "## Where to go next\n"
        "\n"
        "- [`examples/getting_started/`](../getting_started) — the from-zero walkthrough,\n"
        "  if any of the API above is unfamiliar\n"
        "- [Issue Code Reference](https://github.com/imann128/tsauditor/wiki/Issue-code-reference) —\n"
        "  every code, including PRF007 and PNL004, with full evidence tables\n"
        "- [Panel Data](https://github.com/imann128/tsauditor/wiki/Panel-Data) — PNL001-PNL004\n"
        "  in full, including the other three panel-only checks\n"
        "- [CHANGELOG.md](https://github.com/imann128/tsauditor/blob/main/CHANGELOG.md) —\n"
        "  every change in this release, with the full measurement behind the LEK002\n"
        "  threshold change above\n"
        "- Confused about any of this, or need a tutorial for something not covered\n"
        "  here? [Open a GitHub Discussion](https://github.com/imann128/tsauditor/discussions) —\n"
        "  a comprehensive tutorial section is actively being built, and questions\n"
        "  directly shape what gets written next."
    ),
]

nb = new_notebook(cells=cells)
nb.metadata["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}
nb.metadata["language_info"] = {"name": "python"}

nbformat.write(nb, OUT)
print("wrote", OUT)

if __name__ == "__main__":
    from nbclient import NotebookClient

    client = NotebookClient(
        nb,
        timeout=300,
        kernel_name="python3",
        resources={"metadata": {"path": str(HERE)}},
    )
    client.execute()
    nbformat.write(nb, OUT)
    print("executed and re-wrote", OUT)
