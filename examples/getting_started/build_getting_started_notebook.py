"""
Builder for getting_started/getting_started.ipynb.

The other notebooks in examples/ are each a focused case study (the OGDC leak,
validity on volume/RSI, the TimesFM adapter...). None of them is a from-zero
walkthrough for someone who has never run tsauditor before, so newcomers end up
asking an LLM to explain the API instead, and get confidently wrong answers
back (see the last section of this notebook for a real, verified example of
that happening). This notebook is the "start here" that was missing: one
self-contained synthetic dataset, the full scan -> read -> repair loop, and a
section that corrects specific hallucinated API usage against the real
library, with real output.

Run:  python examples/getting_started/build_getting_started_notebook.py
It writes the .ipynb next to this file and executes it to confirm it is clean.
"""

from __future__ import annotations

import pathlib

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "getting_started.ipynb"


def md(text: str):
    return new_markdown_cell(text)


def code(text: str):
    return new_code_cell(text)


cells = [
    md(
        "# Getting started with tsauditor\n"
        "\n"
        "**Start here if this is your first time using the library.** Every other\n"
        "notebook in `examples/` assumes you already know the basic workflow; this one\n"
        "builds it from zero, on a small synthetic dataset, so you can run every cell\n"
        "yourself and see real output.\n"
        "\n"
        "What you'll do:\n"
        "1. Build a tiny dataset with a planted leak and a planted sensor fault\n"
        "2. Run `tsa.scan()` and read the report\n"
        "3. Repair it with `apply_fixes()`\n"
        "4. See a real, verified example of an LLM getting the API wrong, and what the\n"
        "   correct usage actually is\n"
        "\n"
        "Install: `pip install tsauditor`"
    ),
    md(
        "## 1. Build a small dataset\n"
        "\n"
        "300 trading days of a synthetic price series. `direction` is the target: did\n"
        "the price go up **today**? `change_pct` is *today's* percentage price change,\n"
        "computed the obvious way. Watch what happens when both exist in the same\n"
        "frame, this is the exact bug class tsauditor was built for (it is a smaller\n"
        "version of a real leak found in a Pakistani equity dataset, see\n"
        "`examples/ogdc_leakage_case/` for the full story).\n"
        "\n"
        "`volume` has two planted faults: a sensor-style stuck run (8 identical values\n"
        "in a row) and a 5-day outage (missing values)."
    ),
    code(
        "import numpy as np\n"
        "import pandas as pd\n"
        "import tsauditor as tsa\n"
        "\n"
        "n = 300\n"
        "idx = pd.bdate_range('2024-01-02', periods=n)\n"
        "\n"
        "rng = np.random.default_rng(0)\n"
        "price = pd.Series(100 + np.cumsum(rng.normal(0, 1, n)), index=idx)\n"
        "change_pct = price.pct_change().fillna(0.0)\n"
        "direction = (change_pct > 0).astype(float)          # today's move, the target\n"
        "\n"
        "vrng = np.random.default_rng(0)                      # independent stream\n"
        "volume = vrng.integers(1000, 5000, n).astype(float)\n"
        "volume[100:108] = volume[99]                          # stuck sensor run\n"
        "volume[150:155] = np.nan                               # a 5-day outage\n"
        "\n"
        "price_ma5 = price.rolling(5, min_periods=1).mean().shift(1)   # past-only\n"
        "\n"
        "df = pd.DataFrame({\n"
        "    'price': price,\n"
        "    'change_pct': change_pct,       # <- the leak: defines direction's sign\n"
        "    'price_ma5': price_ma5,         # <- clean: uses only past data\n"
        "    'volume': volume,               # <- has a stuck run and a gap\n"
        "    'direction': direction,\n"
        "})\n"
        "df.head()"
    ),
    md(
        "## 2. Scan it\n"
        "\n"
        "One call. `target=` matters: without it, every leakage check (LEK001-LEK005)\n"
        "is **silently skipped**, no error, no warning, they simply never run. Forgetting\n"
        "this is the single most common way to get a falsely clean report."
    ),
    code(
        "report = tsa.scan(df, target='direction', domain='finance', run_stationarity=False)\n"
        "report.summary()"
    ),
    md(
        "`scan()` returns a `GuardReport`, a real Python object with structured data,\n"
        "not a block of printed text. Issues are bucketed by severity:"
    ),
    code(
        "print('critical:', len(report.critical))\n"
        "print('warnings:', len(report.warnings))\n"
        "print('info    :', len(report.info))\n"
        "print()\n"
        "print('leaky_columns():', report.leaky_columns())"
    ),
    md(
        "`leaky_columns()` is the shortlist to review or remove before training. Notice\n"
        "`price_ma5` is **not** in it, a rolling mean of *past* prices is a legitimate\n"
        "feature and tsauditor stays quiet about it, even though it is built from the\n"
        "same underlying price series as the leaky one. Only `change_pct`, which is\n"
        "mathematically tied to the target's own sign, gets flagged.\n"
        "\n"
        "Two other findings show up in the table above that are worth explaining rather\n"
        "than ignoring, since a real report will have noise like this too:\n"
        "\n"
        "- **`ANO002`/`ANO003` on `change_pct`**, a raw daily percentage-change series\n"
        "  has some genuinely large-swing days by construction. These are separate,\n"
        "  correct findings about volatility, not part of the leakage story.\n"
        "- **`ANO001` on `direction` itself**, a binary 0/1 column naturally contains\n"
        "  runs of repeated values (several up-days in a row). `ANO001` doesn't know or\n"
        "  care that this column happens to be the target, it just reports what it\n"
        "  sees. This is harmless here, but if it fired on a column you *do* intend to\n"
        "  use as a feature, it would be worth a look."
    ),
    md(
        "## 3. Read a finding properly\n"
        "\n"
        "Every `Issue` carries its reasoning, not just a verdict, `column`,\n"
        "`description`, `evidence` (the numbers behind the decision), and `suggestion`."
    ),
    code(
        "leak = report.filter(code='LEK001')[0]\n"
        "\n"
        "print('column     :', leak.column)\n"
        "print('description:', leak.description)\n"
        "print('evidence   :', leak.evidence)\n"
        "print('suggestion :', leak.suggestion)"
    ),
    md(
        "`evidence['separation']` is the AUC-based separation score: 1.0 means the\n"
        "feature perfectly determines the target's class, there is no ambiguity, this is\n"
        "not a borderline call.\n"
        "\n"
        "The sensor fault on `volume` shows up as its own, unrelated finding:"
    ),
    code(
        "for i in report.filter(column='volume'):\n"
        "    print(i.severity.upper(), i.code, '|', i.description)"
    ),
    md(
        "## 4. Repair it\n"
        "\n"
        "`apply_fixes()` never touches your original `df`, it returns an independent\n"
        "copy. The target column is never repaired (interpolating a 0/1 label would be\n"
        "meaningless), and every change is logged."
    ),
    code(
        "clean = report.apply_fixes(df)\n"
        "\n"
        "for f in report.last_fixes:\n"
        "    print(f)\n"
        "\n"
        "print()\n"
        "print('NaNs before:', int(df['volume'].isna().sum()))\n"
        "print('NaNs after :', int(clean['volume'].isna().sum()))\n"
        "print('df untouched:', df['volume'].isna().sum() == 5)"
    ),
    md(
        "`change_pct` is still there, `apply_fixes()` does not delete leaky columns\n"
        'unless you explicitly ask it to (`leakage="drop"`), because dropping a feature\n'
        "is a modeling decision, not something a data-quality tool should do silently."
    ),
    code(
        "clean_no_leak = report.apply_fixes(df, leakage='drop')\n"
        "print('columns after leakage=\"drop\":', list(clean_no_leak.columns))"
    ),
    md(
        "---\n"
        "\n"
        "## 5. A real example of an LLM getting this wrong\n"
        "\n"
        "This isn't hypothetical. Someone pasted a link to this repo into ChatGPT and\n"
        "asked it to explain how to use `tsauditor`. Here is what it produced, and what\n"
        "actually happens when you run it, verified against the real library, not\n"
        "guessed."
    ),
    md(
        "**What ChatGPT wrote:**\n"
        "\n"
        "```python\n"
        "df = pd.DataFrame({\n"
        '    "date": pd.date_range("2025-01-01", periods=10),\n'
        '    "open": [...], "high": [...], "low": [...], "close": [...], "volume": [...]\n'
        "})\n"
        'df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)\n'
        'df["future_close"] = df["close"].shift(-1)\n'
        "\n"
        'report = tsa.scan(df, target="target", domain="finance")\n'
        "```\n"
        "\n"
        "**Mistake #1: this raises, it does not scan.** `date` is a plain column, not\n"
        "the index, and `time_col` was never passed. tsauditor deliberately refuses to\n"
        "guess here, a numeric `RangeIndex` would otherwise be silently misread as\n"
        "nanosecond timestamps near 1970."
    ),
    code(
        "df_bad = pd.DataFrame({\n"
        "    'date': pd.date_range('2025-01-01', periods=10),\n"
        "    'close': [101,102,101,103,104,105,104,106,107,108],\n"
        "})\n"
        "df_bad['target'] = (df_bad['close'].shift(-1) > df_bad['close']).astype(int)\n"
        "\n"
        "try:\n"
        "    tsa.scan(df_bad, target='target', domain='finance')\n"
        "except ValueError as e:\n"
        "    print(f'{type(e).__name__}: {e}')"
    ),
    md(
        '**The fix:** pass `time_col="date"` (or set the index yourself with\n'
        '`df.set_index("date")` before calling `scan()`).\n'
        "\n"
        "**Mistake #2:** ChatGPT then claimed `report.leaky_columns()` would return\n"
        '`["future_close"]`. Even after fixing the index problem, it does not, because\n'
        "the example only has **10 rows**, and every check in tsauditor requires\n"
        "`min_obs=30` pairwise-complete observations by default before it will trust a\n"
        "score enough to report anything. Below that, checks skip the column rather\n"
        "than risk a spurious result from a handful of points."
    ),
    code(
        "df_fixed = df_bad.copy()\n"
        "df_fixed['future_close'] = df_fixed['close'].shift(-1)\n"
        "\n"
        "report_fixed = tsa.scan(df_fixed, target='target', domain='finance', time_col='date')\n"
        "print('leaky_columns():', report_fixed.leaky_columns())\n"
        "print('(empty, not [\"future_close\"], because 10 rows < min_obs=30)')"
    ),
    md(
        "**Mistake #3:** ChatGPT invented a `report.summary()` output, an ASCII checklist\n"
        'with `✓`/`⚠`/`✗` symbols, and even labeled it *"illustrative"* rather than\n'
        "admitting it had not run the code. The real format is a fixed-width report\n"
        "block with a dataset header and severity counts, shown earlier in this notebook.\n"
        "It never used checkmark symbols, and it groups by severity, not by check type.\n"
        "\n"
        "**Mistake #4:** a suggested type hint used `-> AuditReport`. The real return\n"
        "type is `GuardReport`. A class that does not exist cannot be imported, so code\n"
        "written against that hint fails immediately."
    ),
    md(
        "**The pattern behind all of this:** an LLM without the actual source in\n"
        "context will reach for what a *typical* Python data-quality library looks like\n"
        "(dtypes, null counts, a friendly checklist print), not what this one actually\n"
        "does (an object-based report, strict index handling, a `min_obs` floor against\n"
        "spurious small-sample findings). None of the individual guesses were crazy,\n"
        "they just weren't checked against the real package. The fix is the same one\n"
        "every cell in this notebook follows: run it, and paste what actually happened."
    ),
    md(
        "---\n"
        "\n"
        "## Where to go next\n"
        "\n"
        "- [`examples/ogdc_leakage_case/`](../ogdc_leakage_case) — the real leak this\n"
        "  library was built for, on real equity data, with the measured accuracy\n"
        "  collapse once it's removed\n"
        "- [`examples/sensor-example/`](../sensor-example) — structural and anomaly\n"
        "  checks on a synthetic sensor stream, plus PDF report export\n"
        "- [Quickstart](https://github.com/imann128/tsauditor/wiki/Quickstart) — every\n"
        "  parameter used above, explained in full, including `available_at=` and\n"
        "  `constraints=` for the two leakage checks that need explicit rules\n"
        "- [API Reference](https://github.com/imann128/tsauditor/wiki/API-Reference) —\n"
        "  every public function and parameter"
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
