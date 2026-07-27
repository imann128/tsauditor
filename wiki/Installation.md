# Installation

## From PyPI

```bash
pip install tsauditor
```

Requires **Python 3.9 or newer**.

This installs the core dependencies, which are all `tsauditor` needs for a full scan:

| Package | Version range | What it is used for |
| ------- | ------------- | ------------------- |
| `pandas` | `>=1.5,<3` | The DataFrame type everything operates on |
| `numpy` | `>=1.23,<3` | Array maths in the detectors |
| `scipy` | `>=1.9,<2` | Statistical helpers |
| `statsmodels` | `>=0.13,<1` | The Augmented Dickey-Fuller test behind PRF003 |
| `rich` | `>=13.0,<16` | The formatted console table printed by `report.summary()` |

---

## Optional extras

An "extra" is an optional group of dependencies you install only if you need that feature. Install one by adding its name in square brackets:

```bash
pip install 'tsauditor[pdf]'      # PDF export via report.to_pdf()   -> adds matplotlib
pip install 'tsauditor[polars]'   # accept polars DataFrames         -> adds polars, pyarrow
```

You can combine them:

```bash
pip install 'tsauditor[pdf,polars]'
```

**Quote the whole thing.** The square brackets are special characters in `zsh` and `bash`. Without quotes you will get `zsh: no matches found`.

### What each extra unlocks

**`[pdf]`** — enables `GuardReport.to_pdf("report.pdf")`, which writes a formal, text-selectable PDF. Without it, calling `to_pdf` raises an `ImportError`. Every other output format (`summary()`, `to_json()`, `to_dict()`) works without this extra.

**`[polars]`** — lets you pass a polars DataFrame directly to `scan()`. `tsauditor` converts it to pandas at the boundary and does all its work in pandas. Because polars has no index, you **must** also pass `time_col=`:

```python
report = tsa.scan(polars_df, time_col="date")   # time_col is required for polars
```

### What needs no extra

The **TimesFM adapter** (`tsauditor.adapters.to_timesfm`) works out of the box. It never imports `timesfm` — it only returns a NumPy array, and you own the model. `tsauditor` gains no dependency from it.

---

## Development setup

```bash
git clone https://github.com/imann128/tsauditor.git
cd tsauditor
pip install -e ".[dev]"
```

The `-e` flag installs in *editable* mode: your local edits take effect immediately, with no reinstall.

The `[dev]` extra adds the full test and lint toolchain: `pytest`, `pytest-cov`, `ruff`, `matplotlib`, `polars`, `pyarrow`, and `joblib`.

Run the test suite:

```bash
pytest -q
```

Run the linter and formatter check, exactly as CI does:

```bash
ruff check .
ruff format --check .
```

### What CI enforces

Every pull request runs:

- **Lint** — `ruff check` and `ruff format --check` on Python 3.11
- **Tests** — the full `pytest` suite across Python **3.9, 3.10, 3.11, 3.12, 3.13, 3.14** on **Ubuntu, Windows, and macOS** (18 combinations)
- **Coverage** — uploaded to Codecov from the Ubuntu / Python 3.9 job

A PR must pass all of these before it can be merged.

---

## Verifying the install

```python
import tsauditor as tsa
print(tsa.__version__)
```

```
0.3.0
```

If that prints a version, you are ready for the [Quickstart](Quickstart).

---

## Troubleshooting

**`ImportError: ... requires matplotlib`** when calling `to_pdf()` — install the PDF extra: `pip install 'tsauditor[pdf]'`.

**`ValueError: polars input requires time_col=`** — polars has no index, so `tsauditor` cannot guess which column holds the timestamps. Pass `time_col="your_date_column"`.

**`ImportError: Converting a polars DataFrame requires pyarrow`** — install the polars extra rather than polars alone: `pip install 'tsauditor[polars]'`.

**`zsh: no matches found: tsauditor[pdf]`** — you forgot the quotes. Use `pip install 'tsauditor[pdf]'`.
