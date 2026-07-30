# Getting Started

**The place to start if you've never run `tsauditor` before.** Every other
notebook in `examples/` is a focused case study that assumes you already know
the basic workflow. This one builds it from zero, on a small self-contained
synthetic dataset, so every cell is runnable with no external data files.

## What it demonstrates

`getting_started.ipynb` builds a 300-row synthetic price series with two
planted problems, then walks the full `scan()` → read → `apply_fixes()` loop:

| Injected fault | Column | Detected as | Severity |
|----------------|--------|-------------|----------|
| Same-day feature that defines the target's sign | `change_pct` | `LEK001` (target equivalence) | critical |
| Stuck sensor run (8 identical values) | `volume` | `ANO001` (stuck values) | warning |
| 5-day outage (missing values) | `volume` | `PRF002` (clustered missing) | warning |

It also includes a genuinely clean feature (`price_ma5`, a past-only rolling
mean) to show that `tsauditor` stays quiet on legitimate features built from
the same underlying series as the leaky one.

The notebook closes with a section addressing why this notebook exists at
all: a real, verified example of what happened when someone asked an LLM to
explain the `tsauditor` API from just a link to the repo. Every claim in that
section is checked against the actual library and shown side by side with
what really happens, including the exact `ValueError` the generated code
raises and why its predicted `leaky_columns()` output is wrong.

## Running it

From the repository root:

```bash
pip install -e ".[dev,examples]"
jupyter notebook examples/getting_started/getting_started.ipynb
```

Then run all cells. No external data files or network access required.

## Regenerating

```bash
python examples/getting_started/build_getting_started_notebook.py
```

Writes the notebook and executes every cell; a failure means the example is
out of sync with the library.

## Files

- `getting_started.ipynb` — the walkthrough (markdown + executed code).
- `build_getting_started_notebook.py` — the builder script that generates and
  executes it.
- `README.md` — this file.
