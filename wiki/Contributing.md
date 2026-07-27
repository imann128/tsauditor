# Contributing

**The canonical guide is [`CONTRIBUTING.md`](https://github.com/imann128/tsauditor/blob/main/CONTRIBUTING.md) in the repository.** Read that first — it covers the branch workflow, PR description requirements, and how to report bugs.

This page adds what that file does not cover: what a new *detector* needs, and how to keep this wiki accurate.

---

## The short version

```bash
git clone https://github.com/imann128/tsauditor.git
cd tsauditor
pip install -e ".[dev]"

git checkout -b feat/short-description

pytest -q
ruff check .
ruff format --check .
```

Branch prefixes: `fix/`, `feat/`, `docs/`, `test/`.

CI runs the suite across **Python 3.9–3.14 on Linux, Windows, and macOS** — 18 combinations — plus lint and format checks. `main` is branch-protected; every check must be green before merge.

---

## English only

All code, comments, docstrings, variable names, commit messages, and PR descriptions must be in English. Docstrings are rendered to users, and the project has contributors and users across several countries.

If English is not your first language, that is completely fine — write plainly and do not worry about polish. Clear, simple English is better than elaborate English. A reviewer will help with wording.

---

## What a new detector needs

The `leakage/` module is the project's core contribution and is held to a higher bar. Open an issue or discussion before writing code for it.

**1. Signature.** `(df, ..., domain=None) -> List[Issue]`, in the module matching its prefix.

**2. Fail softly on bad columns.** Skip degenerate input — constant columns, too few observations, all-NaN — by `continue`, not by raising. One odd column must never abort someone's entire scan. Raise only for caller mistakes: a missing target, a non-numeric declared column.

**3. Populate `evidence` properly.** This is the requirement that matters most.

`evidence` must contain the numbers behind the decision, **including the threshold applied**:

```python
evidence={
    "metric": "auc",
    "auc": round(float(auc), 4),
    "separation": round(float(score), 4),
    "threshold": threshold,          # <- the user must see what it was compared against
    "target_type": target_type,
    "n_obs": int(len(pair)),
}
```

A detector that reports a verdict without its reasoning cannot be trusted, debugged, or tuned. It is an opinion, not a measurement. Every existing detector meets this bar; keep it that way.

**4. Add a suggestion template** to `tsauditor/report/remediation.py`. Without one, your code silently falls back to `"Review this issue before using the data for modeling."`

```python
"XXX001": (
    "Do the specific thing to {target} because of {your_evidence_key}."
),
```

Templates may reference `{target}` (rendered as `"column 'X'"` or `"the dataset"`) and any key in your evidence dict.

**5. Wire it into `scanner.py`** behind the appropriate `run_*` flag. If your check needs metadata the user must supply — like `available_at` or `constraints` — make it opt-in and default to off. **Never guess.**

**6. Justify your thresholds.** Every threshold in this library has a documented reason: fat tails for `finance`, weekend closures for the 5-day gap, the √(2/π) ceiling for using AUC. A number with no rationale will be questioned in review.

**7. Add tests, including a negative case.** A test that your check fires is half a test. Add one proving it does *not* fire on clean data. Most false-positive bugs in detection code are found this way.

**8. Update the docs.** A section on the relevant detector page, plus a row in [Issue Code Reference](Issue-code-reference).

---

## Contributing to this wiki

The wiki lives in `wiki/` in the main repository and is copied to GitHub Wiki. Edit the files in `wiki/` and open a PR — do not edit the GitHub Wiki directly, or your change will be overwritten.

### Every example must be executed

Do not write an example you have not run. Paste the **real** output, not what you expect the output to be. Every code block in this wiki was executed against the installed package before being committed.

This is not pedantry. Documentation examples get copied into real projects. An example that does not run is worse than no example, because it costs the reader time before they realise it is wrong.

### Every detector page follows the same template

1. What it detects — one plain sentence
2. Signature — every parameter, type, default, and effect
3. How it works — the algorithm, and why *this* method rather than the obvious alternative
4. Issue codes and evidence — every evidence key documented
5. When it does not fire — the skip conditions
6. Worked example — runnable, with real output
7. **Limitations and false positives** — mandatory

Section 7 is not optional. Users need to know when a check is wrong, and a tool that hides its failure modes is worse than no tool. If you cannot think of a limitation, you have not thought hard enough about your check.

### Explain before you assume

Write for someone competent in Python who may not know the statistics. Explain AUC, ADF, and point-biserial correlation in plain language before using them. Assume no background in time-series econometrics.

---

## Good first contributions

**Documentation** — if something on this wiki is unclear or wrong, fix it. You do not need permission.

**Test coverage for negative cases** — pick a detector and add a test proving it stays quiet on clean data.

**Consolidate `_encode_target`** — it exists three times, in `leakage/correlation.py`, `leakage/temporal.py`, and inline in `leakage/equivalence.py`. Moving it to one shared helper is a clean, self-contained PR.

See [Internals — Known rough edges](Internals#known-rough-edges) for more.

### The threshold convention

When you add a parameter that also has a `domain` preset, follow the established pattern: default the argument to `None`, and consult `domain` only when it *is* `None`.

```python
def audit_something(df, threshold: float = None, domain: str = None):
    if threshold is None:
        threshold = {"finance": 5.0, "sensor": 3.5}.get(domain, 4.0)
```

Use `is None`, never `threshold or default` — the `or` idiom treats a deliberate `0` as "unset". `tests/test_threshold_resolution.py` pins this across all three detectors that take both.

---

## Things that will be asked to change

**A `requirements.txt`.** Dependencies live in `pyproject.toml` and are managed by Dependabot. A second source of truth will drift, and exact pins in a library cause resolver conflicts downstream. Use `pip install -e ".[dev]"`.

**Machine-specific paths.** `sys.path.insert(0, r"C:\Users\...")` will not work on any other machine. Install the package in editable mode and import it normally.

**Test files outside `tests/`.** `testpaths = ["tests"]` means pytest will not collect them, and a test file inside the package directory ships in the wheel.

**Tests with no assertions.** A script that prints is a manual check, not a test. Use `assert`.

**Non-English identifiers or commit messages.** See above.

**New hardcoded thresholds without a domain-aware default** — this is on the PR checklist.
