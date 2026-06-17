# Contributing to tsauditor

Thanks for considering a contribution. This document covers the workflow,
testing requirements, and how to find a good place to start.

## Workflow

1. **Fork** the repository and clone your fork locally.
2. **Create a branch** off `main` for your change:

   ```bash
   git checkout -b fix/short-description
   ```

   Use a prefix that matches the change: `fix/`, `feat/`, `docs/`, `test/`.

3. **Make your change.** If you're touching code in `tsauditor/`, add or update
   tests in `tests/` for it — PRs that change behavior without test coverage
   will be asked to add tests before merge.
4. **Run the full test suite** before opening a PR:

   ```bash
   pytest -q
   ```

   All tests must pass locally. CI will also run the suite across Python
   3.9–3.14 on Linux, Windows, and macOS, and a PR can't merge until those
   checks are green (this repo uses branch protection on `main`).
5. **Push your branch and open a PR** against `imann128/tsauditor:main`.
   GitHub will pre-fill the PR description from the template — fill in every
   section; don't leave it blank.
6. **Respond to review feedback.** For a solo-maintained project, expect
   review within a few days, not necessarily a few hours.

## What a good PR description includes

The PR template (auto-filled when you open a PR) asks for:

- **Summary** — one or two sentences on what the PR does and why.
- **Files changed** — a short list of which files were touched and what
  changed in each. For anything beyond a one-line fix, this saves the
  reviewer from having to reconstruct your intent from the diff alone.
- **Testing** — what you ran to confirm the change works (e.g. "added 3 new
  cases to `tests/test_missing.py`, full suite passes locally").

## Reporting bugs

Use the **Bug report** issue template. It asks for your Python version, OS,
`tsauditor` version, a minimal reproduction snippet, and what you expected
versus what happened. Issues without a reproduction snippet are much harder
to act on and may sit longer before being picked up.

## Proposing features

Use the **Feature request** issue template, or open a discussion first if
the idea is large (e.g. a new domain preset, a new leakage detection method)
before investing time in an implementation — it's better to align on
approach before writing code, especially for anything touching the
`leakage/` module, which is the project's core contribution and held to a
higher bar for statistical justification.

## Where to start

Look for issues labeled
[`good first issue`](https://github.com/imann128/tsauditor/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22).
These are scoped to be approachable without needing to understand the full
leakage-detection design. Broadly, contributions tend to fall into three
categories:

- **Documentation** — clarifying docstrings, expanding the README, adding
  worked examples beyond the OGDC case in `examples/`.
- **Edge-case tests** — every module in `profiler/`, `anomaly/`, and
  `leakage/` has a test file; if you can think of an input that isn't
  covered (empty columns, all-NaN data, extreme value ranges, single-row
  DataFrames), a PR adding that test case is genuinely useful even if it
  doesn't change any implementation code.
- **New domain presets** — `domain="finance"` and `domain="sensor"` are
  currently the only presets. Extending `tsauditor` to a new domain (e.g.
  `"iot"`, `"healthcare"`, `"crypto"`) means proposing sensible default
  thresholds for that domain and justifying them, then wiring them through
  each module that branches on `domain`.

## Code style

- Keep functions consistent with the existing pattern in each module: input
  validation first, then the detection logic, returning a `list[Issue]`.
- Match the `Issue` dataclass fields exactly (`module`, `code`, `severity`,
  `description`, `column`, `evidence`) — see `tsauditor/report/summary.py`.
- New issue codes should follow the existing prefix convention (`PRF*` for
  profiler, `ANO*` for anomaly, `LEK*` for leakage) and get a corresponding
  entry in `tsauditor/report/remediation.py` so the report can suggest an
  action for it.

  - If you add a new code, also add an entry for it in
  `tsauditor/report/remediation.py`.

  This file is a simple lookup: each issue code maps to one sentence of
  advice that gets shown to the user in the report. Open the file and you'll
  see a dictionary like this:

```python
  _REMEDIATIONS: Dict[str, str] = {
      "LEK001": (
          "Remove or reconstruct {target}: it near-deterministically "
          "reproduces the target variable and will leak. Keep it only if "
          "you can confirm it is genuinely available at prediction time."
      ),
      "ANO001": (
          "Investigate {target} for a stuck sensor or a forward-filled gap: "
          "the value repeats unchanged for an unusually long run."
      ),
      # ...
  }
```

  To add advice for a new code, add one more line in the same style:

```python
  "YOUR_NEW_CODE": (
      "A short sentence telling the user what to check or do about this issue."
  ),
```

  Use `{target}` inside your sentence — it gets automatically replaced with
  the column name (or "the dataset" if the issue isn't about one specific
  column). You don't need to write any other code; this one line is the
  whole change.

## Questions

Open a [discussion](https://github.com/imann128/tsauditor/discussions) or a
regular issue if anything here is unclear.
