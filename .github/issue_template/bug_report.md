---
name: Bug report
about: Something in tsauditor isn't behaving as documented
title: "[BUG] "
labels: bug
assignees: imann128
---

## Describe the bug

<!-- A clear description of what's wrong. -->

## Minimal reproduction

<!--
A short, self-contained snippet that reproduces the issue. Synthetic data
is fine and preferred over pasting a large real dataset — the smaller the
reproduction, the faster this can be diagnosed.

```python
import pandas as pd
import tsauditor as tsa

df = ...  # smallest DataFrame that triggers the bug
report = tsa.scan(df, target="...", domain="...")
```
-->

## Expected behavior

<!-- What you expected to happen. -->

## Actual behavior

<!-- What actually happened. Paste the full error traceback if there is one. -->

## Environment

- `tsauditor` version: <!-- python -c "import tsauditor; print(tsauditor.__version__)" -->
- Python version: <!-- python --version -->
- OS: <!-- Windows / Linux / macOS, and version if relevant -->
- Installed via: <!-- pip install tsauditor / editable dev install -->

## Additional context

<!-- Anything else worth knowing — does this happen on every run, only with
certain domains, only above a certain dataset size, etc. -->
