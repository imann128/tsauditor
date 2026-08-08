"""Sphinx configuration for tsauditor's API reference site."""

from __future__ import annotations

import sys
import os
from pathlib import Path

html_baseurl = os.environ.get("READTHEDOCS_CANONICAL_URL", "")

# Make the repo's tsauditor/ package importable without installing it.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import tsauditor  # noqa: E402

project = "tsauditor"
copyright = "2026, Iman"
author = "Iman"
release = tsauditor.__version__
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",  # reads Google/NumPy-style docstrings
    "sphinx.ext.viewcode",  # links each doc entry to its source
    "sphinx.ext.intersphinx",
    "sphinx.ext.doctest",
]

# Not currently used: every function is documented directly via
# autofunction:: on its category page (api/leakage.rst etc.) instead of a
# separate autosummary-generated stub page, since a category page groups
# related issue codes (e.g. all of LEK001-005) in one place. Left in the
# extensions list in case that changes later.
autosummary_generate = False

# Napoleon is configured for NumPy style specifically, since that's the
# format every docstring in this codebase already uses (Parameters /
# Returns / Examples sections) -- confirmed against scan(), fix(), and
# GuardReport before this was set up, not assumed.
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_include_init_with_doc = False

autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_title = f"tsauditor {release} documentation"

# Fail the build on a broken cross-reference or import rather than silently
# emitting a warning that's easy to miss in CI logs.
nitpicky = False  # left off for now: numpydoc type strings (e.g. "Optional[str]")
# aren't all valid Sphinx xref targets and would otherwise flood the build
# with false-positive warnings unrelated to real doc problems.
