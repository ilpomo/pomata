"""Sphinx configuration for the pomata documentation site.

The API reference is generated from the package's docstrings (Google style with embedded reStructuredText math and
cross-references), so the site stays in lockstep with the code: there is nothing to hand-maintain per function.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

# -- Project information -----------------------------------------------------

project = "pomata"
author = "Thomas Cercato"
project_copyright = "2026, Thomas Cercato"

try:
    release = _version("pomata")
except PackageNotFoundError:  # building from a source tree that was never installed
    release = "0.0.0"
version = release

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_parser",  # author the prose pages in Markdown
    "sphinx.ext.autodoc",  # pull docstrings from the package
    "sphinx.ext.napoleon",  # parse the Google-style sections
    "sphinx.ext.intersphinx",  # link out to the Python standard library
    "sphinx.ext.mathjax",  # render the ``.. math::`` formula blocks
    "sphinx.ext.doctest",  # run the prose-page examples in CI, like the docstrings
    "sphinx.ext.viewcode",  # add a "source" link next to each object
]

# Markdown niceties: ::: fenced admonitions, definition lists, and pretty arrows/dashes.
myst_enable_extensions = ["colon_fence", "deflist", "smartquotes"]
myst_heading_anchors = 3

# Deterministic Polars table rendering so DataFrame reprs in doctests do not drift with terminal width.
doctest_global_setup = "import polars as pl\npl.Config.set_tbl_width_chars(120)"

# Document members in source order, signatures in the heading, hints kept in the signature (the Args prose carries
# meaning, the signature carries types).
autodoc_member_order = "bysource"

# Render every signature multi-line -- one parameter per line with a trailing comma -- so the busy OHLCV and
# keyword-only signatures stay readable. The threshold of 1 forces the wrap for all signatures, not just the long ones.
python_maximum_signature_line_length = 1

napoleon_google_docstring = True
napoleon_numpy_docstring = False

intersphinx_mapping = {"python": ("https://docs.python.org/3", None)}

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- HTML output -------------------------------------------------------------

html_theme = "furo"
html_title = f"pomata {release}"
