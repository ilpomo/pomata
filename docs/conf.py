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
    "sphinx_copybutton",  # copy button on code blocks; strips the >>> / ... doctest prompts automatically
]

# Markdown niceties: ::: fenced admonitions, definition lists, and pretty arrows/dashes.
myst_enable_extensions = ["colon_fence", "deflist", "smartquotes"]
myst_heading_anchors = 3

# The copy button strips the doctest prompts and keeps only the input lines, so a copied example pastes straight into
# a REPL or script; the regex covers both the ``>>>`` and the ``...`` continuation prompt.
copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True

# Deterministic Polars table rendering so DataFrame reprs in doctests do not drift with terminal width.
doctest_global_setup = "import polars as pl\npl.Config.set_tbl_width_chars(120)"

# Members render in ``__all__`` order — alphabetical, since RUF022 keeps every ``__all__`` sorted ("bysource" is
# inert when a module defines ``__all__``; the thematic grouping lives in the families pages instead). Signatures in
# the heading, hints kept in the signature (the Args prose carries meaning, the signature carries types).
autodoc_member_order = "bysource"

# Render every signature multi-line -- one parameter per line with a trailing comma -- so the busy OHLCV and
# keyword-only signatures stay readable. The threshold of 1 forces the wrap for all signatures, not just the long ones.
python_maximum_signature_line_length = 1

napoleon_google_docstring = True
napoleon_numpy_docstring = False

intersphinx_mapping = {"python": ("https://docs.python.org/3", None)}

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Link checking -----------------------------------------------------------

# The ``linkcheck`` builder verifies every external reference resolves. It runs out of band -- on the nightly Docs
# schedule and on demand, never on the merge path (see .github/workflows/docs.yml) -- because a dead upstream link is a
# maintenance signal, not a reason to redden a merge (mirroring the Pages deploy decoupling). Anchors are checked, so a
# ``#section`` that a page later drops is caught.
linkcheck_anchors = True
linkcheck_retries = 2
linkcheck_timeout = 30
# DOIs are permanent by design (the DOI system guarantees the redirect, not the publisher URL), and the publisher target
# frequently answers a bot with 403 -- exactly the anti-bot false positive linkcheck cannot distinguish from real
# breakage. Each DOI is verified once when the reference is added; thereafter it is trusted, not re-fetched.
linkcheck_ignore = [r"https://doi\.org/.*"]

# -- HTML output -------------------------------------------------------------

html_theme = "furo"
# Just the name in the sidebar: the docs deploy from main (ahead of the latest tag), so the hatch-vcs version would
# render as a noisy dev string (e.g. 0.1.2.dev3+g...). The exact version lives on PyPI and in the API reference.
html_title = "pomata"

# One small override (loaded after the theme): align a multi-line signature's closing parenthesis with the name.
html_static_path = ["_static"]
html_css_files = ["custom.css"]
