"""
The round-trip guard: every public docstring tail is byte-exact with what its declaration generates.

The head of a docstring — the summary and the LaTeX formula, everything above ``Args:`` — is human-authored; the tail
(``Args:`` through the closing quotes) is a pure function of the declaration, produced by
:func:`tests.support.docstring.tail_for`. This guard reads each source tail and asserts it equals that generator output
line-for-line, so a hand-edited docstring or a changed declaration prose field that was never regenerated fails here
instead of drifting silently. It is the source-and-docs complement of the ladder: the ladder proves the declaration
against the code, this proves the docstring against the declaration. Neither ruff's pydocstyle checks nor the doctest
gate can see this coupling — the tail can be perfectly valid prose and still not be what the declaration says — so it
is proven from the source.

To repair a red case, regenerate the tails from the declarations::

    uv run python tests/regenerate_docstrings.py --write
"""

from __future__ import annotations

import difflib

import pytest

import tests.all_declarations as _registered
from tests.regenerate_docstrings import current_tail, locate
from tests.support.docstring import tail_for
from tests.support.registry import registry_all

# ``all_declarations`` is imported only to run its registration side effects; nothing is referenced from it directly.
del _registered

_DECLS = {declaration.name: declaration for declaration in registry_all()}
_NAMES = sorted(_DECLS)


@pytest.mark.parametrize("name", _NAMES)
def test_docstring_tail_is_generated(name: str) -> None:
    """The source docstring tail equals ``tail_for(declaration)`` byte-for-byte."""
    declaration = _DECLS[name]
    source = current_tail(locate(declaration))
    generated = tail_for(declaration).splitlines()
    if source != generated:
        diff = "\n".join(
            difflib.unified_diff(
                source, generated, fromfile=f"{name} (source)", tofile=f"{name} (generated)", lineterm=""
            )
        )
        pytest.fail(
            f"{name}: the docstring tail is not what the declaration generates. Either the source was hand-edited or a "
            f"declaration prose field changed without regenerating — run "
            f"`uv run python tests/regenerate_docstrings.py --write`.\n{diff}"
        )
