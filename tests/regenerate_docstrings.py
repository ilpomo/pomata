"""
Regenerate (or, with ``--check``, verify) every public function's docstring tail from its declaration.

The head of a docstring — the summary and the LaTeX formula, everything above the ``Args:`` line — is human-authored
and preserved untouched. Everything from ``Args:`` to the closing quotes is replaced by
:func:`tests.support.docstring.tail_for`, so the docstring becomes a function of the declaration.

Usage::

    uv run python tests/regenerate_docstrings.py            # --check (the default): compare, never write
    uv run python tests/regenerate_docstrings.py --check     # explicit compare
    uv run python tests/regenerate_docstrings.py --write      # rewrite the source tails in place
    uv run python tests/regenerate_docstrings.py --diff       # --check plus a unified diff per mismatch

The default is ``--check`` so an accidental run never touches ``src/``. ``--check`` exits non-zero when any tail
differs from what the generator produces; while the declaration does not yet carry every per-function prose field, a
difference is expected and is the input to the parity report, not a failure of the source.
"""

from __future__ import annotations

import argparse
import ast
import difflib
import sys
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:  # so ``python tests/regenerate_docstrings.py`` finds the ``tests`` package
    sys.path.insert(0, str(_REPO_ROOT))

from tests.all_declarations import ALL_DECLARATIONS  # importing runs every ``suite_*`` registration
from tests.support.declaration import Declaration
from tests.support.docstring import tail_for


@dataclass(frozen=True)
class DocSpan:
    """The located docstring of one function: its source path, the whole file's lines, and the tail's line span."""

    path: Path
    lines: list[str]
    tail_start: int  # index of the ``Args:`` line (0-based)
    tail_stop: int  # index of the closing-quotes line (0-based, exclusive of the tail)


def _source_path(declaration: Declaration) -> Path:
    """The ``src/`` file the function's factory is defined in."""
    module_path = str(declaration.factory.__module__)
    return _REPO_ROOT / "src" / (module_path.replace(".", "/") + ".py")


def locate(declaration: Declaration) -> DocSpan:
    """Find the function's docstring and the ``Args:``-to-closing-quotes span within its source file."""
    path = _source_path(declaration)
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    lines = source.splitlines()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == declaration.name:
            expr = node.body[0]
            if not (isinstance(expr, ast.Expr) and isinstance(expr.value, ast.Constant)):
                break
            doc_start = expr.value.lineno - 1  # the ``r"""`` line
            doc_stop = expr.value.end_lineno  # exclusive; the line after the closing ``"""``
            assert doc_stop is not None
            for i in range(doc_start, doc_stop):
                if lines[i].strip() == "Args:":
                    return DocSpan(path, lines, tail_start=i, tail_stop=doc_stop - 1)
            break
    msg = f"{declaration.name}: could not locate an Args: docstring tail in {path}"
    raise RuntimeError(msg)


def current_tail(span: DocSpan) -> list[str]:
    """The source lines of the current tail (``Args:`` through the last content line, closing quotes excluded)."""
    return span.lines[span.tail_start : span.tail_stop]


def rewrite(span: DocSpan, generated: str) -> str:
    """The full source text with the tail replaced by ``generated`` (the head and the closing quotes preserved)."""
    head = span.lines[: span.tail_start]
    closing = span.lines[span.tail_stop :]
    return "\n".join([*head, *generated.splitlines(), *closing]) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Compare (or rewrite) every tail; return an exit code (non-zero when a tail differs under ``--check``)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="compare only, never write (the default)")
    parser.add_argument("--write", action="store_true", help="rewrite the source tails in place")
    parser.add_argument("--diff", action="store_true", help="under --check, print a unified diff per mismatch")
    args = parser.parse_args(argv)

    mismatches = 0
    total = 0
    for declaration in ALL_DECLARATIONS:
        total += 1
        span = locate(declaration)
        generated = tail_for(declaration)
        if current_tail(span) == generated.splitlines():
            continue
        mismatches += 1
        if args.write:
            span.path.write_text(rewrite(span, generated), encoding="utf-8")
        elif args.diff:
            diff = difflib.unified_diff(
                current_tail(span),
                generated.splitlines(),
                fromfile=f"{declaration.name} (source)",
                tofile=f"{declaration.name} (generated)",
                lineterm="",
            )
            print("\n".join(diff))

    verb = "rewrote" if args.write else "differ from the generator"
    print(f"{mismatches}/{total} tails {verb}.")
    return 1 if mismatches and not args.write else 0


if __name__ == "__main__":
    sys.exit(main())
