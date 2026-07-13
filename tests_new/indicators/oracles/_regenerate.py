"""
Regenerate ``tests_new/indicators/oracles/__init__.py`` — the flat re-export of every ``*_reference`` oracle.

Each oracle lives in its own module (one per indicator); this script discovers every ``*_reference`` function across the
package and rewrites ``__init__.py`` so the imports and ``__all__`` stay in sync (and alphabetically ordered) as oracles
are added or renamed. The module docstring already at the top of ``__init__.py`` is preserved; modules whose name starts
with ``_`` (this script, ``_helpers``) are skipped. Run it after adding or renaming an oracle:

    uv run python -m tests_new.indicators.oracles._regenerate
"""

import ast
import pathlib

ORACLES = pathlib.Path(__file__).parent


def regenerate() -> int:
    """
    Rewrite the package's ``__init__.py`` from the ``*_reference`` functions found in it, returning the count.
    """
    init = ORACLES / "__init__.py"
    old = init.read_text()
    preamble = old[: old.index("from tests_new.indicators.oracles.")].rstrip() + "\n"
    entries: list[tuple[str, str]] = []
    for path in sorted(ORACLES.glob("*.py")):
        if path.name.startswith("_"):
            continue
        entries.extend(
            (path.stem, node.name)
            for node in ast.parse(path.read_text()).body
            if isinstance(node, ast.FunctionDef) and node.name.endswith("_reference")
        )
    entries.sort(key=lambda entry: entry[0])
    imports = "\n".join(f"from tests_new.indicators.oracles.{module} import {function}" for module, function in entries)
    exports = "\n".join(f'    "{function}",' for _, function in entries)
    init.write_text(preamble + "\n" + imports + "\n\n__all__ = (\n" + exports + "\n)\n")
    return len(entries)


if __name__ == "__main__":
    print(f"regenerated tests_new/indicators/oracles/__init__.py: {regenerate()} oracles")
