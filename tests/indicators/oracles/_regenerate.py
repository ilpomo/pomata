"""
Regenerate ``tests/indicators/oracles/__init__.py`` — the flat re-export of every ``*_reference`` oracle.

Each oracle lives in its own module (one per indicator); this script discovers every ``*_reference`` function across the
package and rewrites ``__init__.py`` so the imports and ``__all__`` stay in sync (and alphabetically ordered) as oracles
are added or renamed. The module docstring already at the top of ``__init__.py`` is preserved; modules whose name starts
with ``_`` (this script, ``_helpers``) are skipped. Run it after adding or renaming an oracle:

    uv run python -m tests.indicators.oracles._regenerate
"""

import ast
import pathlib

ORACLES = pathlib.Path(__file__).parent


def regenerate() -> int:
    """
    Rewrite the package's ``__init__.py`` from the ``*_reference`` functions found in it, returning the count.
    """
    init = ORACLES / "__init__.py"
    old = init.read_text(encoding="utf-8")
    preamble = old[: old.index("from tests.indicators.oracles.")].rstrip() + "\n"
    entries: list[tuple[str, str]] = []
    for path in sorted(ORACLES.glob("*.py")):
        if path.name.startswith("_"):
            continue
        entries.extend(
            (path.stem, node.name)
            for node in ast.parse(path.read_text(encoding="utf-8")).body
            if isinstance(node, ast.FunctionDef) and node.name.endswith("_reference")
        )
    # One from-import per module with its names grouped and sorted (isort-clean), the parenthesized one-name-per-line
    # form from 120 columns up (the wrapped form is stable under ruff format, which keeps a magic trailing comma and
    # never unwraps it), and a globally function-sorted __all__ (RUF022-clean) — so the emitted file is exactly the
    # committed style and a regeneration is a no-op.
    by_module: dict[str, list[str]] = {}
    for module, function in sorted(entries):
        by_module.setdefault(module, []).append(function)
    import_lines: list[str] = []
    for module, functions in sorted(by_module.items()):
        flat = f"from tests.indicators.oracles.{module} import {', '.join(sorted(functions))}"
        if len(flat) < 120:
            import_lines.append(flat)
        else:
            wrapped = "\n".join(f"    {function}," for function in sorted(functions))
            import_lines.append(f"from tests.indicators.oracles.{module} import (\n{wrapped}\n)")
    imports = "\n".join(import_lines)
    exports = "\n".join(f'    "{function}",' for function in sorted(function for _, function in entries))
    init.write_text(preamble + "\n" + imports + "\n\n__all__ = (\n" + exports + "\n)\n")
    return len(entries)


if __name__ == "__main__":
    print(f"regenerated tests/indicators/oracles/__init__.py: {regenerate()} oracles")
