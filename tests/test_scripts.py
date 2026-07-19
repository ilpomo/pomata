"""
The ``scripts/`` smoke sweep — every project import a dev script states must resolve against the live modules.

``scripts/`` sits outside every lint and type gate by design (not shipped, not collected), yet ``docs/correctness.md``
publicly invites readers to run the regenerators, so an import rotting against a renamed export would break that
invitation invisibly. The sweep resolves each ``from X import name`` naming a project module against the real module
surface via AST — without executing a script, and without requiring the optional ``talib`` dependency.
"""

import ast
import importlib
from pathlib import Path

import pytest

_SCRIPTS = sorted(Path("scripts").glob("*.py"))
_PROJECT_PREFIXES = ("pomata", "tests")


@pytest.mark.parametrize("script", _SCRIPTS, ids=lambda path: path.stem)
def test_script_imports_resolve(script: Path) -> None:
    """Every ``from X import name`` naming a project module resolves to a real attribute of ``X``."""
    tree = ast.parse(script.read_text(encoding="utf-8"))
    missing: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.module is None or node.level:
            continue
        if not node.module.startswith(_PROJECT_PREFIXES):
            continue
        module = importlib.import_module(node.module)
        missing += [f"{node.module}.{alias.name}" for alias in node.names if not hasattr(module, alias.name)]
    assert not missing, f"{script.name}: imports that do not resolve: {missing}"
