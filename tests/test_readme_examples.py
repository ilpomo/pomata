"""
Executes the README's python fences, in order, as one script: the page shows a cumulative pipeline (the loaded OHLCV
frame, the pnl chain built on it, the final report), so the fences run concatenated in a single namespace, from the
repository root (the parquet fixture they read ships with the docs). A fence that stops parsing, an import that
breaks, or a renamed column now reddens the build instead of rotting on the front page — and every printed table is
compared against the live repr of the value its fence displays, so a drifted number reddens it too.
"""

import ast
import re
import runpy
from pathlib import Path

import polars as pl
import pytest

_ROOT = Path(__file__).parent.parent
_README = (_ROOT / "README.md").read_text(encoding="utf-8")
_FENCES: list[str] = re.findall(r"```python\n(.*?)```", _README, flags=re.DOTALL)
_TABLES: list[str] = re.findall(r"```python\n.*?```\n\n```text\n(.*?)```", _README, flags=re.DOTALL)


def _display_script() -> str:
    """Concatenates the fences, capturing each fence's displayed value into a ``_shown_<i>`` binding."""
    pieces: list[str] = []
    for index, fence in enumerate(_FENCES):
        last = ast.parse(fence).body[-1]
        if isinstance(last, ast.Expr):
            lines = fence.splitlines(keepends=True)
            offset = sum(len(line) for line in lines[: last.lineno - 1])
            piece = f"{fence[:offset]}_shown_{index} = {fence[offset:]}"
        elif isinstance(last, ast.Assign) and isinstance(last.targets[0], ast.Name):
            piece = f"{fence}\n_shown_{index} = {last.targets[0].id}\n"
        else:
            piece = fence
        pieces.append(piece)
    return "\n\n".join(pieces)


def test_readme_carries_the_expected_pipelines() -> None:
    """Verifies the fence count, so a new example is consciously added to this sweep when the page grows one."""
    assert len(_FENCES) == 5
    assert len(_TABLES) == len(_FENCES), "every python fence pairs with one printed table"


def test_readme_pipelines_execute(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies the fences run end to end and leave the names the page builds on (the frame, the pnl, the report)."""
    script = tmp_path / "readme_pipelines.py"
    script.write_text("\n\n".join(_FENCES), encoding="utf-8")
    monkeypatch.chdir(_ROOT)
    namespace = runpy.run_path(str(script))
    for name in ("ohlcv", "pnl", "report"):
        value = namespace.get(name)
        assert isinstance(value, pl.DataFrame), f"the README pipeline leaves no {name!r} frame"
        assert value.height > 0, f"{name!r} is empty"


def test_readme_printed_tables_match(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies each table the page prints equals the live repr of the value its fence displays."""
    script = tmp_path / "readme_tables.py"
    script.write_text(_display_script(), encoding="utf-8")
    monkeypatch.chdir(_ROOT)
    namespace = runpy.run_path(str(script))
    for index, table in enumerate(_TABLES):
        assert str(namespace[f"_shown_{index}"]) == table.rstrip("\n"), f"printed table {index} drifted from the page"
