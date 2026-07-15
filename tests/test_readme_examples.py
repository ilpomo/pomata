"""
Executes the README's python fences, in order, as one script: the page shows a cumulative pipeline (the loaded OHLCV
frame, the pnl chain built on it, the final report), so the fences run concatenated in a single namespace, from the
repository root (the parquet fixture they read ships with the docs). A fence that stops parsing, an import that
breaks, or a renamed column now reddens the build instead of rotting on the front page.
"""

import re
import runpy
from pathlib import Path

import polars as pl
import pytest

_ROOT = Path(__file__).parent.parent
_FENCES: list[str] = re.findall(
    r"```python\n(.*?)```", (_ROOT / "README.md").read_text(encoding="utf-8"), flags=re.DOTALL
)


def test_readme_carries_the_expected_pipelines() -> None:
    """Verifies the fence count, so a new example is consciously added to this sweep when the page grows one."""
    assert len(_FENCES) == 5


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
