"""
Package-level import smoke test.
"""

import importlib
import importlib.metadata

import pytest

import pomata


def test_version_is_exposed() -> None:
    """
    ``pomata.__version__`` is a non-empty string resolved from the installed distribution metadata.
    """
    assert isinstance(pomata.__version__, str)
    assert pomata.__version__


def test_version_falls_back_when_not_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    When the distribution metadata is absent (an uninstalled source tree), ``__version__`` degrades to ``"0.0.0"``
    rather than raising at import time.
    """

    def _raise(distribution_name: str) -> str:
        raise importlib.metadata.PackageNotFoundError(distribution_name)

    monkeypatch.setattr(importlib.metadata, "version", _raise)
    try:
        importlib.reload(pomata)
        assert pomata.__version__ == "0.0.0"
    finally:
        monkeypatch.undo()
        importlib.reload(pomata)  # restore the real, installed version for the rest of the session
