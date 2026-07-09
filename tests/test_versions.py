"""
Support-claim drift guards: every prose claim about the Polars floor, the supported Python versions, and the
supported operating systems must match the machine-read source of truth.

``pyproject.toml`` is that source of truth: the Polars floor comes from the ``[project]`` dependencies, the Python
floor from ``requires-python``, and the supported Python list from the ``Programming Language :: Python :: 3.X``
classifiers. Prose restatements (the README badges and bullets, the docs-site pages, the contributing guide) repeat
the same numbers only as a courtesy to the reader; this module fails CI whenever one of them drifts, naming the file
and the expected value, so bumping a floor is a one-file edit plus whatever this guard then points at. The CI-matrix
coherence check reads ``.github/workflows/ci.yml``, which is not shipped in the sdist; that single test skips there.
"""

import re
import tomllib
from pathlib import Path
from typing import cast

import pytest

_ROOT = Path(__file__).parent.parent
_PYPROJECT = cast("dict[str, object]", tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8")))

_PROSE_FILES = (
    _ROOT / "README.md",
    _ROOT / "CONTRIBUTING.md",
    *sorted((_ROOT / "docs").glob("*.md")),
    *sorted((_ROOT / "docs" / "families").glob("*.md")),
)


def _nested_str(mapping: object, *keys: str) -> str:
    """Walk nested TOML tables to a string leaf, asserting each step exists."""
    node: object = mapping
    for key in keys:
        assert isinstance(node, dict), f"expected a table on the way to {keys!r}"
        node = cast("object", node[key])
    assert isinstance(node, str), f"expected a string at {keys!r}"
    return node


def _project_list(key: str) -> list[str]:
    """A list-of-strings field from the ``[project]`` table."""
    project = _PYPROJECT["project"]
    assert isinstance(project, dict)
    values = cast("object", project[key])
    assert isinstance(values, list)
    return list(cast("list[str]", values))


def _polars_floor() -> str:
    """The declared Polars floor (``X.Y.Z``), from the ``[project]`` dependencies."""
    for spec in _project_list("dependencies"):
        match = re.fullmatch(r"polars>=(\d+\.\d+\.\d+)", spec)
        if match:
            return match.group(1)
    pytest.fail("pyproject.toml declares no 'polars>=X.Y.Z' dependency")


def _python_versions() -> list[str]:
    """The supported Python versions, from the trove classifiers, ascending."""
    versions = [
        match.group(1)
        for classifier in _project_list("classifiers")
        if (match := re.fullmatch(r"Programming Language :: Python :: (3\.\d+)", classifier))
    ]
    assert versions, "pyproject.toml declares no versioned Python classifiers"
    return sorted(versions, key=lambda v: int(v.split(".")[1]))


_POLARS_FLOOR = _polars_floor()
_POLARS_FLOOR_PROSE = _POLARS_FLOOR.removesuffix(".0")  # "1.39.0" -> "1.39", the form prose and badges use
_PYTHON_VERSIONS = _python_versions()
_PYTHON_FLOOR = _PYTHON_VERSIONS[0]


def _version_claims(text: str, keyword: str, exclude: str) -> list[str]:
    """Versions stated near ``keyword`` (within 40 non-digit characters, ``exclude`` not intervening)."""
    pattern = re.compile(rf"{keyword}([^\d]{{0,40}}?)(\d+\.\d+(?:\.\d+)?)", re.IGNORECASE)
    return [match.group(2) for match in pattern.finditer(text) if exclude not in match.group(1).lower()]


def test_polars_floor_claims_match_pyproject() -> None:
    """Every Polars version stated in prose is the declared floor from pyproject.toml."""
    expected = {_POLARS_FLOOR, _POLARS_FLOOR_PROSE}
    drifted = [
        f"{path.relative_to(_ROOT)}: claims polars {claim}, pyproject.toml declares {_POLARS_FLOOR}"
        for path in _PROSE_FILES
        for claim in _version_claims(path.read_text(encoding="utf-8"), "polars", "ython")
        if claim not in expected
    ]
    assert not drifted, "\n".join(drifted)


def test_python_version_claims_match_classifiers() -> None:
    """Every Python version stated in prose is one of the classifier-supported versions."""
    drifted = [
        f"{path.relative_to(_ROOT)}: claims python {claim}, classifiers support {_PYTHON_VERSIONS}"
        for path in _PROSE_FILES
        for claim in _version_claims(path.read_text(encoding="utf-8"), "python", "olars")
        if claim not in _PYTHON_VERSIONS
    ]
    assert not drifted, "\n".join(drifted)


def test_readme_lists_every_supported_python() -> None:
    """The README badge and the Dependencies bullet spell out the full classifier list."""
    readme = (_ROOT / "README.md").read_text(encoding="utf-8")
    badge = "python-" + "%20|%20".join(_PYTHON_VERSIONS)
    bullet = ", ".join(_PYTHON_VERSIONS)
    assert badge in readme, f"README python badge must list {_PYTHON_VERSIONS} (expected '{badge}')"
    assert bullet in readme, f"README Dependencies bullet must list {_PYTHON_VERSIONS} (expected '{bullet}')"


def test_python_floor_is_coherent_across_pyproject() -> None:
    """requires-python and every tool's target pin agree with the oldest classifier version."""
    requires = _nested_str(_PYPROJECT, "project", "requires-python")
    assert requires == f">={_PYTHON_FLOOR}", f"requires-python is {requires!r}, classifiers start at {_PYTHON_FLOOR}"
    pins = {
        "tool.mypy.python_version": _nested_str(_PYPROJECT, "tool", "mypy", "python_version"),
        "tool.pyright.pythonVersion": _nested_str(_PYPROJECT, "tool", "pyright", "pythonVersion"),
        "tool.pyrefly.python-version": _nested_str(_PYPROJECT, "tool", "pyrefly", "python-version"),
        "tool.ty.environment.python-version": _nested_str(_PYPROJECT, "tool", "ty", "environment", "python-version"),
    }
    drifted = [
        f"{name} is {value!r}, floor is {_PYTHON_FLOOR}" for name, value in pins.items() if value != _PYTHON_FLOOR
    ]
    ruff_target = _nested_str(_PYPROJECT, "tool", "ruff", "target-version")
    expected_ruff = f"py{_PYTHON_FLOOR.replace('.', '')}"
    if ruff_target != expected_ruff:
        drifted.append(f"tool.ruff.target-version is {ruff_target!r}, floor implies {expected_ruff!r}")
    assert not drifted, "\n".join(drifted)


def test_ci_matrix_and_os_claims_are_coherent() -> None:
    """The CI matrix tests exactly the classifier Pythons, and the OS prose matches the matrix legs."""
    ci = _ROOT / ".github" / "workflows" / "ci.yml"
    if not ci.exists():  # pragma: no cover -- absent only in the sdist, where .github/ is not shipped
        pytest.skip(".github/workflows/ci.yml is not shipped in the sdist")
    text = ci.read_text(encoding="utf-8")
    matrix = re.search(r"python-version: \[([^\]]+)\]", text)
    assert matrix, "ci.yml declares no python-version matrix"
    matrix_pythons = sorted(re.findall(r"\d+\.\d+", matrix.group(1)), key=lambda v: int(v.split(".")[1]))
    assert matrix_pythons == _PYTHON_VERSIONS, f"ci.yml matrix {matrix_pythons} != classifiers {_PYTHON_VERSIONS}"
    runners = set(re.findall(r"- (ubuntu|macos|windows)-latest", text))
    assert runners == {"ubuntu", "macos", "windows"}, f"ci.yml os matrix is {sorted(runners)}"
    os_names = ("Linux", "macOS", "Windows")
    readme = (_ROOT / "README.md").read_text(encoding="utf-8")
    installation = (_ROOT / "docs" / "installation.md").read_text(encoding="utf-8")
    missing = [
        f"{where} does not name {name}"
        for where, text_ in (("README.md badges", readme), ("docs/installation.md", installation))
        for name in os_names
        if name not in text_
    ]
    assert not missing, "\n".join(missing)
