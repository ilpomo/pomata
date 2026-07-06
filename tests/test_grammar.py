"""
The test-naming and presence grammar, enforced against the declared policies.

Two guarantees in one source-only walk, keyed on the public surface:

- **presence** -- every public function's test file carries at least one interior-``null`` test, one interior-``NaN``
  test, and one ``matches_reference`` test. A function shipped without one of these edge anchors is a red build, not the
  next audit's finding.
- **canonical names do not lie** -- a reserved ``test_null_*`` / ``test_nan_*`` name is used only by a function whose
  declared policy (:mod:`tests.support.policies`) it names, and a scale-family rung is spelled from the one scale
  vocabulary and sits in a ``Test*Properties`` class. Descriptive per-input names (``test_null_in_volume_propagates``)
  are left free; only the canonical ones are held.

Scale carries no presence mandate: a dimensionless ratio or a rolling metric legitimately has no scale test, so scale is
name-only. All detectors are ``test_``-scoped, so a helper like ``apply_domi``**``nan``**``t_cycle_period`` cannot be
mistaken for a NaN test.
"""

import ast
from pathlib import Path

import pytest
from tests.support.policies import POLICIES, NanPolicy, NullPolicy

from pomata import indicators, metrics, pnl

_TESTS_ROOT = Path(__file__).parent
_PACKAGES = {"indicators": indicators, "pnl": pnl, "metrics": metrics}

_RESERVED_NULL = {
    "test_null_skipped": NullPolicy.SKIPPED,
    "test_null_propagates": NullPolicy.PROPAGATES,
    "test_null_in_window_is_null": NullPolicy.IN_WINDOW_IS_NULL,
    "test_null_bridged": NullPolicy.BRIDGED,
    "test_null_latches": NullPolicy.LATCHES,
}
_RESERVED_NAN = {
    "test_nan_poisons": NanPolicy.POISONS,
    "test_nan_propagates": NanPolicy.PROPAGATES,
    "test_nan_latches": NanPolicy.LATCHES,
}
_SCALE_CANONICAL = frozenset(
    {
        "test_scale_homogeneity",
        "test_scale_invariance",
        "test_price_scale_homogeneity",
        "test_price_scale_invariance",
        "test_volume_scale_homogeneity",
        "test_volume_scale_invariance",
        "test_additive_shift_invariance",
    }
)
_SCALE_WORDS = ("homogen", "invari", "shift")


def _test_file(name: str) -> Path:
    for family, package in _PACKAGES.items():
        if name in package.__all__:
            return _TESTS_ROOT / family / f"test_{name}.py"
    raise KeyError(name)


def _methods(path: Path) -> list[tuple[str, str]]:
    """Every ``(class, method)`` defined inside a class in a test module."""
    module = ast.parse(path.read_text(encoding="utf-8"))
    return [
        (node.name, item.name)
        for node in module.body
        if isinstance(node, ast.ClassDef)
        for item in node.body
        if isinstance(item, ast.FunctionDef)
    ]


def _is_null(method: str) -> bool:
    return (
        method.startswith("test_")
        and "null" in method
        and not any(x in method for x in ("all_null", "warmup", "warm_up"))
    )


def _is_nan(method: str) -> bool:
    return method.startswith("test_") and "nan" in method


def _is_scale(method: str) -> bool:
    return method.startswith("test_") and (method.startswith("test_scale_") or any(w in method for w in _SCALE_WORDS))


def _is_scale_canonical(method: str) -> bool:
    return method in _SCALE_CANONICAL or method.startswith("test_scale_homogeneity_in_")


_METHODS: dict[str, list[tuple[str, str]]] = {name: _methods(_test_file(name)) for name in POLICIES}


def _reserved_usages() -> list[tuple[str, str, object, object]]:
    usages: list[tuple[str, str, object, object]] = []
    for name in sorted(POLICIES):
        null_policy, nan_policy = POLICIES[name]
        for _, method in _METHODS[name]:
            if method in _RESERVED_NULL:
                usages.append((name, method, _RESERVED_NULL[method], null_policy))
            elif method in _RESERVED_NAN:
                usages.append((name, method, _RESERVED_NAN[method], nan_policy))
    return usages


def _scale_usages() -> list[tuple[str, str, str]]:
    return [(name, cls, method) for name in sorted(POLICIES) for cls, method in _METHODS[name] if _is_scale(method)]


_RESERVED = _reserved_usages()
_SCALE = _scale_usages()


def test_the_scan_covered_every_function() -> None:
    """Guards the guard: every function's test file was parsed, and reserved / scale usages were actually found."""
    assert len(_METHODS) == len(POLICIES)
    assert _RESERVED
    assert _SCALE


@pytest.mark.parametrize("name", sorted(POLICIES))
def test_interior_null_test_present(name: str) -> None:
    """Verifies the function's test file carries at least one interior-``null`` test."""
    assert any(_is_null(method) for _, method in _METHODS[name]), f"{name}: no interior-null test"


@pytest.mark.parametrize("name", sorted(POLICIES))
def test_interior_nan_test_present(name: str) -> None:
    """Verifies the function's test file carries at least one interior-``NaN`` test."""
    assert any(_is_nan(method) for _, method in _METHODS[name]), f"{name}: no interior-nan test"


@pytest.mark.parametrize("name", sorted(POLICIES))
def test_matches_reference_present(name: str) -> None:
    """Verifies the function's test file carries at least one ``matches_reference`` test."""
    assert any(method.startswith("test_matches_reference") for _, method in _METHODS[name]), (
        f"{name}: no reference test"
    )


@pytest.mark.parametrize(
    ("name", "method", "reserved", "declared"), _RESERVED, ids=[f"{n}::{m}" for n, m, _, _ in _RESERVED]
)
def test_reserved_name_matches_policy(name: str, method: str, reserved: object, declared: object) -> None:
    """Verifies a canonical ``test_null_*`` / ``test_nan_*`` name is used only by a function declaring its policy."""
    assert reserved is declared, f"{name}: {method} is canonical for {reserved} but {name} declares {declared}"


@pytest.mark.parametrize(("name", "cls", "method"), _SCALE, ids=[f"{n}::{m}" for n, _, m in _SCALE])
def test_scale_name_is_canonical(name: str, cls: str, method: str) -> None:
    """Verifies every scale-family rung is drawn from the scale vocabulary and sits in a ``Test*Properties`` class."""
    assert _is_scale_canonical(method), f"{name}: {method} is not a canonical scale-rung name"
    assert cls.endswith("Properties"), f"{name}: {method} sits in {cls}, not a *Properties class"
