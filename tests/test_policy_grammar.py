"""
The null / NaN policy test names, enforced against the registry.

A function's interior-``null`` / interior-``NaN`` value anchors stay in its own file: they pin an exact value the shared
flow guard (value-blind) and the oracle tier (relative) do not -- see :doc:`the policy <POLICY>`. What *can* drift, and
did, is their name: a windowed or recursive factory whose null anchor was named ``test_null_propagates`` claims a
pointwise policy it does not have. This meta-test makes that a red build.

The rule is **reserved names, not mandated presence**. The five canonical ``test_null_*`` names and three canonical
``test_nan_*`` names each belong to exactly one policy; a function may use a canonical name only if it declares that
policy. A factory is otherwise free to test its flow under a descriptive, function-specific name -- a multi-input
factory routinely must (``test_null_in_volume_propagates``, ``test_interior_null_in_high_is_absorbed``) -- and those are
left alone. The guard only forbids a canonical name from lying about the declared policy. It is the null/NaN analogue of
the scale grammar in :mod:`tests.test_scale_grammar`, keyed on the registry policy fields.
"""

import ast
from pathlib import Path

import pytest
from tests.support.registry import REGISTRY, NanPolicy, NullPolicy

_TESTS_ROOT = Path(__file__).parent

# Each canonical name belongs to one policy; using it under any other declared policy is a lie the guard forbids.
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


def _methods(path: Path) -> list[str]:
    """Every method name defined inside a class in a test module."""
    module = ast.parse(path.read_text(encoding="utf-8"))
    return [
        item.name
        for node in module.body
        if isinstance(node, ast.ClassDef)
        for item in node.body
        if isinstance(item, ast.FunctionDef)
    ]


def _reserved_usages() -> list[tuple[str, str, str, str]]:
    """Every reserved-name use in the suite, as ``(function, method, reserved_policy, declared_policy)``."""
    usages: list[tuple[str, str, str, str]] = []
    for name, profile in sorted(REGISTRY.items()):
        path = _TESTS_ROOT / profile.macro.value / f"test_{name}.py"
        for method in _methods(path):
            if method in _RESERVED_NULL:
                usages.append((name, method, _RESERVED_NULL[method].name, profile.null_policy.name))
            elif method in _RESERVED_NAN:
                usages.append((name, method, _RESERVED_NAN[method].name, profile.nan_policy.name))
    return usages


_USAGES = _reserved_usages()
_IDS = [f"{name}::{method}" for name, method, _, _ in _USAGES]


def test_the_scan_discovers_reserved_names() -> None:
    """Guards the guard: a discovery regression would leave the check below trivially green over an empty set."""
    assert len(_USAGES) > 100


@pytest.mark.parametrize(("name", "method", "reserved_policy", "declared_policy"), _USAGES, ids=_IDS)
def test_reserved_policy_name_matches_declared_policy(
    name: str,
    method: str,
    reserved_policy: str,
    declared_policy: str,
) -> None:
    """Verifies a canonical ``test_null_*`` / ``test_nan_*`` name is used only by a function declaring its policy."""
    assert reserved_policy == declared_policy, (
        f"{name}: {method} is the canonical name for {reserved_policy}, but {name} declares {declared_policy}"
    )
