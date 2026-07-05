"""
The scale-property naming grammar, enforced across the whole suite.

The scale axis is the one edge-case family that does *not* live in a shared, registry-driven contract: a function's
homogeneity degree is per-input and family-specific (a variance is degree-2, a VWAP is degree-1 in price and degree-0
in volume, a borrow cost is degree-1 in quantity), so the scale tests stay in each function's own file -- see
:doc:`the policy <POLICY>` (`tests/POLICY.md`), §2. What *can* be held uniform is the **name**: the same two-axis
price/volume property is otherwise easy to spell four different ways. This meta-test turns that into a red build. It
reads every test module's source with :mod:`ast` and asserts that every scale-family rung it finds is drawn from the
one canonical vocabulary and lives in a ``Test*Properties`` class -- so a future ``test_scale_behavior`` or
``test_price_homogeneity`` fails the build.

It is the scale analogue of the registry bijection in :mod:`tests.test_registry`: structural, source-only, and exact.
"""

import ast
from pathlib import Path

import pytest

_TESTS_ROOT = Path(__file__).parent
_FAMILIES = ("indicators", "pnl", "metrics")

# A rung is "scale-family" if its name speaks of homogeneity / invariance / shift, or is a ``test_scale_*`` variant.
# The ``test_scale_`` prefix is what catches a vague non-conforming name like ``test_scale_behavior`` (which names no
# property at all) -- the pattern words alone would miss it, and missing it is exactly the failure this guard prevents.
_SCALE_FAMILY_WORDS = ("homogen", "invari", "shift")
_SCALE_PREFIX = "test_scale_"

# The one canonical vocabulary. A per-input pnl rung is ``test_scale_homogeneity_in_<role>`` (in_quantity / in_weight /
# in_each_input); the prefix admits a future role without editing this set.
_CANONICAL_NAMES = frozenset(
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
_PER_INPUT_PREFIX = "test_scale_homogeneity_in_"


def _is_scale_family(name: str) -> bool:
    """Whether a test-method name belongs to the scale property family this grammar governs."""
    if not name.startswith("test_"):
        return False
    return name.startswith(_SCALE_PREFIX) or any(word in name for word in _SCALE_FAMILY_WORDS)


def _is_canonical(name: str) -> bool:
    """Whether a scale-family name is one the grammar allows."""
    return name in _CANONICAL_NAMES or name.startswith(_PER_INPUT_PREFIX)


def _scale_methods() -> list[tuple[str, str, str]]:
    """Every scale-family test in the suite, as ``(module path relative to tests/, enclosing class, method name)``.

    A method defined at module level (not inside a class) is reported with an empty class, so the placement check can
    flag it: a scale rung must live in a ``Test*Properties`` class.
    """
    found: list[tuple[str, str, str]] = []
    for family in _FAMILIES:
        for path in sorted((_TESTS_ROOT / family).glob("test_*.py")):
            relative = str(path.relative_to(_TESTS_ROOT))
            module = ast.parse(path.read_text(encoding="utf-8"))
            for node in module.body:
                if isinstance(node, ast.ClassDef):
                    found.extend(
                        (relative, node.name, item.name)
                        for item in node.body
                        if isinstance(item, ast.FunctionDef) and _is_scale_family(item.name)
                    )
                elif isinstance(node, ast.FunctionDef) and _is_scale_family(node.name):
                    found.append((relative, "", node.name))
    return found


_SCALE_METHODS = _scale_methods()
_IDS = [f"{path}::{cls}::{method}" for path, cls, method in _SCALE_METHODS]


def test_the_scan_discovers_the_scale_suite() -> None:
    """Guards the guard: a discovery regression would leave the two checks below trivially green over an empty set."""
    assert len(_SCALE_METHODS) > 100


@pytest.mark.parametrize(("path", "cls", "method"), _SCALE_METHODS, ids=_IDS)
def test_scale_rung_name_is_canonical(path: str, cls: str, method: str) -> None:
    """Verifies every scale-family rung is drawn from the one canonical vocabulary (§4 of the policy)."""
    assert _is_canonical(method), f"{path}::{cls}::{method}: not a canonical scale-rung name"


@pytest.mark.parametrize(("path", "cls", "method"), _SCALE_METHODS, ids=_IDS)
def test_scale_rung_lives_in_a_properties_class(path: str, cls: str, method: str) -> None:
    """Verifies every scale-family rung sits in the ``Test*Properties`` tier, never loose or in another class."""
    assert cls.endswith("Properties"), f"{path}::{cls or '<module>'}::{method}: scale rung outside a *Properties class"
