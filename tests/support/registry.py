"""
Auto-registration and the surface bijection guard.

A per-family suite function registers each :class:`Declaration` as a side effect of importing its module, so no
explicit aggregator import is maintained by hand; the completeness guarantee stays fail-closed (a public function
without a declaration is red), but it is not the contributor's chore. The registry has one
dict per family; :func:`assert_bijection` proves a family's registered names are in exact two-way correspondence with
that family's public ``__all__``.
"""

from collections.abc import Mapping
from types import ModuleType

from tests.support.declaration import Declaration

registry_indicators: dict[str, Declaration] = {}
registry_metrics: dict[str, Declaration] = {}
registry_pnl: dict[str, Declaration] = {}

_REGISTRIES: dict[str, dict[str, Declaration]] = {
    "indicators": registry_indicators,
    "metrics": registry_metrics,
    "pnl": registry_pnl,
}


def register(declaration: Declaration) -> Declaration:
    """
    Add ``declaration`` to its family's registry and return it, so ``FOO = suite_pnl(...)`` both registers and binds.

    Args:
        declaration: The declaration to register; its ``family`` routes it to the right registry.

    Returns:
        The same declaration, unchanged.

    Raises:
        ValueError: If ``declaration.family`` is not a known family, or a declaration with the same name is already
            registered in that family (a duplicate would silently shadow the first).
    """
    if declaration.family not in _REGISTRIES:
        msg = f"{declaration.name}: unknown family {declaration.family!r} (expected one of {sorted(_REGISTRIES)})"
        raise ValueError(msg)
    registry = _REGISTRIES[declaration.family]
    if declaration.name in registry:
        msg = f"duplicate declaration in {declaration.family}: {declaration.name}"
        raise ValueError(msg)
    registry[declaration.name] = declaration
    return declaration


def registry_for(family: str) -> dict[str, Declaration]:
    """The registry dict for a family; raises ``KeyError`` for an unknown family."""
    return _REGISTRIES[family]


def registry_all() -> tuple[Declaration, ...]:
    """Every registered declaration across all families, in family then registration order."""
    return tuple(declaration for registry in _REGISTRIES.values() for declaration in registry.values())


def assert_bijection(package: ModuleType, registry: Mapping[str, Declaration]) -> None:
    """
    Assert a family's registered names are in exact two-way correspondence with its public ``__all__``.

    A stray declaration (a name the package does not export) fails as loudly as a gap (an exported name with no
    declaration), so the public surface itself stays the single source of truth.

    Args:
        package: The family package (e.g. ``pomata.pnl``), whose ``__all__`` is the public surface.
        registry: The family's registry dict, keyed by function name.

    Raises:
        ValueError: If the registered names and ``__all__`` disagree in either direction.
    """
    public = set(package.__all__)
    declared = set(registry)
    if declared != public:
        gaps = sorted(public - declared)
        strays = sorted(declared - public)
        msg = f"{package.__name__}: registry and __all__ disagree — missing declarations {gaps}, stray {strays}"
        raise ValueError(msg)
