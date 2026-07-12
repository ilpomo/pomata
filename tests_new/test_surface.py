"""
The surface guard: the contract registry corresponds exactly to the migrated surface, both directions.

During the migration window ``MIGRATED`` lists, per family, the functions whose contract has landed in
``tests_new``; every category PR extends it, and the guard fails on a stray contract (a class for a function that
is not listed) as much as on a missing one (a listed function without a class) — no skips, no thresholds. At
cutover the lists are replaced by the public ``__all__`` tuples themselves, and this file becomes the bijection
guard of the whole suite.
"""

import pytest
from tests_new.support import REGISTRY

import pomata.indicators
import pomata.metrics
import pomata.pnl

MIGRATED: dict[str, frozenset[str]] = {
    "indicators": frozenset({"ichimoku", "mama"}),
    "metrics": frozenset({"sharpe_ratio"}),
    "pnl": frozenset({"equity_curve"}),
}

_FAMILIES = {"indicators": pomata.indicators, "metrics": pomata.metrics, "pnl": pomata.pnl}


@pytest.mark.parametrize("family", sorted(_FAMILIES))
def test_migrated_names_are_public(family: str) -> None:
    """Verifies every migrated name is (still) part of its family's public ``__all__``."""
    stray = MIGRATED[family] - set(_FAMILIES[family].__all__)
    assert not stray, f"{family}: migrated names not in __all__: {sorted(stray)}"


@pytest.mark.parametrize("family", sorted(_FAMILIES))
def test_registry_matches_migrated_surface(family: str) -> None:
    """Verifies the registry holds exactly one contract per migrated function — no missing, no strays."""
    registered = {name for name, contract in REGISTRY.items() if contract.family == family}
    missing = MIGRATED[family] - registered
    stray = registered - MIGRATED[family]
    assert not missing, f"{family}: migrated without a contract: {sorted(missing)}"
    assert not stray, f"{family}: contracts for unmigrated names: {sorted(stray)}"
