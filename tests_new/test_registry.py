"""
Collection-time surface guard: each registered family's declarations are in exact two-way correspondence with that
family's public ``__all__``. A stray declaration (a name the package does not export) fails as loudly as a gap (an
exported name with no declaration).
"""

import pytest

import pomata.indicators
import pomata.metrics
import pomata.pnl
import tests_new.all_declarations as _registered
from tests_new.support.registry import assert_bijection, registry_all, registry_for

# ``all_declarations`` is imported only to run its registration side effects, so the registry below is populated
# before it is read; nothing is referenced from it directly.
del _registered

_FAMILY_PACKAGES = {
    "indicators": pomata.indicators,
    "metrics": pomata.metrics,
    "pnl": pomata.pnl,
}

# Only families with at least one registered declaration are bijection-checked. While the suite is rebuilt family by
# family, a family whose declarations are not yet ported is legitimately absent; the cutover — when all three families
# are registered — checks all three. This is the single, clearly-scoped allowance, and it dissolves by construction the
# moment every family is registered.
_REGISTERED_FAMILIES = sorted({declaration.family for declaration in registry_all()})


@pytest.mark.parametrize("family", _REGISTERED_FAMILIES)
def test_registry_bijection(family: str) -> None:
    """Every public ``__all__`` name in the family has a declaration, and every declaration a public name."""
    assert_bijection(_FAMILY_PACKAGES[family], registry_for(family))
