"""
Collection-time surface guard: each registered family's declarations are in exact two-way correspondence with that
family's public ``__all__``. A stray declaration (a name the package does not export) fails as loudly as a gap (an
exported name with no declaration).
"""

import pytest

import pomata.indicators
import pomata.metrics
import pomata.pnl
import tests.all_declarations as _registered
from tests.support.registry import assert_bijection, registry_all, registry_for

# ``all_declarations`` is imported only to run its registration side effects, so the registry below is populated
# before it is read; nothing is referenced from it directly.
del _registered

_FAMILY_PACKAGES = {
    "indicators": pomata.indicators,
    "metrics": pomata.metrics,
    "pnl": pomata.pnl,
}

# Each registered family is bijection-checked against its public ``__all__``. Every family registers its declarations
# at import, so all three are covered; deriving the set from the registry keeps the check scoped to what is registered
# rather than to a hard-coded family list.
_REGISTERED_FAMILIES = sorted({declaration.family for declaration in registry_all()})


@pytest.mark.parametrize("family", _REGISTERED_FAMILIES)
def test_registry_bijection(family: str) -> None:
    """Every public ``__all__`` name in the family has a declaration, and every declaration a public name."""
    assert_bijection(_FAMILY_PACKAGES[family], registry_for(family))
