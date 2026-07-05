"""
Universal structural contract for the public indicators factories -- the rungs identical for every function.

Every indicator owes the same four structural guarantees; rather than copy them into all 75 test modules, this contract
parametrizes over ``indicators.__all__`` and applies the shared,
shape-aware assertions in :mod:`tests.support.contracts`. A newly added indicator is swept in automatically.

``all_null`` is **not** shared for indicators: it is not universal here -- ``dm_plus`` / ``dm_minus`` absorb a null
directional movement to ``0`` rather than staying null, a genuine per-function variant -- so it stays in each file. The
``.over`` rung also stays per-file: indicators partition, anchor per session, or reduce to identity by class. Warm-up,
the ``null`` / ``NaN`` policy, struct field names, correctness, and the property tiers likewise stay per-file.
"""

import pytest
from tests.support import contracts

from pomata import indicators

_INDICATORS = sorted(indicators.__all__)


@pytest.mark.parametrize("name", _INDICATORS)
def test_returns_expr(name: str) -> None:
    """Verifies the factory returns a ``pl.Expr`` without touching a frame."""
    contracts.assert_returns_expr(getattr(indicators, name))


@pytest.mark.parametrize("name", _INDICATORS)
def test_shape(name: str) -> None:
    """Verifies the output has a coherent shape (series / struct), at ``Float64``."""
    contracts.assert_shape(getattr(indicators, name))


@pytest.mark.parametrize("name", _INDICATORS)
def test_lazy_eager_parity(name: str) -> None:
    """Verifies eager and lazy application produce identical materialized output."""
    contracts.assert_lazy_eager_parity(getattr(indicators, name))


@pytest.mark.parametrize("name", _INDICATORS)
def test_empty(name: str) -> None:
    """Verifies an empty series yields an empty result."""
    contracts.assert_empty(getattr(indicators, name))
