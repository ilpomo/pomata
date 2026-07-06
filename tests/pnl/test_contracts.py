"""
Universal structural contract for the public pnl factories -- the rungs identical for every function, from one place.

Every pnl factory owes the same five structural guarantees; rather than copy them into all eighteen test modules, this
contract parametrizes over ``pnl.__all__`` and applies the shared,
shape-aware assertions in :mod:`tests.support.contracts`. It is shared *within* pnl, never *across* families: the module
tests each function once; a newly added pnl function is swept in
automatically.

The function-specific rungs stay in each function's own file: ``.over`` semantics (identity vs per-group), warm-up, the
``null`` / ``NaN`` policy, the validation raises, the golden masters, and the property tiers.
"""

import pytest
from tests.support import contracts

from pomata import pnl

_PNL = sorted(pnl.__all__)


@pytest.mark.parametrize("name", _PNL)
def test_returns_expr(name: str) -> None:
    """Verifies the factory returns a ``pl.Expr`` without touching a frame."""
    contracts.assert_returns_expr(getattr(pnl, name))


@pytest.mark.parametrize("name", _PNL)
def test_shape(name: str) -> None:
    """Verifies each factory returns one ``Float64`` value per input row (all pnl factories are elementwise)."""
    contracts.assert_shape(getattr(pnl, name))


@pytest.mark.parametrize("name", _PNL)
def test_lazy_eager_parity(name: str) -> None:
    """Verifies eager and lazy application produce identical materialized output."""
    contracts.assert_lazy_eager_parity(getattr(pnl, name))


@pytest.mark.parametrize("name", _PNL)
def test_empty(name: str) -> None:
    """Verifies an empty series yields an empty result."""
    contracts.assert_empty(getattr(pnl, name))


@pytest.mark.parametrize("name", _PNL)
def test_all_null(name: str) -> None:
    """Verifies an all-null series stays all-null."""
    contracts.assert_all_null(getattr(pnl, name))
