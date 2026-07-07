"""
Consumer-side typing conformance for the public PnL factories.

Three complementary guards. ``test_every_public_factory_builds_an_expr`` is parametrised over ``pomata.pnl.__all__``: it
calls every public factory with minimal valid arguments (synthesized from the signature) and asserts the result is a
``pl.Expr`` at runtime. ``test_every_public_factory_is_annotated_as_expr`` is parametrised the same way and asserts each
factory's *declared* return type is exactly ``pl.Expr``, extending the return-type contract to the whole public surface
— so a function whose annotation drifted off ``pl.Expr`` (a silent failure the runtime sweep would miss) is caught. Both
CANNOT go stale, since a newly added function is swept in automatically. ``test_factories_are_typed_as_expr`` then pins
the *inferred* static contract with ``typing.assert_type`` (a compile-time assertion, runtime no-op) for every function,
so the type checkers (mypy ``--strict``, pyright, ty) confirm the inferred return type is ``pl.Expr``.
"""

from typing import assert_type, get_type_hints

import polars as pl
import pytest
from tests.support import synthesize_call

from pomata import pnl
from pomata.pnl import (
    cost_borrow,
    cost_fixed,
    cost_funding,
    cost_notional,
    cost_per_share,
    cost_proportional,
    cost_slippage,
    cumulative_pnl,
    dividend,
    equity_curve,
    pnl_gross,
    pnl_gross_inverse,
    pnl_net,
    returns_gross,
    returns_log,
    returns_net,
    returns_simple,
    turnover,
)


@pytest.mark.parametrize("name", pnl.__all__)
def test_every_public_factory_builds_an_expr(name: str) -> None:
    """
    Verifies that every public factory in ``__all__``, called with minimal valid arguments, builds a ``pl.Expr``.

    The call is synthesized from the signature (see :func:`tests.support.synthesize_call`), so parametrizing over
    ``__all__`` keeps coverage in lock-step with the public API: a newly added function is swept in automatically.
    """
    factory = getattr(pnl, name)
    positional, keywords = synthesize_call(factory)
    assert isinstance(factory(*positional, **keywords), pl.Expr)


@pytest.mark.parametrize("name", pnl.__all__)
def test_every_public_factory_is_annotated_as_expr(name: str) -> None:
    """
    Verifies every public factory declares ``pl.Expr`` as its return type, exhaustively across ``__all__``.

    Parametrizing over ``__all__`` keeps the return-type contract in lock-step with the public API: a newly added
    function is swept in automatically. A factory whose return annotation drifted off ``pl.Expr`` — the silent failure
    the runtime ``isinstance`` sweep would miss — fails here.
    """
    assert get_type_hints(getattr(pnl, name)).get("return") is pl.Expr


def test_factories_are_typed_as_expr() -> None:
    """
    Verifies that every public PnL factory is statically inferred as returning ``pl.Expr``.

    Each ``assert_type`` covers one factory across the two flows — the cash / position flow (``quantity`` and ``price``)
    and the return flow (``weight`` and ``returns``) — so the checkers confirm the return type across the whole family.
    The body is also exercised at runtime by pytest, which fails if any call no longer builds against the current Polars
    API.
    """
    quantity = pl.col("quantity")
    price = pl.col("price")
    weight = pl.col("weight")
    returns = pl.col("returns")
    series = pl.col("series")

    assert_type(cost_borrow(quantity, price, rate=0.01), pl.Expr)
    assert_type(cost_fixed(quantity, fee=1.0), pl.Expr)
    assert_type(cost_funding(quantity, price, series), pl.Expr)
    assert_type(cost_notional(quantity, price, rate=0.01), pl.Expr)
    assert_type(cost_per_share(quantity, fee=1.0), pl.Expr)
    assert_type(cost_proportional(weight, rate=0.001), pl.Expr)
    assert_type(cost_slippage(weight, half_spread=0.0005), pl.Expr)
    assert_type(cumulative_pnl(returns), pl.Expr)
    assert_type(dividend(quantity, series), pl.Expr)
    assert_type(equity_curve(returns), pl.Expr)
    assert_type(pnl_gross(quantity, price), pl.Expr)
    assert_type(pnl_gross_inverse(quantity, price), pl.Expr)
    assert_type(pnl_net(series, series), pl.Expr)
    assert_type(returns_gross(weight, returns), pl.Expr)
    assert_type(returns_log(price), pl.Expr)
    assert_type(returns_net(series, series), pl.Expr)
    assert_type(returns_simple(price), pl.Expr)
    assert_type(turnover(weight), pl.Expr)
