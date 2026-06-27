"""
Consumer-side typing conformance for the public indicator factories.

Three complementary guards. ``test_every_public_factory_builds_an_expr`` is parametrised over
``pomata.indicators.__all__``: it calls every public factory with minimal valid arguments (synthesized from the
signature) and asserts the result is a ``pl.Expr`` at runtime. ``test_every_public_factory_is_annotated_as_expr`` is
parametrised the same way and asserts each factory's *declared* return type is exactly ``pl.Expr``, extending the
return-type contract to the whole public surface — so a factory whose annotation drifted off ``pl.Expr`` (a silent
failure the runtime sweep would miss) is caught. Both CANNOT go stale, since a newly added indicator is swept in
automatically. ``test_moving_average_factories_are_typed_as_expr`` then pins the *inferred* static contract with
``typing.assert_type`` (a compile-time assertion, runtime no-op) for the overload-bearing moving-average family, so the
type checkers (mypy ``--strict``, pyright, ty) confirm the optional-keyword overloads still resolve to ``pl.Expr``.
"""

from typing import assert_type, get_type_hints

import polars as pl
import pytest
from tests.support import synthesize_call

from pomata import indicators
from pomata.indicators import dema, ema, hma, rma, sma, t3, tema, vwma, wma


@pytest.mark.parametrize("name", indicators.__all__)
def test_every_public_factory_builds_an_expr(name: str) -> None:
    """
    Verifies that every public factory in ``__all__``, called with minimal valid arguments, builds a ``pl.Expr``.

    The call is synthesized from the signature (see :func:`tests.support.synthesize_call`), so parametrizing over
    ``__all__`` keeps coverage in lock-step with the public API: a newly added indicator is swept in automatically.
    """
    factory = getattr(indicators, name)
    positional, keywords = synthesize_call(factory)
    assert isinstance(factory(*positional, **keywords), pl.Expr)


@pytest.mark.parametrize("name", indicators.__all__)
def test_every_public_factory_is_annotated_as_expr(name: str) -> None:
    """
    Verifies every public factory declares ``pl.Expr`` as its return type, exhaustively across ``__all__``.

    Parametrizing over ``__all__`` keeps the return-type contract in lock-step with the public API: a newly added
    indicator is swept in automatically. A factory whose return annotation drifted off ``pl.Expr`` — the silent failure
    the runtime ``isinstance`` sweep would miss — fails here.
    """
    assert get_type_hints(getattr(indicators, name)).get("return") is pl.Expr


def test_moving_average_factories_are_typed_as_expr() -> None:
    """
    Verifies that every public moving-average factory is statically inferred as returning ``pl.Expr``.

    Each ``assert_type`` covers one factory and, where the factory exposes optional keywords (``adjust``,
    ``volume_factor``), the keyword overload as well, so the checkers confirm those paths preserve the ``pl.Expr``
    return type. The body is also exercised at runtime by pytest, which fails if any call no longer builds against the
    current Polars API.
    """
    price = pl.col("close")
    volume = pl.col("volume")

    assert_type(sma(price, 3), pl.Expr)
    assert_type(wma(price, 3), pl.Expr)
    assert_type(rma(price, 3), pl.Expr)
    assert_type(hma(price, 3), pl.Expr)
    assert_type(ema(price, 3), pl.Expr)
    assert_type(ema(price, 3, adjust=True), pl.Expr)
    assert_type(dema(price, 3), pl.Expr)
    assert_type(dema(price, 3, adjust=True), pl.Expr)
    assert_type(tema(price, 3), pl.Expr)
    assert_type(tema(price, 3, adjust=True), pl.Expr)
    assert_type(t3(price, 3), pl.Expr)
    assert_type(t3(price, 3, volume_factor=0.5, adjust=True), pl.Expr)
    assert_type(vwma(price, volume, 3), pl.Expr)
