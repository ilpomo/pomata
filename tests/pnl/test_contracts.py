"""
Universal structural contract for the public pnl factories -- the rungs identical for every function, from one place.

Every pnl factory is an ``ELEMENTWISE`` ``pl.Expr``, so five structural guarantees hold for all of them regardless of
what the number means: it is an expression, it preserves length at ``Float64``, it is lazy/eager stable, an empty input
yields an empty output, and an all-``null`` input stays all-``null``. Rather than copy those into all eighteen pnl test
modules, this contract parametrizes over the pnl subset of the registry (:mod:`tests.support.registry`) -- so it reads
the one central map, tests each function by its own row, and sweeps in a newly added pnl function automatically.

The function-specific rungs stay in each function's own file: ``.over`` semantics (identity vs per-group), warm-up, the
``null`` / ``NaN`` policy, the validation raises, correctness (golden masters), and the property tiers.
"""

import polars as pl
import pytest
from polars.testing import assert_frame_equal
from tests.support import COLUMN_X, synthesize_call
from tests.support.registry import REGISTRY, Macro, Shape

from pomata import pnl

_PNL = sorted(name for name, profile in REGISTRY.items() if profile.macro is Macro.PNL)
_SERIES: list[float | None] = [100.0, 105.0, 102.0, 108.0, 110.0]


def _materialize(name: str, series: list[float | None]) -> pl.DataFrame:
    factory = getattr(pnl, name)
    positional, keywords = synthesize_call(factory)
    frame = pl.DataFrame({COLUMN_X: pl.Series(COLUMN_X, series, dtype=pl.Float64)})
    return frame.select(factory(*positional, **keywords).alias("y"))


@pytest.mark.parametrize("name", _PNL)
def test_returns_expr(name: str) -> None:
    """
    Verifies that the factory returns a ``pl.Expr`` without touching a frame.
    """
    factory = getattr(pnl, name)
    positional, keywords = synthesize_call(factory)
    assert isinstance(factory(*positional, **keywords), pl.Expr)


@pytest.mark.parametrize("name", _PNL)
def test_preserves_length(name: str) -> None:
    """
    Verifies that each (elementwise) factory returns one ``Float64`` value per input row.
    """
    assert REGISTRY[name].shape is Shape.ELEMENTWISE
    result = _materialize(name, _SERIES)
    assert result.height == len(_SERIES)
    assert result.schema["y"] == pl.Float64


@pytest.mark.parametrize("name", _PNL)
def test_lazy_eager_parity(name: str) -> None:
    """
    Verifies that eager and lazy application produce identical materialized output.
    """
    factory = getattr(pnl, name)
    positional, keywords = synthesize_call(factory)
    expr = factory(*positional, **keywords).alias("y")
    frame = pl.DataFrame({COLUMN_X: pl.Series(COLUMN_X, _SERIES, dtype=pl.Float64)})
    assert_frame_equal(frame.select(expr), frame.lazy().select(expr).collect())


@pytest.mark.parametrize("name", _PNL)
def test_empty(name: str) -> None:
    """
    Verifies that an empty series yields an empty result.
    """
    assert _materialize(name, []).height == 0


@pytest.mark.parametrize("name", _PNL)
def test_all_null(name: str) -> None:
    """
    Verifies that an all-null series stays all-null (no defined value can be produced).
    """
    assert _materialize(name, [None, None, None])["y"].to_list() == [None, None, None]
