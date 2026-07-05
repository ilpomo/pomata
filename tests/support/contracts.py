"""
Shape-aware universal-contract assertions, shared by the per-family ``test_contracts.py`` modules.

These are the structural guarantees every public factory owes regardless of what its number means -- it is a
``pl.Expr``, it has the output shape its registry row declares, it is lazy/eager stable, an empty input yields an empty
output, and an all-``null`` input produces all-``null`` (a null scalar for a reducing factory, a null struct for a
struct one). The logic lives here once; each family's ``tests/<family>/test_contracts.py`` wires it to the registry
subset of that family, so the contract is shared *within* a family, never *across* families, and a genuine per-family
difference (only metrics partition uniformly under ``.over``) is expressed by which rungs that family's module runs.
"""

from collections.abc import Callable

import polars as pl
from polars.testing import assert_frame_equal
from tests.support.asserts import assert_matches
from tests.support.columns import COLUMN_X, GROUP_KEY
from tests.support.registry import FunctionProfile, Shape
from tests.support.synthesis import synthesize_call

_SERIES: list[float | None] = [100.0, 105.0, 102.0, 108.0, 110.0]
_GROUP_A: list[float | None] = [100.0, 105.0, 102.0, 108.0, 110.0, 103.0]
_GROUP_B: list[float | None] = [50.0, 52.0, 51.0, 55.0, 53.0]


def _materialize(factory: Callable[..., pl.Expr], series: list[float | None]) -> pl.DataFrame:
    positional, keywords = synthesize_call(factory)
    frame = pl.DataFrame({COLUMN_X: pl.Series(COLUMN_X, series, dtype=pl.Float64)})
    return frame.select(factory(*positional, **keywords).alias("y"))


def _values(result: pl.DataFrame) -> list[float | None]:
    """The output column as a flat list, unnesting a struct's first field so struct and series read alike."""
    column = result.unnest("y").to_series(0) if isinstance(result.schema["y"], pl.Struct) else result["y"]
    return column.to_list()


def assert_returns_expr(factory: Callable[..., pl.Expr]) -> None:
    """The factory returns a ``pl.Expr`` without touching a frame."""
    positional, keywords = synthesize_call(factory)
    assert isinstance(factory(*positional, **keywords), pl.Expr)


def assert_shape(factory: Callable[..., pl.Expr], profile: FunctionProfile) -> None:
    """The output has the shape the row declares: one ``Float64`` scalar, a same-length ``Float64`` series, or a
    same-length ``Float64`` struct.
    """
    result = _materialize(factory, _SERIES)
    dtype = result.schema["y"]
    if profile.shape is Shape.REDUCING:
        assert result.height == 1
        assert dtype == pl.Float64
    elif profile.shape is Shape.STRUCT:
        assert isinstance(dtype, pl.Struct)
        assert result.height == len(_SERIES)
        assert all(field.dtype == pl.Float64 for field in dtype.fields)
    else:
        assert result.height == len(_SERIES)
        assert dtype == pl.Float64


def assert_lazy_eager_parity(factory: Callable[..., pl.Expr]) -> None:
    """Eager and lazy application produce identical materialized output."""
    positional, keywords = synthesize_call(factory)
    expr = factory(*positional, **keywords).alias("y")
    frame = pl.DataFrame({COLUMN_X: pl.Series(COLUMN_X, _SERIES, dtype=pl.Float64)})
    assert_frame_equal(frame.select(expr), frame.lazy().select(expr).collect())


def assert_empty(factory: Callable[..., pl.Expr], profile: FunctionProfile) -> None:
    """An empty input yields an empty result -- or one null scalar for a reducing factory (nothing reduced)."""
    result = _materialize(factory, [])
    if profile.shape is Shape.REDUCING:
        assert result.height == 1
        assert result["y"].to_list() == [None]
    else:
        assert result.height == 0


def assert_over_partitions(factory: Callable[..., pl.Expr], profile: FunctionProfile) -> None:
    """Under ``.over`` the factory is computed per group and never spans a boundary: the two groups' outputs on one
    frame match what each group produces on its own (a reducing factory broadcasts its scalar across the group).
    """
    positional, keywords = synthesize_call(factory)
    frame = pl.DataFrame(
        {
            GROUP_KEY: ["a"] * len(_GROUP_A) + ["b"] * len(_GROUP_B),
            COLUMN_X: pl.Series(COLUMN_X, _GROUP_A + _GROUP_B, dtype=pl.Float64),
        }
    )
    grouped = _values(frame.select(factory(*positional, **keywords).over(GROUP_KEY).alias("y")))
    alone_a, alone_b = _values(_materialize(factory, _GROUP_A)), _values(_materialize(factory, _GROUP_B))
    if profile.shape is Shape.REDUCING:
        expected = alone_a * len(_GROUP_A) + alone_b * len(_GROUP_B)
    else:
        expected = alone_a + alone_b
    assert_matches(grouped, expected)


def assert_all_null(factory: Callable[..., pl.Expr], profile: FunctionProfile) -> None:
    """An all-null input stays all-null: a null scalar (reducing), an all-null series, or an all-null struct."""
    result = _materialize(factory, [None, None, None])
    if profile.shape is Shape.REDUCING:
        assert result.height == 1
        assert result["y"].to_list() == [None]
    elif profile.shape is Shape.STRUCT:
        fields = result.unnest("y")
        assert all(fields[column].to_list() == [None, None, None] for column in fields.columns)
    else:
        assert result["y"].to_list() == [None, None, None]
