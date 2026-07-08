"""
Shape-aware universal-contract assertions, shared by the per-family ``test_contracts.py`` modules.

These are the structural guarantees every public factory owes regardless of what its number means -- it is a
``pl.Expr``, it has a coherent output shape (one ``Float64`` scalar, a same-length ``Float64`` series, or a same-length
``Float64`` struct), it is lazy/eager stable, an empty input yields an empty output, and an all-``null`` input produces
all-``null``. The shape is **observed** from a probe, not declared: a reducing factory returns one row, a struct factory
a ``Struct`` column, everything else a same-length series. The logic lives here once; each family's
``tests/<family>/test_contracts.py`` iterates that family's public ``__all__`` and applies the rungs that family owns.
"""

from collections.abc import Callable

import polars as pl
from polars.testing import assert_frame_equal
from tests.support.asserts import assert_matches
from tests.support.columns import COLUMN_X, GROUP_KEY
from tests.support.synthesis import synthesize_call

_SERIES: list[float | None] = [100.0, 105.0, 102.0, 108.0, 110.0]
_GROUP_A: list[float | None] = [100.0, 105.0, 102.0, 108.0, 110.0, 103.0]
_GROUP_B: list[float | None] = [50.0, 52.0, 51.0, 55.0, 53.0]


def _materialize(factory: Callable[..., pl.Expr], series: list[float | None]) -> pl.DataFrame:
    positional, keywords = synthesize_call(factory)
    frame = pl.DataFrame({COLUMN_X: pl.Series(COLUMN_X, series, dtype=pl.Float64)})
    return frame.select(factory(*positional, **keywords).alias("y"))


def _field_lists(result: pl.DataFrame) -> list[list[float | None]]:
    """Every output column as its own list — all of a struct's fields, in field order — so no field escapes a
    per-field contract (a partition leak in a non-first struct field must fail exactly like one in the first).
    """
    if isinstance(result.schema["y"], pl.Struct):
        fields = result.unnest("y")
        return [fields[column].to_list() for column in fields.columns]
    return [result["y"].to_list()]


def _observe(factory: Callable[..., pl.Expr]) -> str:
    """The factory's output shape, read from a clean probe: ``reducing`` / ``struct`` / ``elementwise``."""
    result = _materialize(factory, _SERIES)
    if result.height == 1:
        return "reducing"
    if isinstance(result.schema["y"], pl.Struct):
        return "struct"
    return "elementwise"


def assert_returns_expr(factory: Callable[..., pl.Expr]) -> None:
    """The factory returns a ``pl.Expr`` without touching a frame."""
    positional, keywords = synthesize_call(factory)
    assert isinstance(factory(*positional, **keywords), pl.Expr)


def assert_shape(factory: Callable[..., pl.Expr]) -> None:
    """The output is coherent: one ``Float64`` scalar, a same-length ``Float64`` series, or a same-length ``Float64``
    struct.
    """
    result = _materialize(factory, _SERIES)
    dtype = result.schema["y"]
    if result.height == 1:
        assert dtype == pl.Float64
    elif isinstance(dtype, pl.Struct):
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


def assert_empty(factory: Callable[..., pl.Expr]) -> None:
    """An empty input yields an empty result -- or one null scalar for a reducing factory (nothing reduced)."""
    result = _materialize(factory, [])
    if _observe(factory) == "reducing":
        assert result.height == 1
        assert result["y"].to_list() == [None]
    else:
        assert result.height == 0


def assert_over_partitions(factory: Callable[..., pl.Expr]) -> None:
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
    grouped = _field_lists(frame.select(factory(*positional, **keywords).over(GROUP_KEY).alias("y")))
    alone_a = _field_lists(_materialize(factory, _GROUP_A))
    alone_b = _field_lists(_materialize(factory, _GROUP_B))
    reducing = _observe(factory) == "reducing"
    for grouped_field, alone_field_a, alone_field_b in zip(grouped, alone_a, alone_b, strict=True):
        if reducing:
            expected = alone_field_a * len(_GROUP_A) + alone_field_b * len(_GROUP_B)
        else:
            expected = alone_field_a + alone_field_b
        assert_matches(grouped_field, expected)


def assert_all_null(factory: Callable[..., pl.Expr]) -> None:
    """An all-null input stays all-null: a null scalar (reducing), an all-null series, or an all-null struct."""
    shape = _observe(factory)
    result = _materialize(factory, [None, None, None])
    if shape == "reducing":
        assert result.height == 1
        assert result["y"].to_list() == [None]
    elif shape == "struct":
        fields = result.unnest("y")
        assert all(fields[column].to_list() == [None, None, None] for column in fields.columns)
    else:
        assert result["y"].to_list() == [None, None, None]
