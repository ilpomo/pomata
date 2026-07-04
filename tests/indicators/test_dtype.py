"""
Dtype-uniformity contract for the public indicator factories.

Every factory routes each input through ``float64_expr`` (a cast to ``Float64``), so the package has one predictable
output dtype regardless of the input's numeric dtype, and rejects a bare string column name with a clear ``TypeError``.
``test_output_is_float64`` parametrizes over ``pomata.indicators.__all__`` x {Float32, Int64}, so the guarantee covers
the whole public surface and a newly added indicator is swept in automatically; ``test_string_input_raises_type_error``
pins the misuse path (a column name passed where a ``pl.Expr`` is required).
"""

import polars as pl
import pytest
from tests.support import COLUMN_X, assert_all_float64, synthesize_call

from pomata import indicators


@pytest.mark.parametrize("input_dtype", [pl.Float32, pl.Int64])
@pytest.mark.parametrize("name", indicators.__all__)
def test_output_is_float64(name: str, input_dtype: pl.DataType) -> None:
    """
    Verifies every public factory returns ``Float64`` (or an all-``Float64`` struct) from a Float32 or Int64 input.
    """
    factory = getattr(indicators, name)
    positional, keywords = synthesize_call(factory)
    frame = pl.DataFrame({COLUMN_X: pl.Series(range(1, 21), dtype=input_dtype)})
    result = frame.select(factory(*positional, **keywords))
    assert_all_float64(result.dtypes[0], name)


@pytest.mark.parametrize("name", indicators.__all__)
def test_string_input_raises_type_error(name: str) -> None:
    """
    Verifies every public factory rejects a bare string column name (instead of a ``pl.Expr``) with a ``TypeError``.
    """
    factory = getattr(indicators, name)
    positional, keywords = synthesize_call(factory)
    expr_index = next(index for index, argument in enumerate(positional) if isinstance(argument, pl.Expr))
    arguments = [*positional]
    arguments[expr_index] = COLUMN_X
    with pytest.raises(TypeError, match="Polars expression"):
        factory(*arguments, **keywords)


_MOVING_AVERAGES = ("sma", "ema", "wma", "rma", "dema", "tema", "hma", "kama", "t3", "trima", "vwma")


@pytest.mark.parametrize("name", _MOVING_AVERAGES)
def test_moving_average_preserves_input_name(name: str) -> None:
    """
    Verifies every single-output moving average keeps the input column's name, never renaming it to ``literal``.
    """
    factory = getattr(indicators, name)
    positional, keywords = synthesize_call(factory)
    frame = pl.DataFrame({COLUMN_X: pl.Series(range(1, 21), dtype=pl.Float64)})
    assert frame.select(factory(*positional, **keywords)).columns[0] == COLUMN_X
