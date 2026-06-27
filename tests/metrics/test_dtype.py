"""
Dtype-uniformity contract for the public metrics factories.

Every factory routes each input through ``float64_expr`` (a cast to ``Float64``), so the package has one predictable
output dtype regardless of the input's numeric dtype, and rejects a bare string column name with a clear ``TypeError``.
``test_output_is_float64`` parametrizes over ``pomata.metrics.__all__`` x {Float32, Int64}, so the guarantee covers the
whole public surface and a newly added function is swept in automatically; ``test_string_input_raises_type_error``
pins the misuse path (a column name passed where a ``pl.Expr`` is required).
"""

import polars as pl
import pytest
from tests.support import COLUMN_X, assert_all_float64, synthesize_call

from pomata import metrics


@pytest.mark.parametrize("input_dtype", [pl.Float32, pl.Int64])
@pytest.mark.parametrize("name", metrics.__all__)
def test_output_is_float64(name: str, input_dtype: pl.DataType) -> None:
    """
    Verifies every public factory returns ``Float64`` (or an all-``Float64`` struct) from a Float32 or Int64 input.
    """
    factory = getattr(metrics, name)
    positional, keywords = synthesize_call(factory)
    frame = pl.DataFrame({COLUMN_X: pl.Series(range(1, 21), dtype=input_dtype)})
    result = frame.select(factory(*positional, **keywords))
    assert_all_float64(result.dtypes[0], name)


@pytest.mark.parametrize("name", metrics.__all__)
def test_string_input_raises_type_error(name: str) -> None:
    """
    Verifies every public factory rejects a bare string column name (instead of a ``pl.Expr``) with a ``TypeError``.
    """
    factory = getattr(metrics, name)
    positional, keywords = synthesize_call(factory)
    expr_index = next(index for index, argument in enumerate(positional) if isinstance(argument, pl.Expr))
    arguments = [*positional]
    arguments[expr_index] = COLUMN_X
    with pytest.raises(TypeError, match="Polars expression"):
        factory(*arguments, **keywords)
