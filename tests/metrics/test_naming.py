"""
Column-name-preservation contract for the public metrics factories.

Every factory keeps the input column's name on its output. A guarded metric ends its expression in a ``pl.when(...)``
whose literal branch (``pl.lit(...)`` / ``None``) would otherwise surface as a ``'literal'`` column -- which both reads
oddly on a bare ``select`` and collides (``DuplicateError``) when two such metrics are selected together.
``test_output_preserves_name`` parametrizes over ``pomata.metrics.__all__``, so the guarantee covers the whole public
surface and a newly added function is swept in automatically.
"""

import polars as pl
import pytest
from tests.support import COLUMN_X, synthesize_call

from pomata import metrics


@pytest.mark.parametrize("name", metrics.__all__)
def test_output_preserves_name(name: str) -> None:
    """
    Verifies every public factory names its output column after the input (``COLUMN_X``), never ``'literal'``.
    """
    factory = getattr(metrics, name)
    positional, keywords = synthesize_call(factory)
    frame = pl.DataFrame({COLUMN_X: pl.Series(range(1, 21), dtype=pl.Float64)})
    result = frame.select(factory(*positional, **keywords))
    assert result.columns[0] == COLUMN_X
