"""
Column-name-preservation contract for the public pnl factories.

Every factory keeps the input column's name on its output. A factory whose expression is rooted at a literal -- e.g.
``equity_curve``'s ``(1.0 + returns).cum_prod()`` -- would otherwise surface as a ``'literal'`` column, which both
reads oddly on a bare ``select`` and collides (``DuplicateError``) when two such factories are selected together.
``test_output_preserves_name`` parametrizes over ``pomata.pnl.__all__``, so the guarantee covers the whole public
surface and a newly added function is swept in automatically.
"""

import polars as pl
import pytest
from tests.support import COLUMN_X, synthesize_call

from pomata import pnl


@pytest.mark.parametrize("name", pnl.__all__)
def test_output_preserves_name(name: str) -> None:
    """
    Verifies every public factory names its output column after the input (``COLUMN_X``), never ``'literal'``.
    """
    factory = getattr(pnl, name)
    positional, keywords = synthesize_call(factory)
    frame = pl.DataFrame({COLUMN_X: pl.Series(range(1, 21), dtype=pl.Float64)})
    result = frame.select(factory(*positional, **keywords))
    assert result.columns[0] == COLUMN_X
