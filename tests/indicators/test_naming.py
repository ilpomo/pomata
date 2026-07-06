"""
Column-name-preservation contract for the public indicator factories.

An indicator whose expression is rooted at a scalar literal -- a leading ``100.0`` / ``-100.0`` factor, a ``1.0 +``
offset, or a guard's ``pl.when(...).then(pl.lit(...))`` -- surfaces a ``'literal'`` column, which reads oddly on a bare
``select`` and collides (``DuplicateError``) when two such indicators are selected together. Terminating each at
``.name.keep()`` names the output after its input instead. Unlike the reducing metrics (which all keep the exact input
name), the indicator surface also holds multi-output factories (a ``Struct`` that keeps its first field name) and a few
derived factories that surface a field name, so the invariant held here is the universal one: no output is ever the
``'literal'`` sentinel. ``test_output_preserves_name`` parametrizes over ``pomata.indicators.__all__``, so the guarantee
covers the whole public surface and a newly added indicator is swept in automatically.
"""

import polars as pl
import pytest
from tests.support import COLUMN_X, synthesize_call

from pomata import indicators


@pytest.mark.parametrize("name", indicators.__all__)
def test_output_preserves_name(name: str) -> None:
    """
    Verifies no public factory names its output the ``'literal'`` sentinel (the Polars default when the expression is
    rooted at a scalar literal); every output instead carries the input's name, or a struct's first field name.
    """
    factory = getattr(indicators, name)
    positional, keywords = synthesize_call(factory)
    frame = pl.DataFrame({COLUMN_X: pl.Series(range(1, 21), dtype=pl.Float64)})
    assert frame.select(factory(*positional, **keywords)).columns[0] != "literal"
