"""
The output-naming contract of every public factory, across all three families.

Every factory terminates at ``.name.keep()``, so the one uniform rule is: **the output column keeps the ROOT name of
the expression's leading input** — never the ``'literal'`` sentinel a scalar-rooted expression would surface, never a
struct field's name, and never an alias the caller put on the *input* (``name.keep`` restores the root, so
``rsi(pl.col("close").alias("x"), 14)`` still lands on ``close``). To name an output, alias the returned expression —
``rsi(...).alias("rsi_14")`` — not the input. Both halves are held for the whole public surface, parametrized over the
three ``__all__`` tuples, so a newly added factory is swept in automatically.
"""

import polars as pl
import pytest
from tests.support import COLUMN_X, synthesize_call

from pomata import indicators, metrics, pnl

_ALL = [
    (family, name)
    for family, module in (("indicators", indicators), ("metrics", metrics), ("pnl", pnl))
    for name in module.__all__
]
_MODULES = {"indicators": indicators, "metrics": metrics, "pnl": pnl}


@pytest.mark.parametrize(("family", "name"), _ALL, ids=[f"{f}.{n}" for f, n in _ALL])
def test_output_keeps_root_name(family: str, name: str) -> None:
    """
    Verifies the output column carries the leading input's root name — not the ``'literal'`` sentinel and not a
    struct field's name.
    """
    factory = getattr(_MODULES[family], name)
    positional, keywords = synthesize_call(factory)
    frame = pl.DataFrame({COLUMN_X: pl.Series(range(1, 21), dtype=pl.Float64)})
    assert frame.select(factory(*positional, **keywords)).columns[0] == COLUMN_X


@pytest.mark.parametrize(("family", "name"), _ALL, ids=[f"{f}.{n}" for f, n in _ALL])
def test_input_alias_is_not_the_output_name(family: str, name: str) -> None:
    """
    Verifies an alias on the *input* never names the output (``name.keep`` restores the root, uniformly): the
    contract is to alias the returned expression, so an aliased input cannot silently land the result on an
    unexpected column.
    """
    factory = getattr(_MODULES[family], name)
    positional, keywords = synthesize_call(factory)
    aliased = [p.alias("user_alias") if isinstance(p, pl.Expr) else p for p in positional]
    keywords = {k: (v.alias("user_alias") if isinstance(v, pl.Expr) else v) for k, v in keywords.items()}
    frame = pl.DataFrame({COLUMN_X: pl.Series(range(1, 21), dtype=pl.Float64)})
    assert frame.select(factory(*aliased, **keywords)).columns[0] == COLUMN_X
