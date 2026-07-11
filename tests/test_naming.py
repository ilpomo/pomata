"""
The output-naming contract of every public factory, across all three families.

Every factory terminates at ``.name.keep()``, so the one uniform rule is: **the output column keeps the ROOT name of
the expression's leading input** — never the ``'literal'`` sentinel a scalar-rooted expression would surface, never a
struct field's name, and never an alias the caller put on the *input* (``name.keep`` restores the root, so
``rsi(pl.col("close").alias("x"), 14)`` still lands on ``close``). To name an output, alias the returned expression —
``rsi(...).alias("rsi_14")`` — not the input. Both halves are held for the whole public surface, parametrized over the
three ``__all__`` tuples, so a newly added factory is swept in automatically.
"""

import inspect

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


# The eleven multi-input functions whose output lands on a non-first input's column — the expression's first column
# leaf, pinned so a refactor can never silently move a landing column (docs/concepts.md §1 documents the same list).
_NON_FIRST_ROOT_INDEX: dict[str, int] = {
    "accumulation_distribution": 2,
    "accumulation_distribution_oscillator": 2,
    "balance_of_power": 3,
    "chaikin_money_flow": 2,
    "di_minus": 1,
    "dm_minus": 1,
    "donchian_channels": 1,
    "keltner_channels": 2,
    "stochastic_fast": 2,
    "stochastic_slow": 2,
    "ultimate_oscillator": 2,
}


def _distinct_call(family: str, name: str, *, alias: bool = False) -> tuple[list[object], dict[str, object], list[str]]:
    """Synthesize a call with a DISTINCT column per ``pl.Expr`` parameter, so the landing column is observable."""
    factory = getattr(_MODULES[family], name)
    positional: list[object] = []
    keywords: dict[str, object] = {}
    columns: list[str] = []
    for parameter in inspect.signature(factory).parameters.values():
        annotation = str(parameter.annotation)
        value: object
        if "Expr" in annotation:
            column = f"column_{len(columns)}"
            columns.append(column)
            value = pl.col(column).alias("user_alias") if alias else pl.col(column)
        elif parameter.default is not inspect.Parameter.empty:
            continue
        elif annotation == "<class 'int'>":
            value = 3
        else:
            value = 0.1
        if parameter.kind == inspect.Parameter.KEYWORD_ONLY:
            keywords[parameter.name] = value
        else:
            positional.append(value)
    return positional, keywords, columns


def _distinct_frame(columns: list[str]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            column: [float(10 * (index + 1) + row) + (0.5 if index % 2 else 0.0) for row in range(8)]
            for index, column in enumerate(columns)
        }
    )


@pytest.mark.parametrize(("family", "name"), _ALL)
def test_output_root_is_pinned(family: str, name: str) -> None:
    """With a distinct column per input, the output lands exactly on its pinned column — for every function."""
    factory = getattr(_MODULES[family], name)
    positional, keywords, columns = _distinct_call(family, name)
    expected = columns[_NON_FIRST_ROOT_INDEX.get(name, 0)]
    assert _distinct_frame(columns).select(factory(*positional, **keywords)).columns == [expected]


@pytest.mark.parametrize(("family", "name"), _ALL)
def test_no_input_alias_leaks_on_any_input(family: str, name: str) -> None:
    """An alias on EVERY input still never becomes the output name; the pinned root is restored."""
    factory = getattr(_MODULES[family], name)
    positional, keywords, columns = _distinct_call(family, name, alias=True)
    expected = columns[_NON_FIRST_ROOT_INDEX.get(name, 0)]
    assert _distinct_frame(columns).select(factory(*positional, **keywords)).columns == [expected]
