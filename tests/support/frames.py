"""
Materialize test inputs into a Polars frame, evaluate the expression under test, and read the result back.

These adapters turn raw input lists and a ``pl.Expr`` into the materialized output a test compares against its oracle.
They are plain functions rather than pytest fixtures so they compose cleanly with Hypothesis ``@given`` tests (a
function-scoped fixture runs once per test function, not once per generated example, so reusing it inside a property
test would leak state across examples).
"""

from collections.abc import Mapping, Sequence

import polars as pl
from tests.support.columns import COLUMN_X


def apply_expr(
    values: Sequence[float | None],
    expr: pl.Expr,
) -> list[float | None]:
    """
    Materialize an indicator expression over a single ``Float64`` column and return the result as a Python list.

    Builds a one-column eager frame whose column ``"x"`` holds ``values`` as ``Float64`` (so ``None`` becomes a Polars
    ``null`` and ``float('nan')`` stays a distinct ``NaN``), evaluates ``expr`` against it via ``select``, and returns
    the single output column as a list. This is the standard adapter for single-input, single-output indicator factories
    in the test ladder: the test passes raw observations and the ``pl.Expr`` under test (typically built on
    ``pl.col("x")``), and gets back a plain list it can compare against the naive reference oracle or a golden master.

    Args:
        values: The raw observations to load into the ``"x"`` column (may contain ``None`` and ``float('nan')``); the
            column is coerced to ``Float64`` so the ``null``/``NaN`` distinction is preserved.
        expr: The indicator expression to evaluate, typically built on ``pl.col("x")`` (e.g. ``sma(pl.col("x"), 3)``).

    Returns:
        The materialized output column as a list the same length as ``values``, carrying ``None`` for a Polars ``null``
        and ``float('nan')`` for a ``NaN``.
    """
    return materialize({COLUMN_X: values}, expr)


def materialize(
    columns: Mapping[str, Sequence[float | None]],
    expr: pl.Expr,
) -> list[float | None]:
    """
    Materialize a multi-input indicator expression over a ``Float64`` frame and return the single output as a list.

    Builds an eager frame from ``columns`` (each name to its values, coerced to ``Float64`` so the ``null`` / ``NaN``
    distinction is preserved), evaluates ``expr`` against it, and returns the output column as a list. The multi-input
    counterpart of :func:`apply_expr`: a test passes the named price columns and the ``pl.Expr`` under test (built on
    ``pl.col(name)`` for the same names) and gets back a plain list to compare against the naive oracle or a golden.

    Args:
        columns: The named input columns (e.g. ``{HIGH: highs, LOW: lows, CLOSE: closes}``); each is coerced to
            ``Float64``.
        expr: The indicator expression to evaluate, built on ``pl.col(name)`` for the same names.

    Returns:
        The materialized output column as a list the same length as the inputs.
    """
    frame = pl.DataFrame({name: pl.Series(name, values, dtype=pl.Float64) for name, values in columns.items()})
    return frame.select(expr.alias("y"))["y"].to_list()


def materialize_struct(
    columns: Mapping[str, Sequence[float | None]],
    expr: pl.Expr,
    fields: tuple[str, ...],
) -> dict[str, list[float | None]]:
    """
    Materialize a multi-output (struct) indicator expression and return one list per struct field.

    Like :func:`materialize`, but ``expr`` returns a ``pl.struct``; each named field is read out with
    ``.struct.field(...)`` into its own list, so the result mirrors a naive oracle's dict-of-lists output.

    Args:
        columns: The named input columns; each is coerced to ``Float64``.
        expr: The struct-valued indicator expression to evaluate.
        fields: The struct field names to extract, in the indicator's canonical order.

    Returns:
        A dict mapping each field in ``fields`` to its materialized list, all the same length as the inputs.
    """
    frame = pl.DataFrame({name: pl.Series(name, values, dtype=pl.Float64) for name, values in columns.items()})
    return {field: frame.select(expr.struct.field(field).alias("y"))["y"].to_list() for field in fields}


def count_leading_nulls(values: Sequence[float | None]) -> int:
    """
    The length of the leading ``None`` (warm-up) run, stopping at the first non-null value.

    The shared warm-up check for the property tier: a windowed or recursive indicator emits ``None`` for its warm-up
    rows, so counting that leading run lets a test pin it against the indicator's documented warm-up length.

    Args:
        values: A materialized output list (e.g. from :func:`apply_expr`), carrying ``None`` for a Polars ``null``.

    Returns:
        The number of leading ``None`` values before the first non-null (``0`` if the first value is non-null).
    """
    count = 0
    for value in values:
        if value is None:
            count += 1
        else:
            break
    return count
