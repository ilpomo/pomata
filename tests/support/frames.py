"""
Frame construction for the rungs: the per-role probe-column builders, the deterministic probe frame, the materialize
adapters, and the bar-tuple splitters.

These turn declared input roles and a ``pl.Expr`` into the materialized output a rung compares against its oracle. They
are plain functions rather than pytest fixtures so they compose cleanly with Hypothesis ``@given`` tests (a
function-scoped fixture runs once per test function, not once per generated example, so reusing it inside a property
test would leak state across examples). Each role builds a distinctly-named ``Float64`` column, so a multi-input factory
that mixes its columns is caught by construction rather than passing on identical inputs.
"""

import math
from collections.abc import Callable, Mapping, Sequence

import polars as pl

COLUMN_X = "x"

# The input-column roles the deterministic probe frame knows how to synthesize; a declaration's ``inputs`` draw from
# these, and :data:`KNOWN_ROLES` is the set the declaration constructor validates against.
ROLE_BUILDERS: dict[str, Callable[[int], list[float]]] = {
    "high": lambda n: [float(i) + 1.5 for i in range(n)],
    "low": lambda n: [float(i) + 0.5 for i in range(n)],
    "open": lambda n: [float(i) + 0.9 for i in range(n)],
    "close": lambda n: [float(i) + 1.1 for i in range(n)],
    "volume": lambda n: [100.0 + float(i) for i in range(n)],
    "expr": lambda n: [float(i) + 1.0 for i in range(n)],
    # A non-monotone oscillating price series: a strictly-monotone probe saturates any RSI to a flat 100, which then
    # drives a stochastic-of-RSI to a 0/0 all-NaN output the flow rungs cannot read. This role gives such a function a
    # varied, well-defined baseline instead.
    "wave": lambda n: [100.0 + 10.0 * math.sin(0.7 * float(i)) for i in range(n)],
    "price": lambda n: [float(i) + 10.0 for i in range(n)],
    # A strictly-positive compounding path whose step is small enough that large frames stay far below the float64
    # ceiling (1.0001 ** n overflows only past ~7.1M rows; 1.02 ** n blows up at ~35.7k).
    "equity_curve": lambda n: [100.0 * (1.0001 ** float(i)) for i in range(n)],
    "returns": lambda n: [0.01 if i % 2 == 0 else -0.005 for i in range(n)],
    "benchmark": lambda n: [0.008 if i % 2 == 0 else -0.004 for i in range(n)],
    "asset_returns": lambda n: [0.01 if i % 2 == 0 else -0.005 for i in range(n)],
    "weight": lambda n: [0.5 + 0.01 * float(i) for i in range(n)],
    # Alternating long/short so the short-only branches (cost_borrow's max(-q, 0)) meet real shorts on the probe: an
    # all-positive quantity would leave them identically zero and their scale axes vacuously green.
    "quantity": lambda n: [(-1.0) ** i * (10.0 + float(i % 3)) for i in range(n)],
    "cost": lambda n: [0.1 for _ in range(n)],
    "dividend_per_share": lambda n: [0.05 for _ in range(n)],
    "returns_gross": lambda n: [0.01 if i % 2 == 0 else -0.005 for i in range(n)],
    "funding_rate": lambda n: [0.0001 for _ in range(n)],
    "pnl_gross": lambda n: [10.0 + float(i) for i in range(n)],
}

KNOWN_ROLES: frozenset[str] = frozenset(ROLE_BUILDERS)


def probe_frame(inputs: tuple[str, ...], length: int) -> pl.DataFrame:
    """A well-conditioned deterministic frame, one distinctly-named ``Float64`` column per declared input role."""
    return pl.DataFrame({role: pl.Series(ROLE_BUILDERS[role](length), dtype=pl.Float64) for role in inputs})


def apply_expr(values: Sequence[float | None], expr: pl.Expr) -> list[float | None]:
    """
    Materialize an expression over a single ``Float64`` column named ``"x"`` and return the result as a Python list.

    Builds a one-column eager frame whose column holds ``values`` as ``Float64`` (so ``None`` becomes a Polars ``null``
    and ``float('nan')`` stays a distinct ``NaN``), evaluates ``expr`` against it, and returns the single output column
    as a list the same length as ``values``.

    Args:
        values: The raw observations to load into the ``"x"`` column (may contain ``None`` and ``float('nan')``).
        expr: The expression to evaluate, typically built on ``pl.col("x")``.

    Returns:
        The materialized output column as a list, carrying ``None`` for a Polars ``null`` and ``float('nan')`` for a
        ``NaN``.
    """
    return materialize({COLUMN_X: values}, expr)


def materialize(columns: Mapping[str, Sequence[float | None]], expr: pl.Expr) -> list[float | None]:
    """
    Materialize a multi-input expression over a ``Float64`` frame and return the single output column as a list.

    Builds an eager frame from ``columns`` (each name to its values, coerced to ``Float64`` so the ``null`` / ``NaN``
    distinction is preserved), evaluates ``expr`` against it, and returns the output column as a list.

    Args:
        columns: The named input columns; each is coerced to ``Float64``.
        expr: The expression to evaluate, built on ``pl.col(name)`` for the same names.

    Returns:
        The materialized output column as a list the same length as the inputs.
    """
    frame = pl.DataFrame({name: pl.Series(name, values, dtype=pl.Float64) for name, values in columns.items()})
    return frame.select(expr.alias("y"))["y"].to_list()


def count_leading_nulls(values: Sequence[float | None]) -> int:
    """
    The length of the leading ``None`` (warm-up) run, stopping at the first non-null value.

    A windowed or recursive function emits ``None`` for its warm-up rows, so counting that leading run lets a rung
    pin it against the declared warm-up length.

    Args:
        values: A materialized output list, carrying ``None`` for a Polars ``null``.

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


def split_pairs[T](rows: Sequence[tuple[T, T]]) -> tuple[list[T], list[T]]:
    """Unzip a list of 2-tuples (e.g. ``high`` / ``low`` bars) into two aligned columns."""
    return [row[0] for row in rows], [row[1] for row in rows]


def split_triples[T](rows: Sequence[tuple[T, T, T]]) -> tuple[list[T], list[T], list[T]]:
    """Unzip a list of 3-tuples (e.g. ``high`` / ``low`` / ``close`` bars) into three aligned columns."""
    return [row[0] for row in rows], [row[1] for row in rows], [row[2] for row in rows]


def split_quads[T](rows: Sequence[tuple[T, T, T, T]]) -> tuple[list[T], list[T], list[T], list[T]]:
    """Unzip a list of 4-tuples (OHLC or HLCV bars) into four aligned columns."""
    return (
        [row[0] for row in rows],
        [row[1] for row in rows],
        [row[2] for row in rows],
        [row[3] for row in rows],
    )
