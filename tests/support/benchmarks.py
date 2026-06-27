"""
Timing helper for the opt-in ``benchmark`` tier: the fastest of repeated evaluations of an expression over a frame.

A plain function rather than a pytest fixture so it composes with the per-family ``test_benchmark`` modules, each of
which builds its own ``size``-row frame and passes it in.
"""

import time
from collections.abc import Callable

import polars as pl


def fastest_eval(frame: pl.DataFrame, build: Callable[[], pl.Expr]) -> float:
    """
    The fastest of three evaluations of ``build`` over ``frame``, in seconds (the min is robust to load spikes).

    Args:
        frame: The pre-built frame to evaluate against (each caller sizes its own).
        build: A no-argument factory returning the ``pl.Expr`` to time.

    Returns:
        The minimum wall-clock time, in seconds, across three ``select`` evaluations (after one warm-up pass).
    """
    expr = build().alias("y")
    frame.select(expr)  # warm any caches before timing

    def once() -> float:
        start = time.perf_counter()
        frame.select(expr)
        return time.perf_counter() - start

    return min(once() for _ in range(3))
