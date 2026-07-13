"""
Naive reference oracle for ``pomata.indicators.standard_deviation_rolling``.
"""

import math
from collections.abc import Sequence

from tests_new.indicators.oracles.variance_rolling import variance_rolling_reference


def standard_deviation_rolling_reference(
    expr: Sequence[float | None],
    window: int,
    ddof: int = 0,
) -> list[float | None]:
    """
    Naive rolling standard deviation over a Python list.

    The square root of :func:`variance_rolling_reference`, recomputed from scratch as the oracle for
    :func:`pomata.indicators.standard_deviation_rolling`.

    Args:
        expr: Input series (may contain ``None`` and ``float('nan')``).
        window: Number of observations in the moving window. Must be ``>= 1``.
        ddof: Delta degrees of freedom; the divisor is ``window - ddof``. ``0`` is the population standard deviation.
            Must be ``< window``.

    Returns:
        A list the same length as ``expr``: the first ``window - 1`` entries are ``None`` (warm-up); thereafter the
        square root of the trailing-window variance (``None`` where the variance itself is ``None``).

    Raises:
        ValueError: If ``window < 1``, or if ``ddof >= window`` (the divisor ``window - ddof`` would be non-positive).
    """
    results: list[float | None] = []
    for value in variance_rolling_reference(expr, window, ddof):
        if value is None or math.isnan(value):
            results.append(value)
        else:
            results.append(math.sqrt(value))
    return results
