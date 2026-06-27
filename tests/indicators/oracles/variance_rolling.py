"""
Naive reference oracle for ``pomata.indicators.variance_rolling``.
"""

import math
from collections.abc import Sequence


def variance_rolling_reference(
    expr: Sequence[float | None],
    window: int,
    ddof: int = 0,
) -> list[float | None]:
    """
    Naive rolling variance over a Python list.

    The mean squared deviation of each trailing window from its own mean, recomputed from scratch as the oracle for
    :func:`pomata.indicators.variance_rolling`.

    Args:
        expr: Input series (may contain ``None`` and ``float('nan')``).
        window: Number of observations in the moving window. Must be ``>= 1``.
        ddof: Delta degrees of freedom; the divisor is ``window - ddof``. ``0`` is the population variance. Must be
            ``< window``.

    Returns:
        A list the same length as ``expr``: the first ``window - 1`` entries are ``None`` (warm-up); thereafter the
        trailing-window variance.

    Raises:
        ValueError: If ``window < 1``, or if ``ddof >= window`` (the divisor ``window - ddof`` would be non-positive).

    Note:
        Edge-case behavior:

        - **Null** — a window that contains a ``None`` yields ``None`` (Polars' ``min_samples=window``); ``None`` takes
          precedence over ``nan``.
        - **NaN** — a window free of ``None`` but containing a ``nan`` yields ``nan``.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    if ddof >= window:
        raise ValueError(f"ddof must be < window, got ddof={ddof} and window={window}")

    results: list[float | None] = []
    for index in range(len(expr)):
        if index + 1 < window:
            results.append(None)
            continue
        window_values = expr[index + 1 - window : index + 1]
        if any(value is None for value in window_values):
            results.append(None)
        elif any(isinstance(value, float) and math.isnan(value) for value in window_values):
            results.append(math.nan)
        else:
            numbers = [value for value in window_values if value is not None]
            mean = sum(numbers) / window
            squared = sum((value - mean) ** 2 for value in numbers)
            results.append(squared / (window - ddof))
    return results
