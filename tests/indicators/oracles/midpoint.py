"""
Naive reference oracle for ``pomata.indicators.midpoint``.
"""

import math
from collections.abc import Sequence


def reference_midpoint(
    expr: Sequence[float | None],
    window: int,
) -> list[float | None]:
    """
    Naive window midpoint over a Python list.

    The mean of the maximum and minimum of each trailing window, recomputed from scratch as the oracle for
    :func:`pomata.indicators.midpoint`.

    Args:
        expr: Input series (may contain ``None`` and ``float('nan')``).
        window: Number of observations in the moving window. Must be ``>= 1``.

    Returns:
        A list the same length as ``expr``: the first ``window - 1`` entries are ``None`` (warm-up); thereafter the
        midpoint of the window's range.

    Raises:
        ValueError: If ``window < 1``.

    Note:
        Edge-case behavior:

        - **Null** — a window containing a ``None`` yields ``None`` (Polars' ``min_samples=window``); ``None`` takes
          precedence over ``nan``.
        - **NaN** — a window free of ``None`` but containing a ``nan`` yields ``nan``.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")

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
            results.append((max(numbers) + min(numbers)) / 2)
    return results
