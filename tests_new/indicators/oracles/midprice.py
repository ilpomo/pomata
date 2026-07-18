"""
Naive reference oracle for ``pomata.indicators.midprice``.
"""

import math
from collections.abc import Sequence


def reference_midprice(
    high: Sequence[float | None],
    low: Sequence[float | None],
    window: int,
) -> list[float | None]:
    """
    Naive window midprice over Python lists.

    The mean of the maximum ``high`` and the minimum ``low`` of each trailing window, recomputed from scratch as the
    oracle for :func:`pomata.indicators.midprice`.

    Args:
        high: High-price series (may contain ``None`` and ``float('nan')``).
        low: Low-price series (may contain ``None`` and ``float('nan')``).
        window: Number of observations in the moving window. Must be ``>= 1``.

    Returns:
        A list the same length as the inputs: the first ``window - 1`` entries are ``None`` (warm-up); thereafter the
        midpoint of the window's high-low range.

    Raises:
        ValueError: If ``window < 1``.

    Note:
        A window containing a ``None`` in either input yields ``None`` (``None`` taking precedence); a window free of
        ``None`` but containing a ``nan`` yields ``nan``.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")

    results: list[float | None] = []
    for index in range(len(high)):
        if index + 1 < window:
            results.append(None)
            continue
        high_window = high[index + 1 - window : index + 1]
        low_window = low[index + 1 - window : index + 1]
        if any(value is None for value in high_window) or any(value is None for value in low_window):
            results.append(None)
        elif any(isinstance(value, float) and math.isnan(value) for value in [*high_window, *low_window]):
            results.append(math.nan)
        else:
            highs = [value for value in high_window if value is not None]
            lows = [value for value in low_window if value is not None]
            results.append((max(highs) + min(lows)) / 2)
    return results
