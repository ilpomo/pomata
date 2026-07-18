"""
Naive reference oracle for ``pomata.indicators.absolute_price_oscillator``.
"""

import math
from collections.abc import Sequence

from tests.indicators.oracles.ema import reference_ema


def reference_absolute_price_oscillator(
    close: Sequence[float | None],
    window_fast: int,
    window_slow: int,
) -> list[float | None]:
    """
    Naive Absolute Price Oscillator over a Python list.

    The difference of a fast and a slow exponential moving average (:func:`reference_ema`), recomputed from scratch as
    the oracle for :func:`pomata.indicators.absolute_price_oscillator`.

    Args:
        close: Close-price series (may contain ``None`` and ``float('nan')``).
        window_fast: Span of the fast EMA. Must be ``>= 1``.
        window_slow: Span of the slow EMA. Must be ``>= 1``.

    Returns:
        A list the same length as ``close``: the fast EMA minus the slow EMA, ``None`` until the slow EMA is defined.

    Raises:
        ValueError: If ``window_fast < 1`` or ``window_slow < 1``.
    """
    if window_fast < 1 or window_slow < 1:
        raise ValueError(f"window_fast and window_slow must be >= 1, got {window_fast} and {window_slow}")
    fast = reference_ema(close, window_fast)
    slow = reference_ema(close, window_slow)
    results: list[float | None] = []
    for fast_value, slow_value in zip(fast, slow, strict=True):
        if fast_value is None or slow_value is None:
            results.append(None)
        elif math.isnan(fast_value) or math.isnan(slow_value):
            results.append(math.nan)
        else:
            results.append(fast_value - slow_value)
    return results
