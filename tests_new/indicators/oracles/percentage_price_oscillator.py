"""
Naive reference oracle for ``pomata.indicators.percentage_price_oscillator``.
"""

import math
from collections.abc import Sequence

from tests_new.indicators.oracles.ema import ema_reference


def percentage_price_oscillator_reference(
    close: Sequence[float | None],
    window_fast: int,
    window_slow: int,
) -> list[float | None]:
    """
    Naive Percentage Price Oscillator over a Python list.

    The fast-minus-slow exponential moving average gap (:func:`ema_reference`) as a percentage of the slow EMA,
    recomputed from scratch as the oracle for :func:`pomata.indicators.percentage_price_oscillator`.

    Args:
        close: Close-price series (may contain ``None`` and ``float('nan')``); the tested inputs are positive, so the
            slow EMA in the denominator stays positive.
        window_fast: Span of the fast EMA. Must be ``>= 1``.
        window_slow: Span of the slow EMA. Must be ``>= 1``.

    Returns:
        A list the same length as ``close``: ``100 * (fast - slow) / slow``, ``None`` until the slow EMA is defined.

    Raises:
        ValueError: If ``window_fast < 1`` or ``window_slow < 1``.

    Note:
        When the slow EMA in the denominator is ``0`` the ratio divides by zero following IEEE-754: a zero gap (``0 /
        0``) is ``nan`` and a non-zero gap over zero is ``+/-inf`` (the sign tracks the gap relative to the signed
        zero). The tested inputs are positive, so the slow EMA stays positive and this branch is reached only by a
        dedicated edge test; the guard keeps the oracle from raising ``ZeroDivisionError`` there.
    """
    if window_fast < 1 or window_slow < 1:
        raise ValueError(f"window_fast and window_slow must be >= 1, got {window_fast} and {window_slow}")
    fast = ema_reference(close, window_fast)
    slow = ema_reference(close, window_slow)
    results: list[float | None] = []
    for fast_value, slow_value in zip(fast, slow, strict=True):
        if fast_value is None or slow_value is None:
            results.append(None)
        elif math.isnan(fast_value) or math.isnan(slow_value):
            results.append(math.nan)
        elif slow_value == 0.0:
            gap = 100.0 * (fast_value - slow_value)
            if gap == 0.0:
                results.append(math.nan)
            else:
                results.append(math.copysign(math.inf, gap) * math.copysign(1.0, slow_value))
        else:
            results.append((fast_value - slow_value) / slow_value * 100.0)
    return results
