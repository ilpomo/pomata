"""
Naive reference oracle for ``pomata.indicators.accumulation_distribution_oscillator``.
"""

import math
from collections.abc import Sequence

from tests.indicators.oracles.accumulation_distribution import accumulation_distribution_reference
from tests.indicators.oracles.ema import ema_reference


def accumulation_distribution_oscillator_reference(
    high: Sequence[float | None],
    low: Sequence[float | None],
    close: Sequence[float | None],
    volume: Sequence[float | None],
    window_fast: int = 3,
    window_slow: int = 10,
) -> list[float | None]:
    """
    Naive Accumulation/Distribution Oscillator over aligned Python lists.

    The fast minus slow :func:`ema_reference` of the :func:`accumulation_distribution_reference` line, recomputed as the
    oracle for :func:`pomata.indicators.accumulation_distribution_oscillator`.

    Args:
        high: High-price series (may contain ``None`` and ``float('nan')``).
        low: Low-price series (may contain ``None`` and ``float('nan')``); same length as ``high``.
        close: Close-price series (may contain ``None`` and ``float('nan')``); same length as ``high``.
        volume: Volume series (may contain ``None`` and ``float('nan')``); same length as ``high``.
        window_fast: Span of the fast EMA. Must be ``>= 1`` and ``<= window_slow``.
        window_slow: Span of the slow EMA. Must be ``>= 1``.

    Returns:
        A list the same length as the inputs: the fast EMA of the AD line minus the slow EMA, ``None`` through the
        ``window_slow - 1`` warm-up rows.

    Raises:
        ValueError: If ``window_fast < 1``, ``window_slow < 1``, or ``window_fast > window_slow``.
    """
    if window_fast < 1 or window_slow < 1:
        raise ValueError(f"window_fast and window_slow must be >= 1, got {window_fast} and {window_slow}")
    if window_fast > window_slow:
        raise ValueError(f"windows must be ordered window_fast <= window_slow, got {window_fast} and {window_slow}")
    line = accumulation_distribution_reference(high, low, close, volume)
    ema_fast = ema_reference(line, window_fast)
    ema_slow = ema_reference(line, window_slow)
    result: list[float | None] = []
    for fast_value, slow_value in zip(ema_fast, ema_slow, strict=True):
        if fast_value is None or slow_value is None:
            result.append(None)
        elif math.isnan(fast_value) or math.isnan(slow_value):
            result.append(math.nan)
        else:
            result.append(fast_value - slow_value)
    return result
