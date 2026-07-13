"""
Naive reference oracle for ``pomata.indicators.awesome_oscillator``.
"""

import math
from collections.abc import Sequence

from tests.indicators.oracles.price_median import price_median_reference
from tests.indicators.oracles.sma import sma_reference


def awesome_oscillator_reference(
    high: Sequence[float | None],
    low: Sequence[float | None],
    window_fast: int,
    window_slow: int,
) -> list[float | None]:
    """
    Naive Awesome Oscillator over Python lists.

    The fast simple average of the bar median (:func:`price_median_reference`) minus the slow one, both via
    :func:`sma_reference`, recomputed from the certified leg references as the oracle for
    :func:`pomata.indicators.awesome_oscillator`.

    Args:
        high: High-price series (may contain ``None`` and ``float('nan')``).
        low: Low-price series (may contain ``None`` and ``float('nan')``).
        window_fast: Fast simple-average window. Must be ``>= 1``.
        window_slow: Slow simple-average window. Must be ``>= 1`` and ``>= window_fast``.

    Returns:
        A list the same length as the inputs: ``None`` until both averages are defined, thereafter their difference.

    Raises:
        ValueError: If ``window_fast < 1``, ``window_slow < 1``, or ``window_fast > window_slow``.

    Note:
        A row is ``None`` where either average is ``None`` and ``nan`` where either is ``nan`` (``None`` taking
        precedence), inherited from the median and the two simple averages.
    """
    if window_fast < 1 or window_slow < 1:
        raise ValueError(f"window_fast and window_slow must be >= 1, got {window_fast} and {window_slow}")
    if window_fast > window_slow:
        raise ValueError(f"windows must be ordered window_fast <= window_slow, got {window_fast} and {window_slow}")

    median = price_median_reference(high, low)
    fast = sma_reference(median, window_fast)
    slow = sma_reference(median, window_slow)
    result: list[float | None] = []
    for fast_value, slow_value in zip(fast, slow, strict=True):
        if fast_value is None or slow_value is None:
            result.append(None)
        elif math.isnan(fast_value) or math.isnan(slow_value):
            result.append(math.nan)
        else:
            result.append(fast_value - slow_value)
    return result
