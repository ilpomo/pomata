"""
Naive reference oracle for ``pomata.indicators.adxr``.
"""

import math
from collections.abc import Sequence

from tests_new.indicators.oracles.adx import adx_reference


def adxr_reference(
    high: Sequence[float | None],
    low: Sequence[float | None],
    close: Sequence[float | None],
    window: int,
) -> list[float | None]:
    """
    Naive Average Directional Index Rating over aligned Python lists.

    The mean of the :func:`adx_reference` and its value ``window`` rows earlier, recomputed as the oracle for
    :func:`pomata.indicators.adxr`.

    Args:
        high: High-price series (may contain ``None`` and ``float('nan')``).
        low: Low-price series (may contain ``None`` and ``float('nan')``); same length as ``high``.
        close: Close-price series (may contain ``None`` and ``float('nan')``); same length as ``high``.
        window: Number of observations in the Wilder window, and the averaging look-back. Must be ``>= 1``.

    Returns:
        A list the same length as the inputs: the average directional index rating, ``None`` through the warm-up.

    Raises:
        ValueError: If ``window < 1``.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    average = adx_reference(high, low, close, window)
    result: list[float | None] = []
    for index in range(len(average)):
        current = average[index]
        past = average[index - window] if index - window >= 0 else None
        if current is None or past is None:
            result.append(None)
        elif math.isnan(current) or math.isnan(past):
            result.append(math.nan)
        else:
            result.append((current + past) / 2.0)
    return result
