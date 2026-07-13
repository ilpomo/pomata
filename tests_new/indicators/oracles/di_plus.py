"""
Naive reference oracle for ``pomata.indicators.di_plus``.
"""

import math
from collections.abc import Sequence

from tests_new.indicators.oracles.atr import atr_reference
from tests_new.indicators.oracles.dm_plus import dm_plus_reference


def di_plus_reference(
    high: Sequence[float | None],
    low: Sequence[float | None],
    close: Sequence[float | None],
    window: int,
) -> list[float | None]:
    """
    Naive Plus Directional Indicator over aligned Python lists.

    ``100 * dm_plus / atr`` (the smoothed plus directional movement, :func:`dm_plus_reference`, as a percentage of the
    average true range, :func:`atr_reference`), recomputed as the oracle for :func:`pomata.indicators.di_plus`.

    Args:
        high: High-price series (may contain ``None`` and ``float('nan')``).
        low: Low-price series (may contain ``None`` and ``float('nan')``); same length as ``high``.
        close: Close-price series (may contain ``None`` and ``float('nan')``); same length as ``high``.
        window: Number of observations in the Wilder moving window. Must be ``>= 1``.

    Returns:
        A list the same length as the inputs: the plus directional indicator, ``None`` through the warm-up.

    Raises:
        ValueError: If ``window < 1``.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    movement = dm_plus_reference(high, low, window)
    average_true_range = atr_reference(high, low, close, window)
    result: list[float | None] = []
    for movement_value, atr_value in zip(movement, average_true_range, strict=True):
        if movement_value is None or atr_value is None:
            result.append(None)
        elif math.isnan(movement_value) or math.isnan(atr_value):
            result.append(math.nan)
        elif atr_value == 0.0:
            result.append(math.nan if movement_value == 0.0 else math.inf)
        else:
            result.append(100.0 * movement_value / atr_value)
    return result
