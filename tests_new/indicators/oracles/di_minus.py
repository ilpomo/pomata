"""
Naive reference oracle for ``pomata.indicators.di_minus``.
"""

import math
from collections.abc import Sequence

from tests_new.indicators.oracles.atr import reference_atr
from tests_new.indicators.oracles.dm_minus import reference_dm_minus


def reference_di_minus(
    high: Sequence[float | None],
    low: Sequence[float | None],
    close: Sequence[float | None],
    window: int,
) -> list[float | None]:
    """
    Naive Minus Directional Indicator over aligned Python lists.

    ``100 * dm_minus / atr`` (the smoothed minus directional movement, :func:`reference_dm_minus`, as a percentage of
    the average true range, :func:`reference_atr`), recomputed as the oracle for :func:`pomata.indicators.di_minus`.

    Args:
        high: High-price series (may contain ``None`` and ``float('nan')``).
        low: Low-price series (may contain ``None`` and ``float('nan')``); same length as ``high``.
        close: Close-price series (may contain ``None`` and ``float('nan')``); same length as ``high``.
        window: Number of observations in the Wilder moving window. Must be ``>= 1``.

    Returns:
        A list the same length as the inputs: the minus directional indicator, ``None`` through the warm-up.

    Raises:
        ValueError: If ``window < 1``.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    movement = reference_dm_minus(high, low, window)
    average_true_range = reference_atr(high, low, close, window)
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
