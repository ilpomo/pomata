"""
Naive reference oracle for ``pomata.indicators.atr_normalized``.
"""

import math
from collections.abc import Sequence

from tests_new.indicators.oracles.atr import reference_atr


def reference_atr_normalized(
    high: Sequence[float | None],
    low: Sequence[float | None],
    close: Sequence[float | None],
    window: int,
) -> list[float | None]:
    """
    Naive Normalized ATR over aligned Python lists.

    The :func:`reference_atr` as a percentage of the close, ``100 * atr / close``, recomputed as the oracle for
    :func:`pomata.indicators.atr_normalized`.

    Args:
        high: High-price series (may contain ``None`` and ``float('nan')``).
        low: Low-price series (may contain ``None`` and ``float('nan')``); same length as ``high``.
        close: Close-price series (may contain ``None`` and ``float('nan')``); same length as ``high``.
        window: Number of observations in the Wilder moving window. Must be ``>= 1``.

    Returns:
        A list the same length as the inputs: ``100 * atr / close``, ``None`` through the warm-up and wherever the
        ATR or the ``close`` is missing.

    Raises:
        ValueError: If ``window < 1``.
    """
    average_true_range = reference_atr(high, low, close, window)
    result: list[float | None] = []
    for atr_value, close_value in zip(average_true_range, close, strict=True):
        if atr_value is None or close_value is None:
            result.append(None)
        elif math.isnan(atr_value) or math.isnan(close_value):
            result.append(math.nan)
        else:
            result.append(100.0 * atr_value / close_value)
    return result
