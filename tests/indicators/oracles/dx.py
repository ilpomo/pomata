"""
Naive reference oracle for ``pomata.indicators.dx``.
"""

import math
from collections.abc import Sequence

from tests.indicators.oracles.di_minus import reference_di_minus
from tests.indicators.oracles.di_plus import reference_di_plus


def reference_dx(
    high: Sequence[float | None],
    low: Sequence[float | None],
    close: Sequence[float | None],
    window: int,
) -> list[float | None]:
    """
    Naive Directional Index over aligned Python lists.

    ``100 * |di_plus - di_minus| / (di_plus + di_minus)`` (:func:`reference_di_plus`, :func:`reference_di_minus`),
    recomputed as the oracle for :func:`pomata.indicators.dx`. A zero total (both indicators zero) yields ``NaN``.

    Args:
        high: High-price series (may contain ``None`` and ``float('nan')``).
        low: Low-price series (may contain ``None`` and ``float('nan')``); same length as ``high``.
        close: Close-price series (may contain ``None`` and ``float('nan')``); same length as ``high``.
        window: Number of observations in the Wilder moving window. Must be ``>= 1``.

    Returns:
        A list the same length as the inputs: the directional index, ``None`` through the warm-up.

    Raises:
        ValueError: If ``window < 1``.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    plus = reference_di_plus(high, low, close, window)
    minus = reference_di_minus(high, low, close, window)
    result: list[float | None] = []
    for plus_value, minus_value in zip(plus, minus, strict=True):
        if plus_value is None or minus_value is None:
            result.append(None)
        elif math.isnan(plus_value) or math.isnan(minus_value):
            result.append(math.nan)
        else:
            denominator = plus_value + minus_value
            result.append(math.nan if denominator == 0.0 else 100.0 * abs(plus_value - minus_value) / denominator)
    return result
