"""
Naive reference oracle for ``pomata.indicators.vwap``.
"""

import math
from collections.abc import Sequence

from tests.indicators.oracles.price_typical import price_typical_reference


def _multiply(left: float | None, right: float | None) -> float | None:
    """
    The elementwise product under Polars rules: ``None`` if either side is ``None``, else ``nan`` if either is.
    """
    if left is None or right is None:
        return None
    if math.isnan(left) or math.isnan(right):
        return math.nan
    return left * right


def _cumulative_sum(values: Sequence[float | None]) -> list[float | None]:
    """
    Matches Polars ``cum_sum``: a ``None`` emits ``None`` at its row regardless of state and contributes nothing to the
    running total; a ``nan`` poisons the running total (so every later non-``None`` row is ``nan``).
    """
    result: list[float | None] = []
    running = 0.0
    poisoned = False
    for value in values:
        if value is None:
            result.append(None)
        elif poisoned:
            result.append(math.nan)
        elif math.isnan(value):
            poisoned = True
            result.append(math.nan)
        else:
            running += value
            result.append(running)
    return result


def vwap_reference(
    high: Sequence[float | None],
    low: Sequence[float | None],
    close: Sequence[float | None],
    volume: Sequence[float | None],
) -> list[float | None]:
    """
    Naive Volume-Weighted Average Price over Python lists.

    The cumulative sum of the typical price (:func:`price_typical_reference`) times volume, over the cumulative sum of
    the volume masked to the same null bars, recomputed from scratch as the oracle for :func:`pomata.indicators.vwap`.
    Masking the denominator means a null in any price input drops that bar from both sums together (a clean missing
    observation), not just the numerator. The two cumulative sums replicate Polars' ``cum_sum`` exactly (``None``
    carried across, ``nan`` poisoning the rest), and the ratio follows IEEE (``0 / 0`` is ``nan``).

    Args:
        high: High-price series (may contain ``None`` and ``float('nan')``).
        low: Low-price series (may contain ``None`` and ``float('nan')``).
        close: Close-price series (may contain ``None`` and ``float('nan')``).
        volume: Volume series (may contain ``None`` and ``float('nan')``).

    Returns:
        A list the same length as the inputs: the running VWAP, ``nan`` where the cumulative volume is still zero.

    Note:
        A row is ``None`` where the numerator or denominator cumulative sum is ``None``; otherwise ``nan`` where either
        is ``nan`` or where the cumulative volume is zero (``0 / 0``).
    """
    typical = price_typical_reference(high, low, close)
    weighted = [_multiply(t, v) for t, v in zip(typical, volume, strict=True)]
    # The denominator drops a bar exactly when the numerator does: a null weighted term (a null price or volume input)
    # nulls the bar's volume too, so both cumulative sums skip it together.
    volume_masked = [None if product is None else value for product, value in zip(weighted, volume, strict=True)]
    numerator = _cumulative_sum(weighted)
    denominator = _cumulative_sum(volume_masked)
    result: list[float | None] = []
    for num, den in zip(numerator, denominator, strict=True):
        if num is None or den is None:
            result.append(None)
        elif math.isnan(num) or math.isnan(den):
            result.append(math.nan)
        elif den == 0.0:
            result.append(math.nan if num == 0.0 else math.copysign(math.inf, num))
        else:
            result.append(num / den)
    return result
