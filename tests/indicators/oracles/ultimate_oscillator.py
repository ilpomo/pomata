"""
Naive reference oracle for ``pomata.indicators.ultimate_oscillator``.
"""

import math
from collections.abc import Sequence

from tests.indicators.oracles._helpers import difference


def _min_horizontal(*values: float | None) -> float | None:
    """
    Polars' ``min_horizontal``: drop nulls; ``NaN`` behaves as ``+inf`` (ignored unless every non-null value is
    ``NaN``); ``None`` if all inputs are null.
    """
    present = [value for value in values if value is not None]
    if not present:
        return None
    finite = [value for value in present if not math.isnan(value)]
    if not finite:
        return math.nan
    return min(finite)


def _max_horizontal(*values: float | None) -> float | None:
    """
    Polars' ``max_horizontal``: drop nulls; ``NaN`` behaves as ``+inf`` (wins the maximum, so any non-null ``NaN`` gives
    ``NaN``); ``None`` if all inputs are null.
    """
    present = [value for value in values if value is not None]
    if not present:
        return None
    if any(math.isnan(value) for value in present):
        return math.nan
    return max(present)


def _rolling_sum(values: Sequence[float | None], window: int) -> list[float | None]:
    """
    Polars' ``rolling_sum``: ``None`` through the warm-up; ``None`` if the window holds any null; ``NaN`` if it holds
    any ``NaN``; otherwise the sum.
    """
    result: list[float | None] = []
    for index in range(len(values)):
        if index < window - 1:
            result.append(None)
            continue
        window_values = values[index - window + 1 : index + 1]
        if any(value is None for value in window_values):
            result.append(None)
        elif any(isinstance(value, float) and math.isnan(value) for value in window_values):
            result.append(math.nan)
        else:
            result.append(sum(value for value in window_values if value is not None))
    return result


def reference_ultimate_oscillator(
    high: Sequence[float | None],
    low: Sequence[float | None],
    close: Sequence[float | None],
    window_short: int = 7,
    window_medium: int = 14,
    window_long: int = 28,
) -> list[float | None]:
    """
    Naive Ultimate Oscillator over aligned Python lists.

    Buying pressure (``close - true low``) summed over three windows as a fraction of the true-range sum, blended
    ``4 : 2 : 1`` and scaled to ``[0, 100]``, recomputed as the oracle for
    :func:`pomata.indicators.ultimate_oscillator`.

    Args:
        high: High-price series (may contain ``None`` and ``float('nan')``).
        low: Low-price series (may contain ``None`` and ``float('nan')``); same length as ``high``.
        close: Close-price series (may contain ``None`` and ``float('nan')``); same length as ``high``.
        window_short: Number of observations in the short averaging window. Must be ``>= 1``.
        window_medium: Number of observations in the medium averaging window. Must be ``>= 1``.
        window_long: Number of observations in the long averaging window. Must be ``>= 1``.

    Returns:
        A list the same length as the inputs: the Ultimate Oscillator, ``None`` through the warm-up.

    Raises:
        ValueError: If ``window_short < 1``, ``window_medium < 1``, or ``window_long < 1``.
    """
    if window_short < 1:
        raise ValueError(f"window_short must be >= 1, got {window_short}")
    if window_medium < 1:
        raise ValueError(f"window_medium must be >= 1, got {window_medium}")
    if window_long < 1:
        raise ValueError(f"window_long must be >= 1, got {window_long}")
    buying_pressure: list[float | None] = []
    true_range: list[float | None] = []
    for index in range(len(close)):
        previous = close[index - 1] if index > 0 else None
        true_low = _min_horizontal(low[index], previous)
        true_high = _max_horizontal(high[index], previous)
        buying_pressure.append(difference(close[index], true_low))
        true_range.append(difference(true_high, true_low))

    def averaged(window: int) -> list[float | None]:
        numerators = _rolling_sum(buying_pressure, window)
        denominators = _rolling_sum(true_range, window)
        result: list[float | None] = []
        for numerator, denominator in zip(numerators, denominators, strict=True):
            if numerator is None or denominator is None:
                result.append(None)
            elif math.isnan(numerator) or math.isnan(denominator):
                result.append(math.nan)
            elif denominator == 0.0:
                # Exactly-zero true range: the 0/0 degenerate (no buying pressure either) is NaN; a finite buying
                # pressure over it -- the missing-low fallback -- surfaces as +/-inf, matching natural IEEE-754.
                result.append(math.nan if numerator == 0.0 else math.copysign(math.inf, numerator))
            else:
                result.append(numerator / denominator)
        return result

    short = averaged(window_short)
    medium = averaged(window_medium)
    long_ = averaged(window_long)
    oscillator: list[float | None] = []
    for short_value, medium_value, long_value in zip(short, medium, long_, strict=True):
        if short_value is None or medium_value is None or long_value is None:
            oscillator.append(None)
        elif math.isnan(short_value) or math.isnan(medium_value) or math.isnan(long_value):
            oscillator.append(math.nan)
        else:
            oscillator.append(100.0 * (4.0 * short_value + 2.0 * medium_value + long_value) / 7.0)
    return oscillator
