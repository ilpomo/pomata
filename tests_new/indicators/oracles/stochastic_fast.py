"""
Naive reference oracle for ``pomata.indicators.stochastic_fast``.
"""

import math
from collections.abc import Callable, Sequence

from tests_new.indicators.oracles.sma import reference_sma


def _rolling_extreme(
    values: Sequence[float | None],
    period: int,
    reducer: Callable[[list[float]], float],
) -> list[float | None]:
    """
    Polars' ``rolling_min`` / ``rolling_max`` semantics: ``None`` through the warm-up; ``None`` if the window holds any
    null; ``NaN`` if it holds any ``NaN``; otherwise the reduced finite extreme.
    """
    result: list[float | None] = []
    for index in range(len(values)):
        if index < period - 1:
            result.append(None)
            continue
        window = values[index - period + 1 : index + 1]
        if any(value is None for value in window):
            result.append(None)
        elif any(isinstance(value, float) and math.isnan(value) for value in window):
            result.append(math.nan)
        else:
            result.append(reducer([value for value in window if value is not None]))
    return result


def reference_stochastic_fast(
    high: Sequence[float | None],
    low: Sequence[float | None],
    close: Sequence[float | None],
    window_k: int,
    window_d: int,
) -> dict[str, list[float | None]]:
    """
    Naive Fast Stochastic Oscillator over aligned Python lists.

    ``%K = 100 * (close - LL) / (HH - LL)`` with ``LL`` / ``HH`` the rolling lowest low / highest high over
    ``window_k``, and ``%D`` the :func:`reference_sma` of ``%K`` over ``window_d``, recomputed as the oracle for
    :func:`pomata.indicators.stochastic_fast`.

    Args:
        high: High-price series (may contain ``None`` and ``float('nan')``).
        low: Low-price series (may contain ``None`` and ``float('nan')``); same length as ``high``.
        close: Close-price series (may contain ``None`` and ``float('nan')``); same length as ``high``.
        window_k: Number of observations in the %K look-back range. Must be ``>= 1``.
        window_d: Number of observations in the %D moving average of %K. Must be ``>= 1``.

    Returns:
        A dict with keys ``"k"`` and ``"d"``, each a list the same length as the inputs.

    Raises:
        ValueError: If ``window_k < 1`` or ``window_d < 1``.
    """
    if window_k < 1:
        raise ValueError(f"window_k must be >= 1, got {window_k}")
    if window_d < 1:
        raise ValueError(f"window_d must be >= 1, got {window_d}")
    lowest_low = _rolling_extreme(low, window_k, min)
    highest_high = _rolling_extreme(high, window_k, max)
    percent_k: list[float | None] = []
    for low_value, high_value, close_value in zip(lowest_low, highest_high, close, strict=True):
        if low_value is None or high_value is None or close_value is None:
            percent_k.append(None)
        elif math.isnan(low_value) or math.isnan(high_value) or math.isnan(close_value):
            percent_k.append(math.nan)
        else:
            numerator = 100.0 * (close_value - low_value)
            denominator = high_value - low_value
            if denominator == 0.0:
                if numerator == 0.0:
                    percent_k.append(math.nan)
                else:
                    sign = math.copysign(1.0, numerator) * math.copysign(1.0, denominator)
                    percent_k.append(math.copysign(math.inf, sign))
            else:
                percent_k.append(numerator / denominator)
    return {"k": percent_k, "d": reference_sma(percent_k, window_d)}
