"""
Naive reference oracle for ``pomata.indicators.supertrend``.
"""

import math
from collections.abc import Sequence

from tests_new.indicators.oracles.atr import atr_reference


def _as_float(value: float | None) -> float:
    """
    Narrow a known-finite oracle value to ``float`` (the finite bars are filtered before this is called).
    """
    assert value is not None
    return value


def _basic_bands(
    high: Sequence[float | None],
    low: Sequence[float | None],
    close: Sequence[float | None],
    window: int,
    multiplier: float,
) -> tuple[list[float | None], list[float | None]]:
    """
    The basic upper / lower bands ``(high + low) / 2 +/- multiplier * ATR``, built on the independent ``atr_reference``.

    A band is ``None`` where the ATR or either price is ``None``, ``nan`` where one is ``nan`` (and none ``None``), else
    the finite band -- matching how ``null`` / ``NaN`` flow through the implementation's native band arithmetic.
    """
    atr_values = atr_reference(high, low, close, window)
    upper: list[float | None] = []
    lower: list[float | None] = []
    for index in range(len(close)):
        atr_value = atr_values[index]
        high_value = high[index]
        low_value = low[index]
        if atr_value is None or high_value is None or low_value is None:
            upper.append(None)
            lower.append(None)
            continue
        if math.isnan(atr_value) or math.isnan(high_value) or math.isnan(low_value):
            upper.append(math.nan)
            lower.append(math.nan)
            continue
        mid = (high_value + low_value) / 2.0
        half = multiplier * atr_value
        upper.append(mid + half)
        lower.append(mid - half)
    return upper, lower


def supertrend_reference(
    high: Sequence[float | None],
    low: Sequence[float | None],
    close: Sequence[float | None],
    window: int,
    multiplier: float,
) -> dict[str, list[float | None]]:
    """
    Naive SuperTrend over Python lists, written as two explicit passes rather than the implementation's single-pass
    state machine, so agreement is evidence rather than a shared shape.

    The first pass ratchets the *final* band arrays over the finite bars (the upper falls only while the prior close
    holds below it, the lower rises only while the prior close holds above it); the second pass walks those arrays,
    flipping the line and ``direction`` on a strict close-cross. The trend seeds short when the first finite close is at
    or below the lower band, else long.

    Args:
        high: High-price series (may contain ``None`` and ``float('nan')``).
        low: Low-price series (may contain ``None`` and ``float('nan')``).
        close: Close-price series (may contain ``None`` and ``float('nan')``).
        window: Number of observations in the ATR moving window. Must be ``>= 1``.
        multiplier: Band half-width as a multiple of the ATR. Must be ``> 0``.

    Returns:
        A dict with two lists the same length as the inputs — ``"line"`` and ``"direction"`` — matching the fields of
        the indicator's struct. The first ``window - 1`` entries are ``None`` (the ATR warm-up).

    Raises:
        ValueError: If ``window < 1`` or ``multiplier <= 0``.

    Note:
        A bar whose band or close is ``None`` (warm-up, or touching a ``None``) is skipped and stays ``None``; a
        ``nan`` band or close yields ``nan`` on both fields. Either way the running state, and the last finite close the
        ratchet reads, bridge the gap rather than latching.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    if not math.isfinite(multiplier) or multiplier <= 0.0:
        raise ValueError(f"multiplier must be a finite number > 0, got {multiplier}")

    upper, lower = _basic_bands(high, low, close, window, multiplier)
    line: list[float | None] = [None] * len(close)
    direction: list[float | None] = [None] * len(close)

    finite: list[int] = []
    for index in range(len(close)):
        upper_value = upper[index]
        lower_value = lower[index]
        close_value = close[index]
        if upper_value is None or lower_value is None or close_value is None:
            continue
        if math.isnan(upper_value) or math.isnan(lower_value) or math.isnan(close_value):
            line[index] = math.nan
            direction[index] = math.nan
            continue
        finite.append(index)
    if not finite:
        return {"line": line, "direction": direction}

    # Pass 1 -- the ratchet, building the final-band arrays over the finite bars.
    final_upper: list[float] = [_as_float(upper[finite[0]])]
    final_lower: list[float] = [_as_float(lower[finite[0]])]
    for position in range(1, len(finite)):
        index = finite[position]
        previous_close = _as_float(close[finite[position - 1]])
        upper_value = _as_float(upper[index])
        lower_value = _as_float(lower[index])
        carried_upper = final_upper[position - 1]
        carried_lower = final_lower[position - 1]
        reset_upper = upper_value < carried_upper or previous_close > carried_upper
        reset_lower = lower_value > carried_lower or previous_close < carried_lower
        final_upper.append(upper_value if reset_upper else carried_upper)
        final_lower.append(lower_value if reset_lower else carried_lower)

    # Pass 2 -- the flip pass, reading the final-band arrays.
    seed = finite[0]
    trend = -1.0 if _as_float(close[seed]) <= final_lower[0] else 1.0
    line[seed] = final_upper[0] if trend < 0.0 else final_lower[0]
    direction[seed] = trend
    for position in range(1, len(finite)):
        index = finite[position]
        close_value = _as_float(close[index])
        band_upper = final_upper[position]
        band_lower = final_lower[position]
        if trend > 0.0:
            if close_value < band_lower:
                trend = -1.0
                line[index] = band_upper
            else:
                line[index] = band_lower
        elif close_value > band_upper:
            trend = 1.0
            line[index] = band_lower
        else:
            line[index] = band_upper
        direction[index] = trend
    return {"line": line, "direction": direction}
