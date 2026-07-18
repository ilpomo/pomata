"""
Naive reference oracle for ``pomata.indicators.macd``.
"""

import math
from collections.abc import Sequence

from tests_new.indicators.oracles.ema import reference_ema


def difference(
    left: Sequence[float | None],
    right: Sequence[float | None],
) -> list[float | None]:
    """
    Elementwise ``left - right`` with ``None`` taking precedence over ``nan``.
    """
    result: list[float | None] = []
    for left_value, right_value in zip(left, right, strict=True):
        if left_value is None or right_value is None:
            result.append(None)
        elif math.isnan(left_value) or math.isnan(right_value):
            result.append(math.nan)
        else:
            result.append(left_value - right_value)
    return result


def reference_macd(
    close: Sequence[float | None],
    window_fast: int = 12,
    window_slow: int = 26,
    window_signal: int = 9,
) -> dict[str, list[float | None]]:
    """
    Naive MACD over a Python list.

    The fast-minus-slow :func:`reference_ema` gap (the MACD line), its further EMA (the signal line), and their
    difference (the histogram), recomputed from scratch as the oracle for :func:`pomata.indicators.macd`.

    Args:
        close: Close-price series (may contain ``None`` and ``float('nan')``).
        window_fast: Span of the fast EMA. Must be ``>= 1``.
        window_slow: Span of the slow EMA. Must be ``>= 1``.
        window_signal: Span of the signal EMA over the MACD line. Must be ``>= 1``.

    Returns:
        A dict with three lists the same length as ``close`` — ``"macd"``, ``"signal"``, and ``"histogram"`` — matching
        the fields of the indicator's struct.

    Raises:
        ValueError: If any period is ``< 1``.
    """
    if window_fast < 1 or window_slow < 1 or window_signal < 1:
        raise ValueError(
            f"window_fast, window_slow, and window_signal must be >= 1, "
            f"got {window_fast}, {window_slow}, and {window_signal}"
        )
    ema_fast = reference_ema(close, window_fast)
    ema_slow = reference_ema(close, window_slow)
    macd_line = difference(ema_fast, ema_slow)
    signal = reference_ema(macd_line, window_signal)
    histogram = difference(macd_line, signal)
    return {"macd": macd_line, "signal": signal, "histogram": histogram}
