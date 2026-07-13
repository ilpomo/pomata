"""
Naive reference oracle for ``pomata.indicators.kama``.
"""

import math
from collections.abc import Sequence


def _absolute_difference(left: float | None, right: float | None) -> float | None:
    """
    ``abs(left - right)``, ``None`` if either is null (``nan`` propagates).
    """
    if left is None or right is None:
        return None
    return abs(left - right)


def _smoothing_constant(
    close: Sequence[float | None],
    index: int,
    window: int,
    fast: float,
    slow: float,
) -> float | None:
    """
    The KAMA smoothing constant at ``index`` (``>= window``): the squared, bounded efficiency ratio.

    Efficiency ratio = ``|close[i] - close[i - window]|`` over the rolling sum of ``|close[j] - close[j - 1]|`` across
    the window, following Polars' ``rolling_sum`` null/NaN handling (``None`` if the window holds any null, ``NaN`` if
    it holds any NaN) and the ``when(volatility == 0)`` guard (a flat window gives efficiency ratio ``0``, not
    ``0 / 0``).
    """
    change = _absolute_difference(close[index], close[index - window])
    steps = [_absolute_difference(close[step], close[step - 1]) for step in range(index - window + 1, index + 1)]
    if any(step is None for step in steps):
        volatility: float | None = None
    elif any(math.isnan(step) for step in steps if step is not None):
        volatility = math.nan
    else:
        volatility = sum(step for step in steps if step is not None)
    if change is None or volatility is None:
        return None
    if math.isnan(change) or math.isnan(volatility):
        return math.nan
    efficiency_ratio = 0.0 if volatility == 0.0 else change / volatility
    return (efficiency_ratio * (fast - slow) + slow) ** 2


def kama_reference(
    values: Sequence[float | None],
    window: int,
    window_fast: int = 2,
    window_slow: int = 30,
) -> list[float | None]:
    """
    Naive Kaufman Adaptive Moving Average over a Python list.

    The efficiency ratio and smoothing constant are recomputed in a naive Python loop (the implementation builds them as
    Polars expressions), then the same seeded recurrence is run, as the oracle for :func:`pomata.indicators.kama`.

    Args:
        values: Input series (may contain ``None`` and ``float('nan')``).
        window: Number of observations in the efficiency-ratio look-back. Must be ``>= 1``.
        window_fast: Period of the fast smoothing-constant bound. Must be ``>= 1`` and ``<= window_slow``.
        window_slow: Period of the slow smoothing-constant bound. Must be ``>= 1``.

    Returns:
        A list the same length as ``values``: the KAMA, ``None`` through the ``window - 1`` warm-up rows.

    Raises:
        ValueError: If ``window < 1``, ``window_fast < 1``, ``window_slow < 1``, or ``window_fast > window_slow``.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    if window_fast < 1:
        raise ValueError(f"window_fast must be >= 1, got {window_fast}")
    if window_slow < 1:
        raise ValueError(f"window_slow must be >= 1, got {window_slow}")
    if window_fast > window_slow:
        raise ValueError(
            f"windows must be ordered window_fast <= window_slow, "
            f"got window_fast={window_fast}, window_slow={window_slow}"
        )
    fast = 2.0 / (window_fast + 1)
    slow = 2.0 / (window_slow + 1)
    constants: list[float | None] = [None] * len(values)
    for index in range(window, len(values)):
        constants[index] = _smoothing_constant(values, index, window, fast, slow)
    result: list[float | None] = [None] * len(values)
    kama_previous: float | None = None
    seeded = False
    for index in range(window - 1, len(values)):
        value = values[index]
        constant = constants[index]
        if not seeded:
            if value is None:
                continue
            result[index] = value
            kama_previous = value
            seeded = True
            continue
        if value is None or constant is None:
            continue
        assert kama_previous is not None
        kama_previous = kama_previous + constant * (value - kama_previous)
        result[index] = kama_previous
    return result
