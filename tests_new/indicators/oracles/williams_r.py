"""
Naive reference oracle for ``pomata.indicators.williams_r``.
"""

import math
from collections.abc import Sequence


def reference_williams_r(
    high: Sequence[float | None],
    low: Sequence[float | None],
    close: Sequence[float | None],
    window: int,
) -> list[float | None]:
    """
    Naive Williams %R over three aligned Python lists.

    Williams %R, ``-100 * (HH - close) / (HH - LL)`` over the windowed highest high and lowest low, recomputed as the
    oracle for :func:`pomata.indicators.williams_r`. Its subtleties are the windowed missing-data contract and the
    IEEE-754 degeneracy when the range collapses (``HH == LL``), detailed below.

    Args:
        high: The bar-high observations (may contain ``None`` and ``float('nan')``).
        low: The bar-low observations (may contain ``None`` and ``float('nan')``); must be the same length as ``high``.
        close: The bar-close observations (may contain ``None`` and ``float('nan')``); must be the same length as
            ``high``.
        window: Window length. Must be ``>= 1``.

    Returns:
        A list of the same length as the inputs. The first ``window - 1`` entries are ``None`` (warm-up); thereafter
        ``-100 * (highest_high - close) / (highest_high - lowest_low)``.

    Raises:
        ValueError: If ``window < 1`` or if the three input lists do not all have the same length.

    Note:
        Edge-case behavior:

        - **Null** — a window in which any of ``high``, ``low``, or ``close`` contains a ``None`` yields ``None``
          (matching the ``rolling_max`` / ``rolling_min`` default ``min_samples=window``, under which a missing
          observation leaves the window short); ``None`` takes precedence over ``nan``.
        - **NaN** — a ``None``-free window that reaches a ``nan`` yields ``nan``. Because the oscillator reads only the
          current window, a ``None`` or ``nan`` contaminates only the positions whose window spans it and never latches
          onto the rest of the series.
        - **HH == LL** — when the windowed range collapses (``HH == LL``, e.g. a flat high-low over the whole
          window) the denominator is zero and the result follows IEEE-754: ``0 / 0`` (the close also equal to that
          level) is ``nan``, and a non-zero numerator over zero is ``+/-inf``.
        - **window == 1** — the highest high and lowest low collapse to the single bar's own ``high`` and ``low``, so
          the result is ``-100 * (high - close) / (high - low)``.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    length = len(high)
    if len(low) != length or len(close) != length:
        raise ValueError(f"high, low, and close must have equal length, got {length}, {len(low)}, {len(close)}")

    results: list[float | None] = []
    for index in range(length):
        if index + 1 < window:
            results.append(None)
            continue
        high_window = high[index + 1 - window : index + 1]
        low_window = low[index + 1 - window : index + 1]
        close_current = close[index]
        window_inputs = [*high_window, *low_window, close_current]
        if any(value is None for value in window_inputs):
            results.append(None)
            continue
        clean_highs = [value for value in high_window if value is not None]  # no nulls survive the guard above
        clean_lows = [value for value in low_window if value is not None]  # no nulls survive the guard above
        clean_close = close_current if close_current is not None else 0.0  # non-null here (guard above); narrow typing
        if any(math.isnan(value) for value in (*clean_highs, *clean_lows, clean_close)):
            results.append(math.nan)
            continue
        highest_high = max(clean_highs)
        lowest_low = min(clean_lows)
        numerator = -100.0 * (highest_high - clean_close)
        denominator = highest_high - lowest_low
        if denominator == 0.0:
            results.append(math.nan if numerator == 0.0 else math.copysign(math.inf, numerator))
        else:
            results.append(numerator / denominator)
    return results
