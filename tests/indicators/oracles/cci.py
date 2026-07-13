"""
Naive reference oracle for ``pomata.indicators.cci``.
"""

import math
from collections.abc import Sequence

from tests.indicators.oracles.sma import sma_reference


def cci_reference(
    high: Sequence[float | None],
    low: Sequence[float | None],
    close: Sequence[float | None],
    window: int,
) -> list[float | None]:
    """
    Naive Commodity Channel Index over three Python lists.

    The Commodity Channel Index ``(TP - SMA(TP)) / (0.015 * MAD)`` on the typical price
    ``TP = (high + low + close) / 3``, recomputed as the oracle for :func:`pomata.indicators.cci`. Its one non-obvious
    point is that the mean absolute deviation is taken about the *current* rolling mean ``SMA(TP)_t`` (the same value
    for every term in the window), not about each point's own mean; warm-up and ``null`` / ``NaN`` are inherited
    from :func:`sma_reference`, detailed below.

    Args:
        high: The high-price observations (may contain ``None`` and ``float('nan')``).
        low: The low-price observations (may contain ``None`` and ``float('nan')``); must be the same length as
            ``high``.
        close: The close-price observations (may contain ``None`` and ``float('nan')``); must be the same length as
            ``high``.
        window: Window length. Must be ``>= 1``.

    Returns:
        A list of the same length as the inputs. The first ``window - 1`` entries are ``None`` (warm-up); thereafter the
        Commodity Channel Index value.

    Raises:
        ValueError: If ``window < 1`` or if the three input lists do not all have the same length.

    Note:
        Edge-case behavior:

        - **Null** — a window in which ``high``, ``low``, or ``close`` contains a ``None`` yields ``None`` (the typical
          price is ``None`` there, and so is any rolling quantity that covers it); ``None`` takes precedence over
          ``nan``.
        - **NaN** — a window containing a ``nan`` (and no ``None``) yields ``nan``.
        - **Flat window** — when every typical price in the window is equal there is no spread to normalize by (the
          ``0 / 0`` degenerate); the window is detected exactly (its maximum equals its minimum) and the result is
          ``nan``.
        - **window == 1** — every one-bar window is trivially flat, so every result is ``nan``.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    if not len(high) == len(low) == len(close):
        raise ValueError(f"high, low, and close must have equal length, got {len(high)}, {len(low)}, and {len(close)}")

    typical_values: list[float | None] = []
    for value_high, value_low, value_close in zip(high, low, close, strict=True):
        if value_high is None or value_low is None or value_close is None:
            typical_values.append(None)
        elif any(math.isnan(value) for value in (value_high, value_low, value_close)):
            typical_values.append(math.nan)
        else:
            typical_values.append((value_high + value_low + value_close) / 3.0)

    typical_mean = sma_reference(typical_values, window)

    results: list[float | None] = []
    for index in range(len(typical_values)):
        mean_value = typical_mean[index]
        if mean_value is None:
            results.append(None)
            continue
        deviation_terms: list[float] = []
        window_values: list[float] = []
        reached_null = False
        reached_nan = math.isnan(mean_value)
        for offset in range(window):
            shifted_index = index - offset
            value = typical_values[shifted_index] if shifted_index >= 0 else None
            if value is None:
                reached_null = True
                break
            if math.isnan(value):
                reached_nan = True
            window_values.append(value)
            deviation_terms.append(abs(value - mean_value))
        if reached_null:
            results.append(None)
            continue
        if reached_nan:
            results.append(math.nan)
            continue
        # A flat window (every typical price equal) is the 0/0 degenerate; detect it exactly so a sub-ULP residual in
        # the mean cannot fake a finite reading, matching the implementation and the documented NaN.
        if max(window_values) == min(window_values):
            results.append(math.nan)
            continue
        mean_deviation = sum(deviation_terms) / window
        typical_value = typical_values[index]
        assert typical_value is not None  # finite here: the None / NaN / flat-window cases returned above
        numerator = typical_value - mean_value
        denominator = 0.015 * mean_deviation
        if denominator == 0.0:
            results.append(math.nan if numerator == 0.0 else math.copysign(math.inf, numerator))
        else:
            results.append(numerator / denominator)
    return results
