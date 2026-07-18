"""
Naive reference oracle for ``pomata.indicators.chaikin_money_flow``.
"""

import math
from collections.abc import Sequence


def reference_chaikin_money_flow(
    high: Sequence[float | None],
    low: Sequence[float | None],
    close: Sequence[float | None],
    volume: Sequence[float | None],
    window: int,
) -> list[float | None]:
    """
    Naive Chaikin Money Flow over four Python lists.

    Chaikin Money Flow, ``rolling_sum(MFV, window) / rolling_sum(volume, window)`` on the money-flow volume, recomputed
    as the oracle for :func:`pomata.indicators.chaikin_money_flow`. Its subtleties are the windowed missing-data
    contract (``None`` taking precedence over ``nan``) and the zero-volume degeneracy, detailed below.

    Args:
        high: The per-bar high observations (may contain ``None`` and ``float('nan')``).
        low: The per-bar low observations (may contain ``None`` and ``float('nan')``); same length as ``high``.
        close: The per-bar close observations (may contain ``None`` and ``float('nan')``); same length as ``high``.
        volume: The per-bar volume observations (may contain ``None`` and ``float('nan')``); same length as ``high``.
        window: Window length. Must be ``>= 1``.

    Returns:
        A list of the same length as the inputs. The first ``window - 1`` entries are ``None`` (warm-up); thereafter the
        volume-weighted money-flow ratio ``sum(MFV) / sum(volume)`` over the window.

    Raises:
        ValueError: If ``window < 1`` or if the four input lists do not all have the same length.

    Note:
        Edge-case behavior:

        - **Null** — a window in which any of ``high`` / ``low`` / ``close`` / ``volume`` contains a ``None`` yields
          ``None``; ``None`` takes precedence over ``nan``.
        - **NaN** — a window containing a ``nan`` (and no ``None``) yields ``nan``.
        - **Zero volume** — a window whose total volume is zero yields ``nan`` (IEEE-754 ``0 / 0``); with non-negative
          volume this is the only reachable case, since a zero volume sum also zeroes the numerator (a non-zero
          numerator over zero would be ``+/-inf``, but that is unreachable here).
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    lengths = {len(high), len(low), len(close), len(volume)}
    if len(lengths) != 1:
        raise ValueError(f"high, low, close, and volume must have equal length, got lengths {sorted(lengths)}")

    money_flow_volume: list[float | None] = []
    for high_value, low_value, close_value, volume_value in zip(high, low, close, volume, strict=True):
        if high_value is None or low_value is None or volume_value is None:
            money_flow_volume.append(None)
            continue
        high_is_nan = math.isnan(high_value)
        low_is_nan = math.isnan(low_value)
        if not (high_is_nan or low_is_nan) and high_value - low_value == 0.0:
            money_flow_volume.append(0.0 * volume_value)  # finite doji: MFM pinned 0, close unread
            continue
        if close_value is None:
            money_flow_volume.append(None)
            continue
        close_is_nan = math.isnan(close_value)
        volume_is_nan = math.isnan(volume_value)
        if high_is_nan or low_is_nan or close_is_nan or volume_is_nan:
            money_flow_volume.append(math.nan)
            continue
        high_low_range = high_value - low_value
        multiplier = ((close_value - low_value) - (high_value - close_value)) / high_low_range
        money_flow_volume.append(multiplier * volume_value)

    results: list[float | None] = []
    for index in range(len(high)):
        if index + 1 < window:
            results.append(None)
            continue
        money_flow_window = money_flow_volume[index + 1 - window : index + 1]
        volume_window = volume[index + 1 - window : index + 1]
        if any(value is None for value in money_flow_window) or any(value is None for value in volume_window):
            results.append(None)
            continue
        if any(isinstance(value, float) and math.isnan(value) for value in money_flow_window) or any(
            isinstance(value, float) and math.isnan(value) for value in volume_window
        ):
            results.append(math.nan)
            continue
        numerator = sum(value for value in money_flow_window if value is not None)
        denominator = sum(value for value in volume_window if value is not None)
        if denominator == 0.0:
            results.append(math.nan if numerator == 0.0 else math.copysign(math.inf, numerator))
        else:
            results.append(numerator / denominator)
    return results
