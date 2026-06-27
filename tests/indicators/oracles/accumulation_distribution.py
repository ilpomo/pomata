"""
Naive reference oracle for ``pomata.indicators.accumulation_distribution``.
"""

import math
from collections.abc import Sequence


def accumulation_distribution_reference(
    high: Sequence[float | None],
    low: Sequence[float | None],
    close: Sequence[float | None],
    volume: Sequence[float | None],
) -> list[float | None]:
    """
    Naive Accumulation/Distribution Line over four aligned Python lists.

    The Accumulation/Distribution line, the running cumulative sum of the money-flow volume ``MFM * volume`` with the
    multiplier pinned to ``0`` on a genuine doji (``high - low == 0``, both finite), recomputed as the oracle for
    :func:`pomata.indicators.accumulation_distribution`. Its non-obvious point is that the doji guard requires a finite
    zero range, so a both-``nan`` bar latches the ``nan`` -propagating cumulative sum rather than contributing ``0``,
    detailed below.

    Args:
        high: Per-bar high prices (may contain ``None`` and ``float('nan')``).
        low: Per-bar low prices; must be the same length as ``high``.
        close: Per-bar close prices; must be the same length as ``high``.
        volume: Per-bar traded volume; must be the same length as ``high``.

    Returns:
        A list of the same length as the inputs. There is no moving window and no warm-up: every bar carries its own
        money-flow volume, so the first bar is already defined.

    Raises:
        ValueError: If the four input sequences do not all have the same length.

    Note:
        Edge-case behavior:

        - **Null** — a bar whose ``high``, ``low``, or ``volume`` is ``None``, or whose ``close`` is ``None`` on a
          non-doji bar, yields ``None`` at that position and leaves the running total untouched (the cumulative sum
          skips it and continues from the prior total). On a genuine doji bar (``high - low == 0``, both finite) the
          multiplier is ``0`` and ``close`` is irrelevant, so a ``None`` ``close`` there still yields ``0`` rather than
          ``None``.
        - **NaN** — a ``nan`` in any operand that reaches the cumulative sum latches: once summed, every later
          non-``null`` bar of the line is ``nan`` (a ``None``-contribution bar still emits ``None`` at its own
          position). A bar whose ``high`` and ``low`` are both ``nan`` does not take the doji branch (``nan - nan`` is
          ``nan``, never ``== 0``), so the ``nan`` poisons the line rather than contributing ``0``.
        - **high == low** — on a genuine doji bar (both finite) the range ``high - low`` is zero, so the Money Flow
          Multiplier is pinned to ``0`` (the standard convention) and the bar contributes nothing.
    """
    length = len(high)
    if not (len(low) == length and len(close) == length and len(volume) == length):
        raise ValueError(
            "high, low, close and volume must have equal length, got "
            f"{len(high)}, {len(low)}, {len(close)} and {len(volume)}"
        )

    results: list[float | None] = []
    running_total = 0.0  # the cumulative AD level; carried unchanged across null bars
    nan_latched = False  # once a nan reaches the cumulative sum, every later non-null row is nan
    for high_value, low_value, close_value, volume_value in zip(high, low, close, volume, strict=True):
        if high_value is None or low_value is None or volume_value is None:
            money_flow_volume: float | None = None
        else:
            high_low_range = high_value - low_value
            is_doji = high_low_range == 0  # a finite equal-range bar; nan - nan is nan (never == 0), so it is not doji
            if is_doji:
                money_flow_volume = 0.0 * volume_value  # 0.0 for a finite volume, nan only when volume is nan
            elif close_value is None:
                money_flow_volume = None
            else:
                money_flow_volume = (
                    ((close_value - low_value) - (high_value - close_value)) / high_low_range
                ) * volume_value
        if money_flow_volume is None:
            results.append(None)
            continue
        if nan_latched:
            results.append(math.nan)
            continue
        if math.isnan(money_flow_volume):
            nan_latched = True
            running_total = math.nan
            results.append(math.nan)
            continue
        running_total = running_total + money_flow_volume
        results.append(running_total)
    return results
