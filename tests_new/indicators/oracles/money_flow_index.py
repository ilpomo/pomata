"""
Naive reference oracle for ``pomata.indicators.money_flow_index``.
"""

import math
from collections.abc import Sequence


def reference_money_flow_index(
    high: Sequence[float | None],
    low: Sequence[float | None],
    close: Sequence[float | None],
    volume: Sequence[float | None],
    window: int,
) -> list[float | None]:
    """
    Naive Money Flow Index over four aligned Python lists.

    The Money Flow Index, ``100 - 100 / (1 + positive_flow / negative_flow)`` over the typical-price-classified raw
    money flows, recomputed as the oracle for :func:`pomata.indicators.money_flow_index`. Its delicate points are the
    typical-price-change classification (where a ``nan`` change, undefined in sign, is poisoned into both flows), the
    ``window`` -of-*changes* warm-up, and the money-ratio saturations, detailed below.

    Args:
        high: The high-price observations (may contain ``None`` and ``float('nan')``).
        low: The low-price observations (may contain ``None`` and ``float('nan')``); same length as ``high``.
        close: The close-price observations (may contain ``None`` and ``float('nan')``); same length as ``high``.
        volume: The volume observations (may contain ``None`` and ``float('nan')``); same length as ``high``.
        window: Window length. Must be ``>= 1``.

    Returns:
        A list of the same length as the inputs. The first ``window`` entries are ``None`` (warm-up); thereafter the
        Money Flow Index in ``[0, 100]``.

    Raises:
        ValueError: If ``window < 1`` or if the four input lists do not all have the same length.

    Note:
        Edge-case behavior:

        - **Null** — a ``None`` in ``high``, ``low``, or ``close`` voids the typical price at that change and at the
          next, so any window reaching either yields ``None``; a ``None`` in ``volume`` voids only that row's money
          flow. ``None`` takes precedence over ``nan``.
        - **NaN** — a ``nan`` in any input contaminates the affected money flow and yields ``nan`` for every window that
          contains it. A ``nan`` typical price makes its own change and the next one undefined in sign, so both are
          poisoned into the positive and the negative flow as ``nan``, voiding every window that reaches either change.
        - **Division by zero** — a window with no negative money flow but non-zero positive flow has money ratio
          ``+inf`` and the MFI saturates at ``100``; symmetrically an all-down window gives ``0``. A window in which
          both flows are zero (the typical price never moves) leaves the money ratio at ``0 / 0`` and yields ``nan``.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    length = len(high)
    if not (len(low) == len(close) == len(volume) == length):
        raise ValueError("high, low, close, and volume must have equal length")

    typical_price: list[float | None] = []
    raw_money_flow: list[float | None] = []
    for high_value, low_value, close_value, volume_value in zip(high, low, close, volume, strict=True):
        typical = (
            None
            if high_value is None or low_value is None or close_value is None
            else (high_value + low_value + close_value) / 3.0
        )
        typical_price.append(typical)
        if typical is None or volume_value is None:
            raw_money_flow.append(None)
        else:
            raw_money_flow.append(typical * volume_value)

    positive_flow: list[float | None] = [None] * length
    negative_flow: list[float | None] = [None] * length
    for index in range(1, length):
        typical_current = typical_price[index]
        typical_previous = typical_price[index - 1]
        if typical_current is None or typical_previous is None:
            continue  # a None difference voids both flows at this position (warm-up / null bridge)
        change = typical_current - typical_previous
        if math.isnan(change):  # an undefined (nan) change is poisoned into both flows, voiding the successor windows
            positive_flow[index] = math.nan
            negative_flow[index] = math.nan
        elif change > 0.0:
            positive_flow[index] = raw_money_flow[index]
            negative_flow[index] = 0.0
        elif change < 0.0:
            positive_flow[index] = 0.0
            negative_flow[index] = raw_money_flow[index]
        else:  # unchanged typical price contributes to neither flow
            positive_flow[index] = 0.0
            negative_flow[index] = 0.0

    results: list[float | None] = []
    for index in range(length):
        if index + 1 < window:
            results.append(None)
            continue
        positive_window = positive_flow[index + 1 - window : index + 1]
        negative_window = negative_flow[index + 1 - window : index + 1]
        if any(flow is None for flow in positive_window) or any(flow is None for flow in negative_window):
            results.append(None)
            continue
        if any(isinstance(flow, float) and math.isnan(flow) for flow in positive_window) or any(
            isinstance(flow, float) and math.isnan(flow) for flow in negative_window
        ):
            results.append(math.nan)
            continue
        positive_sum = sum(flow for flow in positive_window if flow is not None)
        negative_sum = sum(flow for flow in negative_window if flow is not None)
        if negative_sum == 0.0:
            money_ratio = math.inf if positive_sum != 0.0 else math.nan
        else:
            money_ratio = positive_sum / negative_sum
        if math.isnan(money_ratio):
            results.append(math.nan)
        else:
            results.append(100.0 - 100.0 / (1.0 + money_ratio))
    return results
