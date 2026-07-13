"""
Naive reference oracle for ``pomata.indicators.obv``.
"""

import math
from collections.abc import Sequence


def obv_reference(
    close: Sequence[float | None],
    volume: Sequence[float | None],
) -> list[float | None]:
    """
    Naive On-Balance Volume over two aligned Python lists.

    On-Balance Volume, the running ``cum_sum(sign(close.diff()) * volume)``, recomputed as the oracle for
    :func:`pomata.indicators.obv`. Its delicate points are the first row's undefined ``diff`` filled to ``0``, the
    IEEE-754 products (``0 * nan`` is ``nan``), and the ``nan`` -latching cumulative sum, detailed below.

    Args:
        close: The close observations (may contain ``None`` and ``float('nan')``).
        volume: The traded-volume observations (may contain ``None`` and ``float('nan')``); must be the same length as
            ``close``.

    Returns:
        A list of the same length as the inputs. There is no warm-up: every row is defined, starting at ``0.0`` on the
        first bar; thereafter the running cumulative signed-volume total.

    Raises:
        ValueError: If the two input sequences do not have the same length.

    Note:
        Edge-case behavior:

        - **Null** — a ``None`` close zeroes the direction both at its own row and at the following row (each ``diff``
          touching the ``None`` is itself ``None`` and is filled to ``0``), so those bars contribute nothing while the
          running total carries on. A ``None`` volume makes that bar's contribution ``None`` (``0 * None`` is ``None``,
          so this holds even on the first or a flat bar): the output is ``None`` at exactly that row while the
          cumulative sum skips it and continues from the prior total.
        - **NaN** — a ``nan`` close (via ``diff``) or a ``nan`` volume poisons the contribution at its row and, once
          summed, latches the running total to ``nan`` for every subsequent row; because ``0 * nan`` is ``nan``, a
          ``nan`` volume contaminates the total even on a flat or first bar. A ``None``-contribution row still emits
          ``None`` at its own position even after the latch.
    """
    if len(close) != len(volume):
        raise ValueError(f"close and volume must have equal length, got {len(close)} and {len(volume)}")

    contributions: list[float | None] = []
    for index in range(len(close)):
        if index == 0:
            direction: float = 0.0  # the first bar has no predecessor, so its direction is 0
        else:
            previous_close = close[index - 1]
            current_close = close[index]
            if previous_close is None or current_close is None:
                direction = 0.0  # a None diff is filled to 0, mirroring fill_null(0)
            elif math.isnan(previous_close) or (math.isnan(current_close)):
                direction = math.nan
            else:
                change = current_close - previous_close
                direction = 1.0 if change > 0.0 else (-1.0 if change < 0.0 else 0.0)
        volume_value = volume[index]
        if volume_value is None:
            contributions.append(None)  # 0 * None and finite * None are both None
        elif math.isnan(direction) or (math.isnan(volume_value)):
            contributions.append(math.nan)  # a nan on either factor poisons the contribution
        else:
            contributions.append(direction * volume_value)

    results: list[float | None] = []
    running_total = 0.0
    nan_latched = False
    for contribution in contributions:
        if contribution is None:
            results.append(None)
            continue
        if math.isnan(contribution):
            nan_latched = True
        else:
            running_total += contribution
        results.append(math.nan if nan_latched else running_total)
    return results
