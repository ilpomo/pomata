"""
Naive reference oracle for ``pomata.pnl.pnl_gross``.
"""

import math
from collections.abc import Sequence


def pnl_gross_reference(
    quantity: Sequence[float | None],
    price: Sequence[float | None],
    multiplier: float = 1.0,
) -> list[float | None]:
    """
    Naive Gross Position PnL over two aligned Python lists.

    The per-bar mark-to-market PnL ``quantity * (price - price_prev) * multiplier``, recomputed as the oracle for
    :func:`pomata.pnl.pnl_gross`. Its subtleties are the warm-up first row (no previous price) and the missing-data rule
    of plain arithmetic — ``None`` in the quantity, the price, or the previous price propagates to ``None`` (taking
    precedence over ``nan``), and a ``nan`` propagates to ``nan`` — detailed below.

    Args:
        quantity: Signed position sizes for each bar (may contain ``None`` and ``float('nan')``).
        price: Instrument prices (may contain ``None`` and ``float('nan')``); same length as ``quantity``.
        multiplier: Contract multiplier / point value. Must be ``> 0``.

    Returns:
        A list the same length as the inputs. The first entry is ``None`` (warm-up); thereafter
        ``quantity * (price - price_prev) * multiplier``.

    Raises:
        ValueError: If ``multiplier`` is not a finite number ``> 0`` (i.e. ``<= 0``, ``NaN``, or ``±inf``), or if
            ``quantity`` and ``price`` do not have the same length.

    Note:
        Edge-case behavior:

        - **Null** — a ``None`` in the quantity, the price, or the previous price yields ``None`` (``None`` takes
          precedence over ``nan``).
        - **NaN** — a ``nan`` in any of them (with no ``None``) propagates to ``nan``.
    """
    if not math.isfinite(multiplier) or multiplier <= 0.0:
        raise ValueError(f"multiplier must be a finite number > 0, got {multiplier}")
    if len(quantity) != len(price):
        raise ValueError("quantity and price must have equal length")

    results: list[float | None] = []
    for index in range(len(quantity)):
        if index == 0:
            results.append(None)
            continue
        current_quantity = quantity[index]
        current_price = price[index]
        previous_price = price[index - 1]
        if current_quantity is None or current_price is None or previous_price is None:
            results.append(None)
        elif math.isnan(current_quantity) or math.isnan(current_price) or math.isnan(previous_price):
            results.append(math.nan)
        else:
            results.append(current_quantity * (current_price - previous_price) * multiplier)
    return results
