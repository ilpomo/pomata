"""
Naive reference oracle for ``pomata.pnl.pnl_gross_inverse``.
"""

import math
from collections.abc import Sequence


def pnl_gross_inverse_reference(
    quantity: Sequence[float | None],
    price: Sequence[float | None],
    multiplier: float = 1.0,
) -> list[float | None]:
    """
    Naive Gross Inverse-Contract PnL over two aligned Python lists.

    The per-bar coin-settled mark-to-market PnL ``quantity * multiplier * (1 / price_prev - 1 / price)``, recomputed as
    the oracle for :func:`pomata.pnl.pnl_gross_inverse`. Its subtleties are the warm-up first row (no previous price),
    the IEEE-754 division reproduced from Polars on the price reciprocal (a zero price is ``+/-inf``), and the
    missing-data rule of plain arithmetic — ``None`` in the quantity, the price, or the previous price propagates to
    ``None`` (taking precedence over ``nan``), and a ``nan`` propagates to ``nan`` — detailed below.

    Args:
        quantity: Signed position sizes in contracts for each bar (may contain ``None`` and ``float('nan')``).
        price: Instrument prices, the quote per base unit (may contain ``None`` and ``float('nan')``); same length as
            ``quantity``.
        multiplier: Contract notional in the quote currency. Must be ``> 0``.

    Returns:
        A list the same length as the inputs. The first entry is ``None`` (warm-up); thereafter
        ``quantity * multiplier * (1 / price_prev - 1 / price)``.

    Raises:
        ValueError: If ``multiplier`` is not a finite number ``> 0`` (i.e. ``<= 0``, ``NaN``, or ``±inf``), or if
            ``quantity`` and ``price`` do not have the same length.

    Note:
        Edge-case behavior:

        - **Null** — a ``None`` in the quantity, the price, or the previous price yields ``None`` (``None`` takes
          precedence over ``nan``).
        - **NaN** — a ``nan`` in any of them (with no ``None``) propagates to ``nan``.
        - **Domain** — reproducing Polars' IEEE-754 division of the reciprocal: a zero price makes ``1 / price`` carry
          the sign of the zero as ``+/-inf`` (so the bar is signed infinity), and a negative price gives a finite but
          economically meaningless value.
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
            reciprocal_previous = (
                1.0 / previous_price if previous_price != 0.0 else math.copysign(math.inf, previous_price)
            )
            reciprocal_current = 1.0 / current_price if current_price != 0.0 else math.copysign(math.inf, current_price)
            results.append(current_quantity * multiplier * (reciprocal_previous - reciprocal_current))
    return results
