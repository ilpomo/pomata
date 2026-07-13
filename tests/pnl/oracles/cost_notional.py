"""
Naive reference oracle for ``pomata.pnl.cost_notional``.
"""

import math
from collections.abc import Sequence

from tests.pnl.oracles.turnover import turnover_reference


def cost_notional_reference(
    quantity: Sequence[float | None],
    price: Sequence[float | None],
    rate: float,
) -> list[float | None]:
    """
    Naive Notional Transaction Cost over two aligned Python lists.

    The traded notional times a flat rate, ``turnover(quantity) * price * rate``, recomputed as the oracle for
    :func:`pomata.pnl.cost_notional` by composing the independent :func:`turnover_reference` with the price and rate.
    Its subtleties are inherited from turnover (flat start, null voids its own row and the next, NaN propagates),
    plus the missing-data rule of the multiply by price — detailed below.

    Args:
        quantity: Signed position sizes for each bar (may contain ``None`` and ``float('nan')``).
        price: Instrument prices (may contain ``None`` and ``float('nan')``); same length as ``quantity``.
        rate: Proportional cost rate, the fee as a fraction of traded notional. Must be ``>= 0``.

    Returns:
        A list the same length as the inputs: the per-bar notional cost, with the first row charging on
        ``|quantity[0]| * price[0]``.

    Raises:
        ValueError: If ``rate`` is not a finite number ``>= 0`` (i.e. ``< 0``, ``NaN``, or ``±inf``), or if
            ``quantity`` and ``price`` do not have the same length.

    Note:
        Edge-case behavior:

        - **Flat start** — the pre-series quantity is ``0``, so the first row charges on the entry trade.
        - **Null** — a ``None`` in the traded quantity (or its predecessor, via turnover) or the price yields ``None``
          (``None`` takes precedence over ``nan``).
        - **NaN** — a ``nan`` in either input (with no ``None``) propagates to ``nan``.
    """
    if not math.isfinite(rate) or rate < 0.0:
        raise ValueError(f"rate must be a finite number >= 0, got {rate}")
    if len(quantity) != len(price):
        raise ValueError("quantity and price must have equal length")

    results: list[float | None] = []
    for traded, price_value in zip(turnover_reference(quantity), price, strict=True):
        if traded is None or price_value is None:
            results.append(None)
        elif math.isnan(traded) or math.isnan(price_value):
            results.append(math.nan)
        else:
            results.append(traded * price_value * rate)
    return results
