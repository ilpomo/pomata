"""
Naive reference oracle for ``pomata.pnl.cost_borrow``.
"""

import math
from collections.abc import Sequence


def cost_borrow_reference(
    quantity: Sequence[float | None],
    price: Sequence[float | None],
    rate: float,
) -> list[float | None]:
    """
    Naive Short-Borrow Cost over two aligned Python lists.

    The per-bar short-borrow fee ``max(-quantity, 0) * price * rate``, recomputed as the oracle for
    :func:`pomata.pnl.cost_borrow`. Only the short part of the position is charged; its one subtlety is the missing-data
    rule of plain arithmetic — ``None`` in either input propagates to ``None`` (taking precedence over ``nan``) and a
    ``nan`` propagates to ``nan`` — detailed below.

    Args:
        quantity: Signed position sizes for each bar (may contain ``None`` and ``float('nan')``); only the short part
            (``< 0``) is charged.
        price: Instrument prices (may contain ``None`` and ``float('nan')``); same length as ``quantity``.
        rate: Per-bar borrow rate, as a fraction of the short notional. Must be ``>= 0``.

    Returns:
        A list the same length as the inputs: a non-negative cost on short bars and ``0`` on long or flat bars.

    Raises:
        ValueError: If ``rate`` is not a finite number ``>= 0`` (i.e. ``< 0``, ``NaN``, or ``±inf``), or if
            ``quantity`` and ``price`` do not have the same length.

    Note:
        Edge-case behavior:

        - **Long / flat** — a non-negative quantity has zero borrow cost.
        - **Null** — a ``None`` in either input makes that row ``None`` (``None`` takes precedence over ``nan``).
        - **NaN** — a ``nan`` in either input (with no ``None``) propagates to ``nan``.
    """
    if not math.isfinite(rate) or rate < 0.0:
        raise ValueError(f"rate must be a finite number >= 0, got {rate}")
    if len(quantity) != len(price):
        raise ValueError("quantity and price must have equal length")

    results: list[float | None] = []
    for quantity_value, price_value in zip(quantity, price, strict=True):
        if quantity_value is None or price_value is None:
            results.append(None)
        elif math.isnan(quantity_value) or (math.isnan(price_value)):
            results.append(math.nan)
        else:
            results.append(max(-quantity_value, 0.0) * price_value * rate)
    return results
