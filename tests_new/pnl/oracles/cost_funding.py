"""
Naive reference oracle for ``pomata.pnl.cost_funding``.
"""

from collections.abc import Sequence


def cost_funding_reference(
    quantity: Sequence[float | None],
    price: Sequence[float | None],
    rate: Sequence[float | None],
) -> list[float | None]:
    """
    Naive Funding Cost over three aligned Python lists.

    The elementwise product ``quantity * price * rate``, recomputed as the oracle for :func:`pomata.pnl.cost_funding`.
    It is a pure per-row product carrying the signed funding payment on the held notional; its one subtlety is the
    missing-data rule of plain arithmetic — ``None`` in any input propagates to ``None`` (taking precedence over
    ``nan``) and a ``nan`` propagates to ``nan`` — detailed below.

    Args:
        quantity: Signed position sizes for each bar (may contain ``None`` and ``float('nan')``).
        price: Instrument price for each bar (may contain ``None`` and ``float('nan')``); same length as ``quantity``.
        rate: Signed per-bar funding rate (may contain ``None`` and ``float('nan')``); same length as ``quantity``.

    Returns:
        A list the same length as the inputs: ``quantity * price * rate`` for each row.

    Raises:
        ValueError: If ``quantity``, ``price``, and ``rate`` do not all have the same length.

    Note:
        Edge-case behavior:

        - **Null** — a ``None`` in any input makes that row ``None`` (``None`` takes precedence over ``nan``).
        - **NaN** — a ``nan`` in any input (with no ``None``) propagates to ``nan``.
    """
    if not len(quantity) == len(price) == len(rate):
        raise ValueError("quantity, price, and rate must have equal length")

    results: list[float | None] = []
    for quantity_value, price_value, rate_value in zip(quantity, price, rate, strict=True):
        if quantity_value is None or price_value is None or rate_value is None:
            results.append(None)
        else:
            results.append(quantity_value * price_value * rate_value)
    return results
