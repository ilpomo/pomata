"""
Naive reference oracle for ``pomata.pnl.dividend``.
"""

from collections.abc import Sequence


def dividend_reference(
    quantity: Sequence[float | None],
    dividend_per_share: Sequence[float | None],
) -> list[float | None]:
    """
    Naive Dividend Cashflow over two aligned Python lists.

    The elementwise product ``quantity * dividend_per_share``, recomputed as the oracle for
    :func:`pomata.pnl.dividend`. It is a pure per-row product; its one subtlety is the missing-data rule of plain
    arithmetic — ``None`` in either input propagates to ``None`` (taking precedence over ``nan``) and a ``nan``
    propagates to ``nan`` — detailed below.

    Args:
        quantity: Signed position sizes for each bar (may contain ``None`` and ``float('nan')``).
        dividend_per_share: Dividend paid per share for each bar (may contain ``None`` and ``float('nan')``); same
            length as ``quantity``.

    Returns:
        A list the same length as the inputs: ``quantity * dividend_per_share`` for each row.

    Raises:
        ValueError: If ``quantity`` and ``dividend_per_share`` do not have the same length.

    Note:
        Edge-case behavior:

        - **Null** — a ``None`` in either input makes that row ``None`` (``None`` takes precedence over ``nan``).
        - **NaN** — a ``nan`` in either input (with no ``None``) propagates to ``nan``.
    """
    if len(quantity) != len(dividend_per_share):
        raise ValueError("quantity and dividend_per_share must have equal length")

    results: list[float | None] = []
    for quantity_value, dps_value in zip(quantity, dividend_per_share, strict=True):
        if quantity_value is None or dps_value is None:
            results.append(None)
        else:
            results.append(quantity_value * dps_value)
    return results
