"""
Naive reference oracle for ``pomata.pnl.pnl_net``.
"""

from collections.abc import Sequence


def pnl_net_reference(
    pnl_gross: Sequence[float | None],
    cost: Sequence[float | None],
) -> list[float | None]:
    """
    Naive Net Position PnL over two aligned Python lists.

    The elementwise difference ``pnl_gross - cost``, recomputed as the oracle for :func:`pomata.pnl.pnl_net`. It is a
    pure per-row subtraction; its one subtlety is the missing-data rule of plain arithmetic — ``None`` in either input
    propagates to ``None`` (taking precedence over ``nan``) and a ``nan`` propagates to ``nan`` — detailed below.

    Args:
        pnl_gross: Gross per-bar position PnL (may contain ``None`` and ``float('nan')``).
        cost: Per-bar transaction cost (may contain ``None`` and ``float('nan')``); same length as ``pnl_gross``.

    Returns:
        A list the same length as the inputs: ``pnl_gross - cost`` for each row.

    Raises:
        ValueError: If ``pnl_gross`` and ``cost`` do not have the same length.

    Note:
        Edge-case behavior:

        - **Null** — a ``None`` in either input makes that row ``None`` (``None`` takes precedence over ``nan``).
        - **NaN** — a ``nan`` in either input (with no ``None``) propagates to ``nan``.
    """
    if len(pnl_gross) != len(cost):
        raise ValueError("pnl_gross and cost must have equal length")

    results: list[float | None] = []
    for gross_value, cost_value in zip(pnl_gross, cost, strict=True):
        if gross_value is None or cost_value is None:
            results.append(None)
        else:
            results.append(gross_value - cost_value)
    return results
