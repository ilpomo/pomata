"""
Naive reference oracle for ``pomata.pnl.returns_net``.
"""

from collections.abc import Sequence


def returns_net_reference(
    returns_gross: Sequence[float | None],
    cost: Sequence[float | None],
) -> list[float | None]:
    """
    Naive Net Strategy Returns over two aligned Python lists.

    The elementwise difference ``returns_gross - cost``, recomputed as the oracle for :func:`pomata.pnl.returns_net`. It
    is a pure per-row subtraction; its one subtlety is the missing-data rule of plain arithmetic — ``None`` in either
    input propagates to ``None`` (taking precedence over ``nan``) and a ``nan`` propagates to ``nan`` — detailed below.

    Args:
        returns_gross: Gross per-bar strategy returns (may contain ``None`` and ``float('nan')``).
        cost: Per-bar transaction cost (may contain ``None`` and ``float('nan')``); same length as ``returns_gross``.

    Returns:
        A list the same length as the inputs: ``returns_gross - cost`` for each row.

    Raises:
        ValueError: If ``returns_gross`` and ``cost`` do not have the same length.

    Note:
        Edge-case behavior:

        - **Null** — a ``None`` in either input makes that row ``None`` (``None`` takes precedence over ``nan``).
        - **NaN** — a ``nan`` in either input (with no ``None`` at that row) propagates to ``nan``.
    """
    if len(returns_gross) != len(cost):
        raise ValueError("returns_gross and cost must have equal length")

    results: list[float | None] = []
    for gross, drag in zip(returns_gross, cost, strict=True):
        if gross is None or drag is None:
            results.append(None)
        else:
            results.append(gross - drag)
    return results
