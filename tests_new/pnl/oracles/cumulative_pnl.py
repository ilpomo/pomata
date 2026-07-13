"""
Naive reference oracle for ``pomata.pnl.cumulative_pnl``.
"""

from collections.abc import Sequence


def cumulative_pnl_reference(
    returns: Sequence[float | None],
) -> list[float | None]:
    """
    Naive Cumulative P&L over a Python list.

    The additive running total ``sum(returns[:t+1])``, recomputed as the oracle for
    :func:`pomata.pnl.cumulative_pnl`. Its subtlety is matching Polars' ``cum_sum`` missing-data rule: a ``None`` emits
    ``None`` while the running sum carries across it unchanged, whereas a ``nan`` enters the sum and propagates to every
    later row — detailed below.

    Args:
        returns: Input per-bar returns to cumulate (may contain ``None`` and ``float('nan')``).

    Returns:
        A list the same length as ``returns``: the running sum at each row.

    Raises:
        None.

    Note:
        Edge-case behavior:

        - **Null** — a ``None`` emits ``None`` at that row and the running sum carries across it unchanged.
        - **NaN** — a ``nan`` enters the running sum, so that row and every later row are ``nan``.
    """
    accumulator = 0.0
    results: list[float | None] = []
    for value in returns:
        if value is None:
            results.append(None)
        else:
            accumulator += value
            results.append(accumulator)
    return results
