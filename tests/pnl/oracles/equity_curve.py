"""
Naive reference oracle for ``pomata.pnl.equity_curve``.
"""

from collections.abc import Sequence


def equity_curve_reference(
    returns: Sequence[float | None],
) -> list[float | None]:
    """
    Naive Equity Curve over a Python list.

    The compounded running product ``prod(1 + returns[:t+1])``, recomputed as the oracle for
    :func:`pomata.pnl.equity_curve`. Its subtlety is matching Polars' ``cum_prod`` missing-data rule: a ``None`` emits
    ``None`` while the running product carries across it unchanged (a missing bar is a neutral factor of one), whereas a
    ``nan`` enters the product and propagates to every later row — detailed below.

    Args:
        returns: Input per-bar returns to compound (may contain ``None`` and ``float('nan')``).

    Returns:
        A list the same length as ``returns``: the compounded equity at each row, relative to a starting capital of
        ``1``.

    Raises:
        None.

    Note:
        Edge-case behavior:

        - **Null** — a ``None`` emits ``None`` at that row and the running product carries across it unchanged; a
          leading warm-up ``None`` therefore stays ``None`` and the curve begins at the first defined return.
        - **NaN** — a ``nan`` enters the running product, so that row and every later row are ``nan``.
    """
    accumulator = 1.0
    results: list[float | None] = []
    for value in returns:
        if value is None:
            results.append(None)
        else:
            accumulator *= 1.0 + value
            results.append(accumulator)
    return results
