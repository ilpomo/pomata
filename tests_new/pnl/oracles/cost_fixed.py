"""
Naive reference oracle for ``pomata.pnl.cost_fixed``.
"""

import math
from collections.abc import Sequence

from tests_new.pnl.oracles.turnover import turnover_reference


def cost_fixed_reference(
    quantity: Sequence[float | None],
    fee: float,
) -> list[float | None]:
    """
    Naive Fixed Transaction Cost over a Python list.

    A flat ``fee`` on every bar where the position changes (``turnover(quantity) > 0``) and ``0`` where it is held,
    recomputed as the oracle for :func:`pomata.pnl.cost_fixed` by composing the independent :func:`turnover_reference`.
    Its subtleties are inherited from turnover (flat start, null voids its own row and the next, NaN propagates) —
    detailed below.

    Args:
        quantity: Signed position sizes for each bar (may contain ``None`` and ``float('nan')``).
        fee: Flat charge per trade, in the account currency. Must be ``>= 0``.

    Returns:
        A list the same length as ``quantity``: ``fee`` where the quantity changes (the first row counts as a trade from
        a flat start) and ``0`` where it is held.

    Raises:
        ValueError: If ``fee`` is not a finite number ``>= 0`` (i.e. ``< 0``, ``NaN``, or ``±inf``).

    Note:
        Edge-case behavior:

        - **Flat start** — the pre-series quantity is ``0``, so the first row charges the ``fee``.
        - **Null** — a ``None`` voids its own row and the next (via turnover); ``None`` takes precedence over ``nan``.
        - **NaN** — a ``nan`` propagates to its own row and the next.
    """
    if not math.isfinite(fee) or fee < 0.0:
        raise ValueError(f"fee must be a finite number >= 0, got {fee}")

    results: list[float | None] = []
    for traded in turnover_reference(quantity):
        if traded is None:
            results.append(None)
        elif math.isnan(traded):
            results.append(math.nan)
        elif traded > 0.0:
            results.append(fee)
        else:
            results.append(0.0)
    return results
