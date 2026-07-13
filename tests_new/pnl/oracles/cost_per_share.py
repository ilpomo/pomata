"""
Naive reference oracle for ``pomata.pnl.cost_per_share``.
"""

import math
from collections.abc import Sequence

from tests_new.pnl.oracles.turnover import turnover_reference


def cost_per_share_reference(
    quantity: Sequence[float | None],
    fee: float,
) -> list[float | None]:
    """
    Naive Per-Share Transaction Cost over a Python list.

    The units traded times a flat per-unit fee, ``turnover(quantity) * fee``, recomputed as the oracle for
    :func:`pomata.pnl.cost_per_share` by composing the independent :func:`turnover_reference` with the fee. Its
    subtleties are inherited from turnover (flat start, null voids its own row and the next, NaN propagates) — detailed
    below.

    Args:
        quantity: Signed position sizes for each bar (may contain ``None`` and ``float('nan')``).
        fee: Commission per unit traded, in the account currency. Must be ``>= 0``.

    Returns:
        A list the same length as ``quantity``: the per-bar per-share cost, with the first row ``|quantity[0]| * fee``.

    Raises:
        ValueError: If ``fee`` is not a finite number ``>= 0`` (i.e. ``< 0``, ``NaN``, or ``±inf``).

    Note:
        Edge-case behavior:

        - **Flat start** — the pre-series quantity is ``0``, so the first row is ``|quantity[0]| * fee``.
        - **Null** — a ``None`` voids its own row and the next (via turnover); ``None`` takes precedence over ``nan``.
        - **NaN** — a ``nan`` propagates to its own row and the next.
    """
    if not math.isfinite(fee) or fee < 0.0:
        raise ValueError(f"fee must be a finite number >= 0, got {fee}")

    return [None if traded is None else traded * fee for traded in turnover_reference(quantity)]
