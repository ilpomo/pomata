"""
Naive reference oracle for ``pomata.pnl.cost_proportional``.
"""

import math
from collections.abc import Sequence

from tests_new.pnl.oracles.turnover import turnover_reference


def cost_proportional_reference(
    weight: Sequence[float | None],
    rate: float,
) -> list[float | None]:
    """
    Naive Proportional Transaction Cost over a Python list.

    The traded fraction times a flat rate, ``turnover(weight) * rate``, recomputed as the oracle for
    :func:`pomata.pnl.cost_proportional` by composing the independent :func:`turnover_reference` with the rate. Its
    subtleties are inherited from turnover (flat start, null voids its own row and the next, NaN propagates) — detailed
    below.

    Args:
        weight: Signed weights for each bar (may contain ``None`` and ``float('nan')``).
        rate: Proportional cost rate, the fee as a fraction of traded notional. Must be ``>= 0``.

    Returns:
        A list the same length as ``weight``: the per-bar proportional cost, with the first row
        ``|weight[0]| * rate``.

    Raises:
        ValueError: If ``rate`` is not a finite number ``>= 0`` (i.e. ``< 0``, ``NaN``, or ``±inf``).

    Note:
        Edge-case behavior:

        - **Flat start** — the pre-series weight is ``0``, so the first row is ``|weight[0]| * rate``.
        - **Null** — a ``None`` voids its own row and the next (via turnover); ``None`` takes precedence over ``nan``.
        - **NaN** — a ``nan`` propagates to its own row and the next.
    """
    if not math.isfinite(rate) or rate < 0.0:
        raise ValueError(f"rate must be a finite number >= 0, got {rate}")

    return [None if traded is None else traded * rate for traded in turnover_reference(weight)]
