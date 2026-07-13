"""
Naive reference oracle for ``pomata.pnl.cost_slippage``.
"""

import math
from collections.abc import Sequence

from tests_new.pnl.oracles.turnover import turnover_reference


def cost_slippage_reference(
    weight: Sequence[float | None],
    half_spread: float,
) -> list[float | None]:
    """
    Naive Slippage Cost over a Python list.

    The traded fraction times a fixed half-spread, ``turnover(weight) * half_spread``, recomputed as the oracle for
    :func:`pomata.pnl.cost_slippage` by composing the independent :func:`turnover_reference` with the half-spread. Its
    subtleties are inherited from turnover (flat start, null voids its own row and the next, NaN propagates) — detailed
    below.

    Args:
        weight: Signed weights for each bar (may contain ``None`` and ``float('nan')``).
        half_spread: Fixed bid-ask half-spread crossed per trade, as a fraction. Must be ``>= 0``.

    Returns:
        A list the same length as ``weight``: the per-bar slippage cost, with the first row
        ``|weight[0]| * half_spread``.

    Raises:
        ValueError: If ``half_spread`` is not a finite number ``>= 0`` (i.e. ``< 0``, ``NaN``, or ``±inf``).

    Note:
        Edge-case behavior:

        - **Flat start** — the pre-series weight is ``0``, so the first row is ``|weight[0]| * half_spread``.
        - **Null** — a ``None`` voids its own row and the next (via turnover); ``None`` takes precedence over ``nan``.
        - **NaN** — a ``nan`` propagates to its own row and the next.
    """
    if not math.isfinite(half_spread) or half_spread < 0.0:
        raise ValueError(f"half_spread must be a finite number >= 0, got {half_spread}")

    return [None if traded is None else traded * half_spread for traded in turnover_reference(weight)]
