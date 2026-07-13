"""
Naive reference oracle for ``pomata.indicators.dm_minus``.
"""

from collections.abc import Sequence

from tests_new.indicators.oracles._helpers import difference, greater
from tests_new.indicators.oracles.rma import rma_reference


def dm_minus_reference(
    high: Sequence[float | None],
    low: Sequence[float | None],
    window: int,
) -> list[float | None]:
    """
    Naive Wilder-smoothed Minus Directional Movement over aligned Python lists.

    The raw minus directional movement (``down`` when it leads the up-move and is positive, else ``0``; ``0`` on the
    first bar) smoothed by :func:`rma_reference`, recomputed as the oracle for :func:`pomata.indicators.dm_minus`.

    Args:
        high: High-price series (may contain ``None`` and ``float('nan')``).
        low: Low-price series (may contain ``None`` and ``float('nan')``); same length as ``high``.
        window: Number of observations in the Wilder moving window. Must be ``>= 1``.

    Returns:
        A list the same length as the inputs: the smoothed minus directional movement, ``None`` through the
        ``window - 1`` warm-up rows.

    Raises:
        ValueError: If ``window < 1``.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    raw: list[float | None] = []
    for index in range(len(high)):
        if index == 0:
            raw.append(0.0)
            continue
        up = difference(high[index], high[index - 1])
        down = difference(low[index - 1], low[index])
        raw.append(down if greater(down, up) and greater(down, 0.0) else 0.0)
    return rma_reference(raw, window)
