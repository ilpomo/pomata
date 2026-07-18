"""
Naive reference oracle for ``pomata.indicators.dm_plus``.
"""

from collections.abc import Sequence

from tests_new.indicators.oracles._helpers import difference, greater
from tests_new.indicators.oracles.rma import reference_rma


def reference_dm_plus(
    high: Sequence[float | None],
    low: Sequence[float | None],
    window: int,
) -> list[float | None]:
    """
    Naive Wilder-smoothed Plus Directional Movement over aligned Python lists.

    The raw plus directional movement (``up`` when it leads the down-move and is positive, else ``0``; ``0`` on the
    first bar) smoothed by :func:`reference_rma`, recomputed as the oracle for :func:`pomata.indicators.dm_plus`.

    Args:
        high: High-price series (may contain ``None`` and ``float('nan')``).
        low: Low-price series (may contain ``None`` and ``float('nan')``); same length as ``high``.
        window: Number of observations in the Wilder moving window. Must be ``>= 1``.

    Returns:
        A list the same length as the inputs: the smoothed plus directional movement, ``None`` through the
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
        raw.append(up if greater(up, down) and greater(up, 0.0) else 0.0)
    return reference_rma(raw, window)
