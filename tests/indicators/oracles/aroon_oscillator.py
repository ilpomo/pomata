"""
Naive reference oracle for ``pomata.indicators.aroon_oscillator``.
"""

import math
from collections.abc import Sequence

from tests.indicators.oracles.aroon import reference_aroon


def reference_aroon_oscillator(
    high: Sequence[float | None],
    low: Sequence[float | None],
    window: int,
) -> list[float | None]:
    """
    Naive Aroon Oscillator over aligned Python lists.

    Aroon Up minus Aroon Down (:func:`reference_aroon`), recomputed as the oracle for
    :func:`pomata.indicators.aroon_oscillator`.

    Args:
        high: High-price series (may contain ``None`` and ``float('nan')``).
        low: Low-price series (may contain ``None`` and ``float('nan')``); same length as ``high``.
        window: Look-back length; the extremes are sought over the last ``window + 1`` bars. Must be ``>= 1``.

    Returns:
        A list the same length as the inputs: ``up - down``, ``None`` through the ``window`` warm-up rows.

    Raises:
        ValueError: If ``window < 1`` or the inputs differ in length.
    """
    bands = reference_aroon(high, low, window)
    result: list[float | None] = []
    for up_value, down_value in zip(bands["up"], bands["down"], strict=True):
        if up_value is None or down_value is None:
            result.append(None)
        elif math.isnan(up_value) or math.isnan(down_value):
            result.append(math.nan)
        else:
            result.append(up_value - down_value)
    return result
