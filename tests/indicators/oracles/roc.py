"""
Naive reference oracle for ``pomata.indicators.roc``.
"""

import math
from collections.abc import Sequence


def roc_reference(
    expr: Sequence[float | None],
    window: int,
) -> list[float | None]:
    """
    Naive Rate of Change (ROC) over a Python list.

    The percentage change ``100 * (x - x_lag) / x_lag`` against the value ``window`` rows earlier, recomputed as the
    oracle for :func:`pomata.indicators.roc`. Its subtlety is the IEEE-754 behavior when the lagged value is zero (``0
    / 0`` is ``nan``, a non-zero change over zero is ``+/-inf``); like momentum it is a two-endpoint operation, so
    missing data never latches — detailed below.

    Args:
        expr: Input series, the observations to difference (may contain ``None`` and ``float('nan')``).
        window: Number of observations to look back. Must be ``>= 1``.

    Returns:
        A list the same length as ``expr``. The first ``window`` entries are ``None`` (warm-up, clamped to the length);
        thereafter the percentage change ``100 * (current - past) / past``.

    Raises:
        ValueError: If ``window < 1``.

    Note:
        Edge-case behavior:

        - **Null** — a ``None`` at the current row or at the lagged row yields ``None`` at that position.
        - **NaN** — a ``nan`` at the current row or at the lagged row (and no ``None``) yields ``nan``. Because the
          operation is a fixed-lag ratio of two endpoints rather than a recurrence, a ``None`` or ``nan`` contaminates
          only the positions that reference it and never latches onto the rest of the series.
        - **Division by zero** — when the lagged value is ``0`` the ratio divides by zero following IEEE-754: a zero
          change (``0 / 0``) is ``nan`` and a non-zero change over zero is ``+/-inf`` (the sign tracks the change
          relative to the signed zero).
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")

    results: list[float | None] = []
    for index in range(len(expr)):
        if index < window:
            results.append(None)
            continue
        current = expr[index]
        past = expr[index - window]
        if current is None or past is None:
            results.append(None)
        elif math.isnan(current) or math.isnan(past):
            results.append(math.nan)
        else:
            numerator = 100.0 * (current - past)
            if past == 0.0:
                if numerator == 0.0:
                    results.append(math.nan)
                else:
                    results.append(math.copysign(math.inf, numerator) * math.copysign(1.0, past))
            else:
                results.append(numerator / past)
    return results
