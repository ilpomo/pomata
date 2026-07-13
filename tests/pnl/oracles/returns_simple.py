"""
Naive reference oracle for ``pomata.pnl.returns_simple``.
"""

import math
from collections.abc import Sequence


def returns_simple_reference(
    expr: Sequence[float | None],
) -> list[float | None]:
    """
    Naive Simple Returns over a Python list.

    The fractional change ``P_t / P_{t-1} - 1`` against the previous observation, recomputed as the oracle for
    :func:`pomata.pnl.returns_simple`. Its subtlety is the IEEE-754 behavior when the previous value is zero (``0 / 0``
    is ``nan``, a non-zero change over zero is ``+/-inf``); like a fixed-lag difference it is a two-endpoint operation,
    so missing data never latches — detailed below.

    Args:
        expr: Input series, the prices to difference (may contain ``None`` and ``float('nan')``).

    Returns:
        A list the same length as ``expr``. The first entry is ``None`` (warm-up); thereafter the simple return
        ``P_t / P_{t-1} - 1``.

    Raises:
        None.

    Note:
        Edge-case behavior:

        - **Null** — a ``None`` at the current row or at the previous row yields ``None`` at that position.
        - **NaN** — a ``nan`` at the current row or at the previous row (and no ``None``) yields ``nan``.
        - **Division by zero** — when the previous value is ``0`` the relative divides by zero following IEEE-754: a
          zero change (``0 / 0``) is ``nan`` and a non-zero change over zero is ``+/-inf`` (the sign tracks the change
          relative to the signed zero).
    """
    results: list[float | None] = []
    for index in range(len(expr)):
        if index == 0:
            results.append(None)
            continue
        current = expr[index]
        past = expr[index - 1]
        if current is None or past is None:
            results.append(None)
        elif math.isnan(current) or math.isnan(past):
            results.append(math.nan)
        elif past == 0.0:
            # P_t / 0 - 1 following IEEE-754: 0/0 is nan, a non-zero change over signed zero is +/-inf.
            if current == 0.0:
                results.append(math.nan)
            else:
                results.append(math.copysign(math.inf, current) * math.copysign(1.0, past))
        else:
            results.append(current / past - 1.0)
    return results
