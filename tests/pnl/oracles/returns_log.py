"""
Naive reference oracle for ``pomata.pnl.returns_log``.
"""

import math
from collections.abc import Sequence


def returns_log_reference(
    expr: Sequence[float | None],
) -> list[float | None]:
    """
    Naive Logarithmic Returns over a Python list.

    The natural log of the price relative ``ln(P_t / P_{t-1})`` against the previous observation, recomputed as the
    oracle for :func:`pomata.pnl.returns_log`. Its subtlety is matching the IEEE-754 logarithm Polars applies to the
    relative: a zero relative is ``-inf``, a negative relative is ``nan``, and a zero previous price makes the relative
    ``+inf`` (so the log is ``+inf``); like a fixed-lag transform it is a two-endpoint operation, so missing data never
    latches — detailed below.

    Args:
        expr: Input series, the prices to difference (may contain ``None`` and ``float('nan')``).

    Returns:
        A list the same length as ``expr``. The first entry is ``None`` (warm-up); thereafter ``ln(P_t / P_{t-1})``.

    Raises:
        None.

    Note:
        Edge-case behavior:

        - **Null** — a ``None`` at the current row or at the previous row yields ``None`` at that position.
        - **NaN** — a ``nan`` at the current row or at the previous row (and no ``None``) yields ``nan``.
        - **Domain** — reproducing Polars' IEEE-754 logarithm of the relative: a zero relative (``P_t = 0`` over a
          positive ``P_{t-1}``) is ``-inf``, a negative relative (the prices straddle zero) is ``nan``, and a zero
          previous price makes the relative ``+/-inf`` so the log is ``+inf`` (positive relative) or ``nan`` (negative).
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
        else:
            # ln(current / past), reproducing Polars' IEEE-754 boundary values for a zero / negative relative.
            if past == 0.0:
                relative = math.nan if current == 0.0 else math.copysign(math.inf, current) * math.copysign(1.0, past)
            else:
                relative = current / past
            if math.isnan(relative) or relative < 0.0:
                results.append(math.nan)
            elif relative == 0.0:
                results.append(-math.inf)
            elif math.isinf(relative):
                results.append(math.inf)
            else:
                results.append(math.log(relative))
    return results
