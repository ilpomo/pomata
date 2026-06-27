"""
Naive reference oracle for ``pomata.indicators.mom``.
"""

import math
from collections.abc import Sequence


def mom_reference(
    expr: Sequence[float | None],
    window: int,
) -> list[float | None]:
    """
    Naive Momentum (MOM) over a Python list.

    The fixed-lag difference ``x[t] - x[t - window]``, recomputed as the oracle for :func:`pomata.indicators.mom`.
    Being a difference of two endpoints rather than a recurrence, a missing value contaminates only the (at most two)
    positions that reference it — the property the notes below make precise.

    Args:
        expr: Input series, the observations to difference (may contain ``None`` and ``float('nan')``).
        window: Number of observations in the moving window, the look-back lag of the difference. Must be ``>= 1``.

    Returns:
        A list the same length as ``expr``. The first ``window`` entries are ``None`` (warm-up, clamped to the length);
        thereafter the fixed-lag difference ``x[t] - x[t - window]``.

    Raises:
        ValueError: If ``window < 1``.

    Note:
        Edge-case behavior:

        - **Null** — a position whose current value or whose ``window``-back value is ``None`` yields ``None``.
        - **NaN** — a position whose current value or whose ``window``-back value is ``nan`` (with no ``None``) yields
          ``nan``. Because the operation is a fixed-lag difference rather than a recurrence, a ``None`` or ``nan``
          contaminates only the (at most two) positions that reference it and never latches onto the rest of the series.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")

    results: list[float | None] = []
    for index in range(len(expr)):
        if index < window:
            results.append(None)
            continue
        value_current = expr[index]
        value_past = expr[index - window]
        if value_current is None or value_past is None:
            results.append(None)
        elif math.isnan(value_current) or (math.isnan(value_past)):
            results.append(math.nan)
        else:
            results.append(value_current - value_past)
    return results
