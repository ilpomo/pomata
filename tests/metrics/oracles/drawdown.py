"""
Naive reference oracle for ``pomata.metrics.drawdown``.
"""

import math
from collections.abc import Sequence


def drawdown_reference(equity_curve: Sequence[float | None]) -> list[float | None]:
    """
    Naive running drawdown over a Python list.

    For each row, ``equity / running_peak - 1`` where the running peak is the maximum of the non-null, non-nan equities
    seen so far, recomputed from scratch as the oracle for :func:`pomata.metrics.drawdown`. A ``None`` equity yields
    ``None`` (the peak carries across it); a ``nan`` equity yields ``nan`` and is ignored by the running peak (matching
    Polars' ``cum_max``), so later rows are unaffected.
    """
    result: list[float | None] = []
    running_peak: float | None = None
    for value in equity_curve:
        if value is None:
            result.append(None)
            continue
        if math.isnan(value):
            result.append(math.nan)
            continue
        running_peak = value if running_peak is None else max(running_peak, value)
        result.append(value / running_peak - 1)
    return result
