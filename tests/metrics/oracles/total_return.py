"""
Naive reference oracle for ``pomata.metrics.total_return``.
"""

import math
from collections.abc import Sequence


def total_return_reference(equity_curve: Sequence[float | None]) -> float | None:
    """
    Naive total return over a Python list: the last non-null equity minus one.

    Recomputed from scratch as the oracle for :func:`pomata.metrics.total_return`. ``None`` equities are skipped; a
    ``nan`` anywhere poisons the result to ``nan``; with no defined observations the result is ``None``.
    """
    defined = [value for value in equity_curve if value is not None]
    if any(math.isnan(value) for value in defined):
        return math.nan
    if not defined:
        return None
    return defined[-1] - 1
