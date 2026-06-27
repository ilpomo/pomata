"""
Naive reference oracle for ``pomata.metrics.cagr``.
"""

import math
from collections.abc import Sequence


def cagr_reference(equity_curve: Sequence[float | None], periods_per_year: int) -> float | None:
    """
    Naive compound annual growth rate over a Python list.

    ``final ** (periods_per_year / n) - 1`` where ``final`` is the last non-null equity and ``n`` the count of non-null
    observations, recomputed from scratch as the oracle for :func:`pomata.metrics.cagr`. ``None`` equities are skipped;
    a ``nan`` anywhere poisons the result to ``nan``; with no defined observations the result is ``None``.
    """
    if periods_per_year < 1:
        raise ValueError(f"periods_per_year must be >= 1, got {periods_per_year}")
    defined = [value for value in equity_curve if value is not None]
    if any(math.isnan(value) for value in defined):
        return math.nan
    if not defined:
        return None
    return math.pow(defined[-1], periods_per_year / len(defined)) - 1.0
