"""
Naive reference oracle for ``pomata.metrics.sterling_ratio``.
"""

import math
from collections.abc import Sequence

from tests.metrics.oracles.cagr import cagr_reference
from tests.metrics.oracles.pain_index import pain_index_reference


def sterling_ratio_reference(
    equity_curve: Sequence[float | None], periods_per_year: int, risk_free_rate: float, excess: float
) -> float | None:
    """
    Naive Sterling ratio over a Python list.

    The excess compound annual growth rate divided by the average drawdown plus a cushion --
    ``(CAGR - risk_free_rate) / (PI + excess)`` -- recomputed from scratch as the oracle for
    :func:`pomata.metrics.sterling_ratio` by composing the independent :func:`cagr_reference` and
    :func:`pain_index_reference`. ``None`` equities are skipped; a ``nan`` anywhere poisons the result to ``nan``; with
    no observations the result is ``None``. A zero denominator gives ``+/-inf`` (or ``nan`` when the excess growth is
    also zero).
    """
    growth = cagr_reference(equity_curve, periods_per_year)
    pain = pain_index_reference(equity_curve)
    if growth is None or pain is None:
        return None
    if math.isnan(growth) or math.isnan(pain):
        return math.nan
    excess_growth = growth - risk_free_rate
    denominator = pain + excess
    if denominator == 0.0:
        return math.nan if excess_growth == 0.0 else math.copysign(math.inf, excess_growth)
    return excess_growth / denominator
