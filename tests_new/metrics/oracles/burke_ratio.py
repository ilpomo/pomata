"""
Naive reference oracle for ``pomata.metrics.burke_ratio``.
"""

import math
from collections.abc import Sequence

from tests_new.metrics.oracles.cagr import cagr_reference
from tests_new.metrics.oracles.drawdown import drawdown_reference


def burke_ratio_reference(
    equity_curve: Sequence[float | None], periods_per_year: int, risk_free_rate: float
) -> float | None:
    """
    Naive Burke ratio over a Python list.

    The excess compound annual growth rate divided by the square root of the sum of squared drawdowns --
    ``(CAGR - risk_free_rate) / sqrt(sum(D_i**2))`` -- recomputed from scratch as the oracle for
    :func:`pomata.metrics.burke_ratio` by composing the independent :func:`cagr_reference` and
    :func:`drawdown_reference`.
    ``None`` equities are skipped; a ``nan`` anywhere poisons the result to ``nan``; with no observations the result is
    ``None``. A drawdown-free curve gives ``+/-inf`` (or ``nan`` when the excess growth is also zero).
    """
    growth = cagr_reference(equity_curve, periods_per_year)
    if growth is None:
        return None
    if math.isnan(growth):
        return math.nan
    observations = [value for value in equity_curve if value is not None]
    declines = [value for value in drawdown_reference(observations) if value is not None]
    denominator = math.sqrt(sum(value * value for value in declines))
    excess_growth = growth - risk_free_rate
    if denominator == 0.0:
        return math.nan if excess_growth == 0.0 else math.copysign(math.inf, excess_growth)
    return excess_growth / denominator
