"""
Naive reference oracle for ``pomata.metrics.calmar_ratio``.
"""

import math
from collections.abc import Sequence

from tests_new.metrics.oracles.cagr import cagr_reference
from tests_new.metrics.oracles.max_drawdown import max_drawdown_reference


def calmar_ratio_reference(equity_curve: Sequence[float | None], periods_per_year: int) -> float | None:
    """
    Naive Calmar ratio over a Python list.

    The compound annual growth rate divided by the magnitude of the maximum drawdown -- ``CAGR / |MDD|`` -- recomputed
    from scratch as the oracle for :func:`pomata.metrics.calmar_ratio` by composing the independent
    :func:`cagr_reference` and :func:`max_drawdown_reference`. ``None`` equities are skipped; a ``nan`` anywhere poisons
    it to ``nan``; with no defined observations the result is ``None``. A drawdown-free (monotonic) curve gives
    ``+/-inf`` (or ``nan`` when the growth is zero), matching the implementation's division.
    """
    growth = cagr_reference(equity_curve, periods_per_year)
    drawdown_trough = max_drawdown_reference(equity_curve)
    if growth is None or drawdown_trough is None:
        return None
    if math.isnan(growth) or math.isnan(drawdown_trough):
        return math.nan
    denominator = abs(drawdown_trough)
    if denominator == 0.0:
        return math.nan if growth == 0.0 else math.copysign(math.inf, growth)
    return growth / denominator
