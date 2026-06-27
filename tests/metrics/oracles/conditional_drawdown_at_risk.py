"""
Naive reference oracle for ``pomata.metrics.conditional_drawdown_at_risk``.
"""

import math
from collections.abc import Sequence

from tests.metrics.oracles._quantile import type_seven_quantile
from tests.metrics.oracles.drawdown import drawdown_reference


def conditional_drawdown_at_risk_reference(equity_curve: Sequence[float | None], confidence: float) -> float | None:
    """
    Naive conditional drawdown at risk over a Python list.

    The mean of the drawdowns at or below the ``1 - confidence`` empirical quantile (type-7 linear interpolation) of the
    drawdown series, recomputed from scratch as the oracle for :func:`pomata.metrics.conditional_drawdown_at_risk`.
    ``None`` equities are skipped; a ``nan`` anywhere poisons the result to ``nan``; with no observations the result is
    ``None``.
    """
    observations = [value for value in equity_curve if value is not None]
    if any(math.isnan(value) for value in observations):
        return math.nan
    if not observations:
        return None
    declines = [value for value in drawdown_reference(observations) if value is not None]
    ascending = sorted(declines)
    threshold = type_seven_quantile(ascending, 1.0 - confidence)
    tail = [value for value in ascending if value <= threshold]
    return sum(tail) / len(tail)
