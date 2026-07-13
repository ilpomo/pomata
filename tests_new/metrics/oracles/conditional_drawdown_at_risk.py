"""
Naive reference oracle for ``pomata.metrics.conditional_drawdown_at_risk``.
"""

import math
from collections.abc import Sequence

from tests_new.metrics.oracles.drawdown import drawdown_reference


def conditional_drawdown_at_risk_reference(equity_curve: Sequence[float | None], confidence: float) -> float | None:
    """
    Naive conditional drawdown at risk (Rockafellar-Uryasev tail average) over a Python list.

    The Rockafellar-Uryasev tail average of the drawdown series: with ``k = (1 - confidence) * n``, the worst
    ``floor(k)`` order statistics are summed in full and the next carries the fractional weight ``k - floor(k)``,
    divided by ``k``. Recomputed from scratch as the oracle for :func:`pomata.metrics.conditional_drawdown_at_risk`.
    ``None`` equities are skipped; a ``nan`` anywhere poisons the result to ``nan``; with no observations the result is
    ``None``.
    """
    observations = [value for value in equity_curve if value is not None]
    if any(math.isnan(value) for value in observations):
        return math.nan
    if not observations:
        return None
    ascending = sorted(value for value in drawdown_reference(observations) if value is not None)
    n = len(ascending)
    k = (1.0 - confidence) * n
    floor_k = math.floor(k)
    total = sum(ascending[:floor_k])
    fraction = k - floor_k
    if floor_k < n and fraction > 0.0:
        total += fraction * ascending[floor_k]
    return total / k
