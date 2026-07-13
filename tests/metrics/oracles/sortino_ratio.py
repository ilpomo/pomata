"""
Naive reference oracle for ``pomata.metrics.sortino_ratio``.
"""

import math
from collections.abc import Sequence


def sortino_ratio_reference(
    returns: Sequence[float | None], periods_per_year: int, risk_free_rate: float
) -> float | None:
    """
    Naive annualized Sortino ratio over a Python list.

    The mean excess return divided by the (population) downside deviation about the same target, annualized by
    ``sqrt(P)``, where the per-period risk-free target is ``(1 + risk_free_rate) ** (1 / P) - 1``. Recomputed from
    scratch as the oracle for :func:`pomata.metrics.sortino_ratio`. ``None`` returns are skipped; a ``nan`` anywhere
    poisons the result to ``nan``; with no observations the result is ``None``. No downside gives ``+/-inf`` (or ``nan``
    when the mean excess is also zero), matching the implementation's division.
    """
    observations = [value for value in returns if value is not None]
    if any(math.isnan(value) for value in observations):
        return math.nan
    if not observations:
        return None
    rf_period = math.pow(1.0 + risk_free_rate, 1.0 / periods_per_year) - 1.0
    excess = [value - rf_period for value in observations]
    count = len(excess)
    mean_excess = sum(excess) / count
    downside = math.sqrt(sum(min(value, 0.0) ** 2 for value in excess) / count)
    if downside == 0.0:
        return math.nan if mean_excess == 0.0 else math.copysign(math.inf, mean_excess)
    return mean_excess / downside * math.sqrt(periods_per_year)
