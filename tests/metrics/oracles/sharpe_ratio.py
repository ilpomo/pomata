"""
Naive reference oracle for ``pomata.metrics.sharpe_ratio``.
"""

import math
from collections.abc import Sequence


def sharpe_ratio_reference(
    returns: Sequence[float | None], periods_per_year: int, risk_free_rate: float
) -> float | None:
    """
    Naive annualized Sharpe ratio over a Python list.

    The mean excess return divided by its sample standard deviation (``ddof = 1``), annualized by ``sqrt(P)``, where the
    per-period risk-free rate is the geometric conversion ``(1 + risk_free_rate) ** (1 / P) - 1``. Recomputed from
    scratch as the oracle for :func:`pomata.metrics.sharpe_ratio`. ``None`` returns are skipped; with fewer than two
    observations the result is ``None`` (the undefined sample standard deviation takes precedence); otherwise a ``nan``
    anywhere poisons the result to ``nan``. Zero dispersion gives ``+/-inf`` (or ``nan`` when the mean excess is also
    zero), matching the implementation's division.
    """
    observations = [value for value in returns if value is not None]
    if len(observations) < 2:
        return None
    if any(math.isnan(value) for value in observations):
        return math.nan
    rf_period = math.pow(1.0 + risk_free_rate, 1.0 / periods_per_year) - 1.0
    excess = [value - rf_period for value in observations]
    count = len(excess)
    mean_excess = sum(excess) / count
    deviation = math.sqrt(sum((value - mean_excess) ** 2 for value in excess) / (count - 1))
    if deviation == 0.0:
        return math.nan if mean_excess == 0.0 else math.copysign(math.inf, mean_excess)
    return mean_excess / deviation * math.sqrt(periods_per_year)
