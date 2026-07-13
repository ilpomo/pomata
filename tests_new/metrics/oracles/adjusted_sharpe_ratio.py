"""
Naive reference oracle for ``pomata.metrics.adjusted_sharpe_ratio``.
"""

import math
from collections.abc import Sequence


def adjusted_sharpe_ratio_reference(
    returns: Sequence[float | None], periods_per_year: int, risk_free_rate: float
) -> float | None:
    """
    Naive adjusted Sharpe ratio over a Python list.

    The Pezier & White correction ``ASR_p = SR_p * (1 + skew/6 * SR_p - excess_kurt/24 * SR_p**2)`` applied to the
    per-period excess Sharpe ratio ``SR_p`` and the population skewness / excess kurtosis, then annualized by
    ``sqrt(periods_per_year)``, recomputed from scratch as the oracle for
    :func:`pomata.metrics.adjusted_sharpe_ratio`. ``None`` returns are skipped; with fewer than two the result is
    ``None``; a ``nan`` anywhere poisons the result to ``nan``; zero dispersion yields ``nan``.
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
    variance = sum((value - mean_excess) ** 2 for value in excess) / (count - 1)
    if variance == 0.0:
        return math.nan
    sharpe_per_period = mean_excess / math.sqrt(variance)
    mean_return = sum(observations) / count
    second_moment = sum((value - mean_return) ** 2 for value in observations) / count
    if second_moment == 0.0:
        return math.nan
    third_moment = sum((value - mean_return) ** 3 for value in observations) / count
    fourth_moment = sum((value - mean_return) ** 4 for value in observations) / count
    skewness = third_moment / math.pow(second_moment, 1.5)
    excess_kurtosis = fourth_moment / (second_moment * second_moment) - 3.0
    adjusted = sharpe_per_period * (
        1.0 + skewness / 6.0 * sharpe_per_period - excess_kurtosis / 24.0 * sharpe_per_period**2
    )
    return adjusted * math.sqrt(periods_per_year)
