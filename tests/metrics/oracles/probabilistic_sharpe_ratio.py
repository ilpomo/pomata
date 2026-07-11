"""
Naive reference oracle for ``pomata.metrics.probabilistic_sharpe_ratio``.
"""

import math
from collections.abc import Sequence
from statistics import NormalDist


def probabilistic_sharpe_ratio_reference(
    returns: Sequence[float | None], periods_per_year: int, benchmark_sharpe: float, risk_free_rate: float
) -> float | None:
    """
    Naive probabilistic Sharpe ratio over a Python list.

    The Bailey & López de Prado statistic ``Phi((SR - SR*) * sqrt(n - 1) / sqrt(1 - skew*SR + (kurt - 1)/4 * SR**2))``,
    recomputed from scratch as the oracle for :func:`pomata.metrics.probabilistic_sharpe_ratio`. ``SR`` is the
    non-annualized excess Sharpe ratio, ``skew`` the population skewness, and ``kurt`` the population (non-excess)
    kurtosis. ``None`` returns are skipped; with fewer than two the result is ``None``; a ``nan`` anywhere poisons the
    result to ``nan``; zero dispersion or a non-positive inner variance yields ``nan``.
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
    sharpe_ratio = mean_excess / math.sqrt(variance)
    mean_return = sum(observations) / count
    second_moment = sum((value - mean_return) ** 2 for value in observations) / count
    if second_moment == 0.0:
        return math.nan
    third_moment = sum((value - mean_return) ** 3 for value in observations) / count
    fourth_moment = sum((value - mean_return) ** 4 for value in observations) / count
    skewness = third_moment / math.pow(second_moment, 1.5)
    raw_kurtosis = fourth_moment / (second_moment * second_moment)
    inner = 1.0 - skewness * sharpe_ratio + (raw_kurtosis - 1.0) / 4.0 * sharpe_ratio * sharpe_ratio
    if inner <= 0.0:
        # ``inner < 0`` is out of domain (NaN). ``inner == 0`` is the measure-zero boundary where the statistic
        # diverges, so the CDF is the limiting 0 or 1 (the shipped factory's documented behavior) -- except on an
        # exactly-equal Sharpe, where 0 / 0 stays NaN.
        diverges = inner == 0.0 and sharpe_ratio != benchmark_sharpe
        return (1.0 if sharpe_ratio > benchmark_sharpe else 0.0) if diverges else math.nan
    argument = (sharpe_ratio - benchmark_sharpe) * math.sqrt(count - 1) / math.sqrt(inner)
    return NormalDist().cdf(argument)
