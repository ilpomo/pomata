"""
Naive reference oracle for ``pomata.metrics.stability``.
"""

import math
from collections.abc import Sequence


def stability_reference(returns: Sequence[float | None]) -> float | None:
    """
    Naive trend stability over a Python list.

    The coefficient of determination of an ordinary-least-squares fit of the cumulative log returns on time, recomputed
    from scratch as the oracle for :func:`pomata.metrics.stability` (``R**2 = corr(t, cumulative_log)**2``). ``None``
    returns are skipped and the time index runs over the retained observations; a ``nan`` anywhere poisons the result
    to ``nan``, even as the only observation (the poison wins over the count guard, as in the cagr / total_return
    siblings); with fewer than two the result is ``None``; a return at or below ``-1`` (undefined log) poisons the
    result to ``nan``; a perfectly flat cumulative path gives ``nan``.
    """
    observations = [value for value in returns if value is not None]
    if any(math.isnan(value) for value in observations):
        return math.nan
    if len(observations) < 2:
        return None
    if any(value <= -1.0 for value in observations):
        return math.nan
    cumulative: list[float] = []
    running = 0.0
    for value in observations:
        running += math.log1p(value)
        cumulative.append(running)
    count = len(cumulative)
    index = list(range(count))
    mean_index = sum(index) / count
    mean_cumulative = sum(cumulative) / count
    covariance = sum((index[i] - mean_index) * (cumulative[i] - mean_cumulative) for i in range(count))
    variance_index = sum((value - mean_index) ** 2 for value in index)
    variance_cumulative = sum((value - mean_cumulative) ** 2 for value in cumulative)
    if variance_index == 0.0 or variance_cumulative == 0.0:
        return math.nan
    correlation = covariance / math.sqrt(variance_index * variance_cumulative)
    return correlation * correlation
