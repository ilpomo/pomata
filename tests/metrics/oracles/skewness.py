"""
Naive reference oracle for ``pomata.metrics.skewness``.
"""

import math
from collections.abc import Sequence


def skewness_reference(returns: Sequence[float | None]) -> float | None:
    """
    Naive population skewness over a Python list.

    ``m3 / m2**1.5`` from the population central moments of the non-null returns, recomputed from scratch as the oracle
    for :func:`pomata.metrics.skewness`. ``None`` returns are skipped; a ``nan`` anywhere poisons the result to ``nan``;
    with no observations the result is ``None``; a zero-variance series (constant, or a single value) gives ``0 / 0``
    and the result is ``nan``, as does a subnormal-magnitude series whose ``m2 ** 1.5`` underflows to zero (matching the
    implementation, which yields ``nan`` there too).
    """
    observations = [value for value in returns if value is not None]
    if any(math.isnan(value) for value in observations):
        return math.nan
    if not observations:
        return None
    count = len(observations)
    mean = sum(observations) / count
    second_moment = sum((value - mean) ** 2 for value in observations) / count
    denominator = math.pow(second_moment, 1.5)
    if denominator == 0.0:
        return math.nan
    third_moment = sum((value - mean) ** 3 for value in observations) / count
    return third_moment / denominator
