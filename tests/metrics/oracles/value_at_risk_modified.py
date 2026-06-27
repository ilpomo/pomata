"""
Naive reference oracle for ``pomata.metrics.value_at_risk_modified``.
"""

import math
from collections.abc import Sequence
from statistics import NormalDist


def value_at_risk_modified_reference(returns: Sequence[float | None], confidence: float) -> float | None:
    """
    Naive modified (Cornish-Fisher) value-at-risk over a Python list.

    The Gaussian quantile adjusted for skewness and excess kurtosis via the Cornish-Fisher expansion,
    ``mean + z_cf * std`` (sample ``std``, ``ddof = 1``), recomputed from scratch as the oracle for
    :func:`pomata.metrics.value_at_risk_modified`. ``None`` returns are skipped; with fewer than two the result is
    ``None``; a ``nan`` anywhere poisons it to ``nan``; zero dispersion (undefined skew/kurtosis) yields ``nan``.
    """
    observations = [value for value in returns if value is not None]
    if len(observations) < 2:
        return None
    if any(math.isnan(value) for value in observations):
        return math.nan
    count = len(observations)
    mean = sum(observations) / count
    deviation = math.sqrt(sum((value - mean) ** 2 for value in observations) / (count - 1))
    second_moment = sum((value - mean) ** 2 for value in observations) / count
    if second_moment == 0.0:
        return math.nan
    third_moment = sum((value - mean) ** 3 for value in observations) / count
    fourth_moment = sum((value - mean) ** 4 for value in observations) / count
    skewness = third_moment / math.pow(second_moment, 1.5)
    excess_kurtosis = fourth_moment / (second_moment * second_moment) - 3.0
    z = NormalDist().inv_cdf(1.0 - confidence)
    z_cornish_fisher = (
        z
        + (z**2 - 1.0) / 6.0 * skewness
        + (z**3 - 3.0 * z) / 24.0 * excess_kurtosis
        - (2.0 * z**3 - 5.0 * z) / 36.0 * skewness**2
    )
    return mean + z_cornish_fisher * deviation
