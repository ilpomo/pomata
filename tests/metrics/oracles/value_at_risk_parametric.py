"""
Naive reference oracle for ``pomata.metrics.value_at_risk_parametric``.
"""

import math
from collections.abc import Sequence
from statistics import NormalDist


def value_at_risk_parametric_reference(returns: Sequence[float | None], confidence: float) -> float | None:
    """
    Naive parametric (Gaussian) value-at-risk over a Python list.

    The normal-distribution quantile ``mean + Phi_inv(1 - confidence) * std`` (sample ``std``, ``ddof = 1``), recomputed
    from scratch as the oracle for :func:`pomata.metrics.value_at_risk_parametric`. ``None`` returns are skipped; with
    fewer than two the result is ``None``; a ``nan`` anywhere poisons the result to ``nan``.
    """
    observations = [value for value in returns if value is not None]
    if len(observations) < 2:
        return None
    if any(math.isnan(value) for value in observations):
        return math.nan
    count = len(observations)
    mean = sum(observations) / count
    deviation = math.sqrt(sum((value - mean) ** 2 for value in observations) / (count - 1))
    z = NormalDist().inv_cdf(1.0 - confidence)
    return mean + z * deviation
