"""
Naive reference oracle for ``pomata.metrics.conditional_value_at_risk``.
"""

import math
from collections.abc import Sequence

from tests.metrics.oracles._quantile import type_seven_quantile


def conditional_value_at_risk_reference(returns: Sequence[float | None], confidence: float) -> float | None:
    """
    Naive historical conditional value-at-risk (expected shortfall) over a Python list.

    The mean of the non-null returns at or below the ``1 - confidence`` empirical quantile (type-7 linear
    interpolation), recomputed from scratch as the oracle for :func:`pomata.metrics.conditional_value_at_risk`.
    ``None`` returns are skipped; a ``nan`` anywhere poisons the result to ``nan``; with no observations the result is
    ``None``. The shortfall set always contains at least the smallest return, so the mean is defined whenever any
    observation is present.
    """
    observations = [value for value in returns if value is not None]
    if any(math.isnan(value) for value in observations):
        return math.nan
    if not observations:
        return None
    ascending = sorted(observations)
    threshold = type_seven_quantile(ascending, 1.0 - confidence)
    shortfall = [value for value in ascending if value <= threshold]
    return sum(shortfall) / len(shortfall)
