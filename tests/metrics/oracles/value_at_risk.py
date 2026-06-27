"""
Naive reference oracle for ``pomata.metrics.value_at_risk``.
"""

import math
from collections.abc import Sequence

from tests.metrics.oracles._quantile import type_seven_quantile


def value_at_risk_reference(returns: Sequence[float | None], confidence: float) -> float | None:
    """
    Naive historical value-at-risk over a Python list.

    The ``1 - confidence`` empirical quantile (type-7 linear interpolation) of the non-null returns, recomputed from
    scratch as the oracle for :func:`pomata.metrics.value_at_risk`. ``None`` returns are skipped; a ``nan`` anywhere
    poisons the result to ``nan``; with no observations the result is ``None``.
    """
    observations = [value for value in returns if value is not None]
    if any(math.isnan(value) for value in observations):
        return math.nan
    if not observations:
        return None
    return type_seven_quantile(sorted(observations), 1.0 - confidence)
