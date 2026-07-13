"""
Naive reference oracle for ``pomata.metrics.tail_ratio``.
"""

import math
from collections.abc import Sequence

from tests.metrics.oracles._quantile import type_seven_quantile


def tail_ratio_reference(returns: Sequence[float | None]) -> float | None:
    """
    Naive tail ratio over a Python list.

    The magnitude of the 95th-percentile return divided by the 5th-percentile return (both type-7 linear quantiles),
    recomputed from scratch as the oracle for :func:`pomata.metrics.tail_ratio`. ``None`` returns are skipped; a ``nan``
    anywhere poisons the result to ``nan``; with no observations the result is ``None``. A zero 5th-percentile gives the
    IEEE ``inf`` (or ``nan`` when the 95th percentile is also zero), matching the implementation's float division.
    """
    observations = [value for value in returns if value is not None]
    if any(math.isnan(value) for value in observations):
        return math.nan
    if not observations:
        return None
    ascending = sorted(observations)
    right = type_seven_quantile(ascending, 0.95)
    left = type_seven_quantile(ascending, 0.05)
    if left == 0.0:
        return math.nan if right == 0.0 else math.inf
    return abs(right / left)
