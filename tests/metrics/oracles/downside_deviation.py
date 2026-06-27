"""
Naive reference oracle for ``pomata.metrics.downside_deviation``.
"""

import math
from collections.abc import Sequence


def downside_deviation_reference(
    returns: Sequence[float | None],
    periods_per_year: int,
    threshold: float = 0.0,
) -> float | None:
    """
    Naive annualized downside deviation over a Python list.

    The root-mean-square of the below-threshold shortfall ``min(r - threshold, 0)`` over all non-null returns,
    annualized by ``sqrt(periods_per_year)``, recomputed from scratch as the oracle for
    :func:`pomata.metrics.downside_deviation`. ``None`` returns are skipped; a ``nan`` anywhere poisons the result to
    ``nan``; with no observations the result is ``None``.
    """
    if periods_per_year < 1:
        raise ValueError(f"periods_per_year must be >= 1, got {periods_per_year}")
    if not math.isfinite(threshold):
        raise ValueError(f"threshold must be a finite number, got {threshold}")
    observations = [value for value in returns if value is not None]
    if any(math.isnan(value) for value in observations):
        return math.nan
    if not observations:
        return None
    shortfalls = [min(value - threshold, 0.0) for value in observations]
    mean_square = sum(shortfall * shortfall for shortfall in shortfalls) / len(observations)
    return math.sqrt(mean_square) * math.sqrt(periods_per_year)
