"""
Naive reference oracle for ``pomata.metrics.volatility``.
"""

import math
from collections.abc import Sequence


def volatility_reference(returns: Sequence[float | None], periods_per_year: int) -> float | None:
    """
    Naive annualized sample standard deviation over a Python list.

    The two-pass sample standard deviation (``ddof = 1``) of the non-null returns, annualized by
    ``sqrt(periods_per_year)``, recomputed from scratch as the oracle for :func:`pomata.metrics.volatility`. ``None``
    returns are skipped; with fewer than two remaining observations the standard deviation is undefined, so the result
    is ``None``; otherwise a ``nan`` propagates to the result. An exactly-constant series has zero dispersion, detected
    via ``min == max`` rather than the two-pass variance (whose float residual is not reliably zero for every
    constant), and is reported as exactly ``0.0``, matching the implementation's exact zero-dispersion pin.
    """
    if periods_per_year < 1:
        raise ValueError(f"periods_per_year must be >= 1, got {periods_per_year}")
    observations = [value for value in returns if value is not None]
    if len(observations) < 2:
        return None
    if any(math.isnan(value) for value in observations):
        return math.nan
    if max(observations) == min(observations):
        return 0.0
    mean = sum(observations) / len(observations)
    variance = sum((value - mean) ** 2 for value in observations) / (len(observations) - 1)
    return math.sqrt(variance) * math.sqrt(periods_per_year)
