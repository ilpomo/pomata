"""
Naive reference oracle for ``pomata.metrics.information_ratio``.
"""

import math
from collections.abc import Sequence


def information_ratio_reference(
    returns: Sequence[float | None], benchmark: Sequence[float | None], periods_per_year: int
) -> float | None:
    """
    Naive annualized information ratio over two Python lists.

    The mean active return (portfolio minus benchmark) over its sample standard deviation (``ddof = 1``, the tracking
    error), annualized by ``sqrt(P)`` -- recomputed from scratch as the oracle for
    :func:`pomata.metrics.information_ratio`. The series are pairwise-complete: a pair contributes only where both legs
    are present; with fewer than two such pairs the result is ``None`` (taking precedence over poisoning); otherwise a
    ``nan`` in either leg of a retained pair poisons the result to ``nan``. A zero tracking error gives ``+/-inf`` (or
    ``nan`` when the mean active is also zero), matching the implementation's division; an exactly-constant active
    series is detected via ``min == max`` rather than the two-pass deviation (whose float residual is not reliably
    zero for every constant), matching the implementation's exact zero-dispersion pin.
    """
    active = [x - y for x, y in zip(returns, benchmark, strict=True) if x is not None and y is not None]
    if len(active) < 2:
        return None
    if any(math.isnan(value) for value in active):
        return math.nan
    if max(active) == min(active):
        return math.nan if active[0] == 0.0 else math.copysign(math.inf, active[0])
    count = len(active)
    mean_active = sum(active) / count
    tracking_error = math.sqrt(sum((value - mean_active) ** 2 for value in active) / (count - 1))
    if tracking_error == 0.0:
        return math.nan if mean_active == 0.0 else math.copysign(math.inf, mean_active)
    return mean_active / tracking_error * math.sqrt(periods_per_year)
