"""
Naive reference oracle for ``pomata.metrics.max_drawdown_duration``.
"""

import math
from collections.abc import Sequence


def max_drawdown_duration_reference(equity_curve: Sequence[float | None]) -> float | None:
    """
    Naive maximum drawdown duration over a Python list.

    The longest run of consecutive observations strictly below a prior peak, recomputed from scratch as the oracle for
    :func:`pomata.metrics.max_drawdown_duration` (returned as a ``float`` count of bars over the non-null equity).
    ``None`` equities are skipped; a ``nan`` anywhere poisons the result to ``nan``; with no observations the result is
    ``None``.
    """
    observations = [value for value in equity_curve if value is not None]
    if any(math.isnan(value) for value in observations):
        return math.nan
    if not observations:
        return None
    peak = observations[0]
    longest = 0
    current = 0
    for value in observations:
        peak = max(peak, value)
        if value < peak:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return float(longest)
