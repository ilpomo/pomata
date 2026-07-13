"""
Naive reference oracle for ``pomata.metrics.profit_factor``.
"""

import math
from collections.abc import Sequence


def profit_factor_reference(returns: Sequence[float | None]) -> float | None:
    """
    Naive profit factor over a Python list.

    The sum of the positive returns over the magnitude of the sum of the negative returns, recomputed from scratch as
    the oracle for :func:`pomata.metrics.profit_factor`. ``None`` returns are skipped; a ``nan`` anywhere poisons the
    result to ``nan``; with no observations the result is ``None``; with no losses the result is ``+inf`` (or ``nan``
    when there are also no gains), matching the implementation's division.
    """
    observations = [value for value in returns if value is not None]
    if any(math.isnan(value) for value in observations):
        return math.nan
    if not observations:
        return None
    gain = sum(value for value in observations if value > 0.0)
    loss = sum(-value for value in observations if value < 0.0)
    if loss == 0.0:
        return math.inf if gain > 0.0 else math.nan
    return gain / loss
