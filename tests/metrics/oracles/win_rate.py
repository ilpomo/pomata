"""
Naive reference oracle for ``pomata.metrics.win_rate``.
"""

import math
from collections.abc import Sequence


def win_rate_reference(returns: Sequence[float | None]) -> float | None:
    """
    Naive win rate over a Python list.

    The count of strictly positive returns over the count of non-zero returns, recomputed from scratch as the oracle
    for :func:`pomata.metrics.win_rate`. ``None`` returns are skipped; a ``nan`` anywhere poisons the result to ``nan``;
    a return of exactly ``0`` is excluded from the denominator; with no non-zero returns the result is ``None``.
    """
    observations = [value for value in returns if value is not None]
    if any(math.isnan(value) for value in observations):
        return math.nan
    decisive = [value for value in observations if value != 0.0]
    if not decisive:
        return None
    wins = sum(1 for value in decisive if value > 0.0)
    return wins / len(decisive)
