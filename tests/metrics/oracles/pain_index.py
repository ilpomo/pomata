"""
Naive reference oracle for ``pomata.metrics.pain_index``.
"""

import math
from collections.abc import Sequence

from tests.metrics.oracles.drawdown import drawdown_reference


def pain_index_reference(equity_curve: Sequence[float | None]) -> float | None:
    """
    Naive pain index over a Python list.

    The mean absolute drawdown of the non-null equity, recomputed from scratch as the oracle for
    :func:`pomata.metrics.pain_index` by averaging the magnitudes of the independent :func:`drawdown_reference`.
    ``None`` equities are skipped; a ``nan`` anywhere poisons the result to ``nan``; with no observations the result is
    ``None``.
    """
    observations = [value for value in equity_curve if value is not None]
    if any(math.isnan(value) for value in observations):
        return math.nan
    if not observations:
        return None
    declines = drawdown_reference(observations)
    return sum(abs(value) for value in declines if value is not None) / len(declines)
