"""
Naive reference oracle for ``pomata.metrics.ulcer_index``.
"""

import math
from collections.abc import Sequence

from tests.metrics.oracles.drawdown import drawdown_reference


def ulcer_index_reference(equity_curve: Sequence[float | None]) -> float | None:
    """
    Naive Ulcer Index over a Python list: the root-mean-square of the running drawdown.

    Built on :func:`drawdown_reference`. ``None`` drawdowns are skipped; a ``nan`` anywhere poisons the result to
    ``nan``; with no defined drawdown (an all-null series) the result is ``None``; otherwise the quadratic mean of the
    drawdowns.
    """
    declines = [value for value in drawdown_reference(equity_curve) if value is not None]
    if any(math.isnan(value) for value in declines):
        return math.nan
    if not declines:
        return None
    return math.sqrt(sum(value * value for value in declines) / len(declines))
