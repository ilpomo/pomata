"""
Naive reference oracle for ``pomata.metrics.max_drawdown``.
"""

import math
from collections.abc import Sequence

from tests.metrics.oracles.drawdown import drawdown_reference


def max_drawdown_reference(equity_curve: Sequence[float | None]) -> float | None:
    """
    Naive maximum drawdown over a Python list: the minimum of the running drawdown.

    Built on :func:`drawdown_reference`. ``None`` drawdowns are skipped; a ``nan`` anywhere poisons the result to
    ``nan``; with no defined drawdown (an all-null series) the result is ``None``; otherwise the most negative
    drawdown.
    """
    declines = [value for value in drawdown_reference(equity_curve) if value is not None]
    if any(math.isnan(value) for value in declines):
        return math.nan
    if not declines:
        return None
    return min(declines)
