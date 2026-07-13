"""
Naive reference oracle for ``pomata.metrics.payoff_ratio``.
"""

import math
from collections.abc import Sequence


def payoff_ratio_reference(returns: Sequence[float | None]) -> float | None:
    """
    Naive payoff ratio over a Python list.

    The mean of the positive returns over the magnitude of the mean of the negative returns, recomputed from scratch as
    the oracle for :func:`pomata.metrics.payoff_ratio`. ``None`` returns are skipped; a ``nan`` anywhere poisons the
    result to ``nan``; with no winning returns or no losing returns the result is ``None`` (one side is undefined).
    """
    observations = [value for value in returns if value is not None]
    if any(math.isnan(value) for value in observations):
        return math.nan
    wins = [value for value in observations if value > 0.0]
    losses = [value for value in observations if value < 0.0]
    if not wins or not losses:
        return None
    average_win = sum(wins) / len(wins)
    average_loss = sum(losses) / len(losses)
    return average_win / -average_loss
