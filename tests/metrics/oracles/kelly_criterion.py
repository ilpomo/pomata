"""
Naive reference oracle for ``pomata.metrics.kelly_criterion``.
"""

import math
from collections.abc import Sequence

from tests.metrics.oracles.payoff_ratio import payoff_ratio_reference
from tests.metrics.oracles.win_rate import win_rate_reference


def kelly_criterion_reference(returns: Sequence[float | None]) -> float | None:
    """
    Naive Kelly criterion over a Python list.

    The growth-optimal fraction ``p - (1 - p) / W`` from the win rate ``p`` and the payoff ratio ``W``, recomputed from
    scratch as the oracle for :func:`pomata.metrics.kelly_criterion` by composing the independent
    :func:`win_rate_reference` and :func:`payoff_ratio_reference`. ``None`` returns are skipped; a ``nan`` anywhere
    poisons the result to ``nan``; with the win rate or payoff ratio undefined the result is ``None``.
    """
    probability = win_rate_reference(returns)
    payoff = payoff_ratio_reference(returns)
    if probability is None or payoff is None:
        return None
    if math.isnan(probability) or math.isnan(payoff):
        return math.nan
    return probability - (1.0 - probability) / payoff
