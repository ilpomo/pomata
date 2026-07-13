"""
Naive reference oracle for ``pomata.metrics.risk_of_ruin``.
"""

import math
from collections.abc import Sequence

from tests_new.metrics.oracles.win_rate import win_rate_reference


def risk_of_ruin_reference(returns: Sequence[float | None]) -> float | None:
    """
    Naive risk of ruin over a Python list.

    The symmetric gambler's-ruin probability ``min(((1 - p) / p) ** n, 1)`` from the win rate ``p`` and the count ``n``
    of non-null returns (the capital cushion in unit bets), recomputed from scratch as the oracle for
    :func:`pomata.metrics.risk_of_ruin` by composing the independent :func:`win_rate_reference`. ``None`` returns are
    skipped; a ``nan`` anywhere poisons the result to ``nan``; with no decisive returns (the win rate undefined) the
    result is ``None``. A win rate ``p <= 0.5`` gives a ratio ``>= 1``, so the probability is clamped to ``1`` (ruin is
    certain without an edge); ``p == 0`` (all losses) is ``1`` and ``p == 1`` (all wins) is ``0``.
    """
    probability = win_rate_reference(returns)
    if probability is None:
        return None
    if math.isnan(probability):
        return math.nan
    observations = sum(1 for value in returns if value is not None)
    if probability == 0.0:
        return 1.0
    return min(((1.0 - probability) / probability) ** observations, 1.0)
