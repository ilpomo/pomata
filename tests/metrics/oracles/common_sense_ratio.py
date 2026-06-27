"""
Naive reference oracle for ``pomata.metrics.common_sense_ratio``.
"""

import math
from collections.abc import Sequence

from tests.metrics.oracles.profit_ratio import profit_ratio_reference
from tests.metrics.oracles.tail_ratio import tail_ratio_reference


def common_sense_ratio_reference(returns: Sequence[float | None]) -> float | None:
    """
    Naive common sense ratio over a Python list.

    The product of the profit factor and the tail ratio, recomputed from scratch as the oracle for
    :func:`pomata.metrics.common_sense_ratio` by composing the independent :func:`profit_ratio_reference` and
    :func:`tail_ratio_reference`. ``None`` returns are skipped; a ``nan`` anywhere poisons the result to ``nan``; with
    no observations the result is ``None``. It inherits the ``+inf`` / ``nan`` degeneracies of its two factors.
    """
    factor = profit_ratio_reference(returns)
    tail = tail_ratio_reference(returns)
    if factor is None or tail is None:
        return None
    if math.isnan(factor) or math.isnan(tail):
        return math.nan
    return factor * tail
