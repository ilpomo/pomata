"""
Naive reference oracle for ``pomata.metrics.capture_ratio``.
"""

import math
from collections.abc import Sequence

from tests_new.metrics.oracles.capture_downside_ratio import capture_downside_ratio_reference
from tests_new.metrics.oracles.capture_upside_ratio import capture_upside_ratio_reference


def capture_ratio_reference(
    returns: Sequence[float | None], benchmark: Sequence[float | None], periods_per_year: int
) -> float | None:
    """
    Naive capture ratio over two Python lists.

    The :func:`capture_upside_ratio_reference` divided by the :func:`capture_downside_ratio_reference`, recomputed from
    scratch as the oracle for :func:`pomata.metrics.capture_ratio` by composing the two independent capture references.
    Following the implementation's IEEE division semantics: an undefined upside or downside capture (``None``)
    propagates to ``None``; a ``nan`` in either propagates to ``nan``; a (possibly signed) zero downside capture gives
    ``+/-inf`` whose sign follows the signs of both operands (or ``nan`` when the upside capture is also zero).
    """
    upside = capture_upside_ratio_reference(returns, benchmark, periods_per_year)
    downside = capture_downside_ratio_reference(returns, benchmark, periods_per_year)
    if upside is None or downside is None:
        return None
    if math.isnan(upside) or math.isnan(downside):
        return math.nan
    if downside == 0.0:
        if upside == 0.0:
            return math.nan
        return math.copysign(math.inf, math.copysign(1.0, upside) * math.copysign(1.0, downside))
    return upside / downside
