"""
Naive reference oracle for ``pomata.metrics.recovery_ratio``.
"""

import math
from collections.abc import Sequence

from tests_new.metrics.oracles.max_drawdown import max_drawdown_reference
from tests_new.metrics.oracles.total_return import total_return_reference


def recovery_ratio_reference(equity_curve: Sequence[float | None]) -> float | None:
    """
    Naive recovery factor over a Python list.

    The total return over the magnitude of the maximum drawdown, recomputed from scratch as the oracle
    for :func:`pomata.metrics.recovery_ratio` by composing the independent :func:`total_return_reference` and
    :func:`max_drawdown_reference`. Only the drawdown denominator is taken in magnitude; the total-return numerator
    keeps its sign, so a losing curve reports a negative factor. ``None`` equities are skipped; a ``nan`` anywhere
    poisons it to ``nan``; with no defined observations the result is ``None``. A drawdown-free curve gives ``+/-inf``
    with the sign of the total return (or ``nan`` when the total return is also zero), matching the implementation's
    division.
    """
    growth = total_return_reference(equity_curve)
    drawdown_trough = max_drawdown_reference(equity_curve)
    if growth is None or drawdown_trough is None:
        return None
    if math.isnan(growth) or math.isnan(drawdown_trough):
        return math.nan
    denominator = abs(drawdown_trough)
    if denominator == 0.0:
        return math.nan if growth == 0.0 else math.copysign(math.inf, growth)
    return growth / denominator
