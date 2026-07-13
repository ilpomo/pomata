"""
Naive reference oracle for ``pomata.metrics.ulcer_performance_ratio``.
"""

import math
from collections.abc import Sequence

from tests_new.metrics.oracles.cagr import cagr_reference
from tests_new.metrics.oracles.ulcer_index import ulcer_index_reference


def ulcer_performance_ratio_reference(
    equity_curve: Sequence[float | None], periods_per_year: int, risk_free_rate: float
) -> float | None:
    """
    Naive ulcer performance index (Martin ratio) over a Python list.

    The excess compound annual growth rate divided by the ulcer index -- ``(CAGR - risk_free_rate) / UlcerIndex`` --
    recomputed from scratch as the oracle for :func:`pomata.metrics.ulcer_performance_ratio` by composing the
    independent :func:`cagr_reference` and :func:`ulcer_index_reference`. ``None`` equities are skipped; a ``nan``
    anywhere poisons the result to ``nan``; with no defined observations the result is ``None``. A drawdown-free
    (monotonic) curve has a zero ulcer index and gives ``+/-inf`` (or ``nan`` when the excess growth is also zero),
    matching the implementation's division.
    """
    growth = cagr_reference(equity_curve, periods_per_year)
    ulcer = ulcer_index_reference(equity_curve)
    if growth is None or ulcer is None:
        return None
    if math.isnan(growth) or math.isnan(ulcer):
        return math.nan
    excess_growth = growth - risk_free_rate
    if ulcer == 0.0:
        return math.nan if excess_growth == 0.0 else math.copysign(math.inf, excess_growth)
    return excess_growth / ulcer
