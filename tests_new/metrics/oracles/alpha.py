"""
Naive reference oracle for ``pomata.metrics.alpha``.
"""

import math
from collections.abc import Sequence

from tests_new.metrics.oracles.beta import beta_reference


def alpha_reference(
    returns: Sequence[float | None],
    benchmark: Sequence[float | None],
    periods_per_year: int,
    risk_free_rate: float,
) -> float | None:
    """
    Naive annualized Jensen's alpha over two Python lists.

    The per-period mean of ``(r - rf) - beta * (b - rf)`` compounded by ``(1 + .) ** P - 1``, where ``beta`` is the
    independent :func:`beta_reference` slope and the per-period risk-free rate is
    ``(1 + risk_free_rate) ** (1 / P) - 1`` -- recomputed from scratch as the oracle for :func:`pomata.metrics.alpha`.
    The series are pairwise-complete: a pair
    contributes only where both legs are present; with fewer than two such pairs the result is ``None`` (taking
    precedence over poisoning); otherwise a ``nan`` in either leg of a retained pair poisons the result to ``nan``. An
    overflow of the annualizing power is reported as the IEEE infinity, matching the implementation's float64 result.
    """
    pairs = [(x, y) for x, y in zip(returns, benchmark, strict=True) if x is not None and y is not None]
    if len(pairs) < 2:
        return None
    if any(math.isnan(x) or math.isnan(y) for x, y in pairs):
        return math.nan
    slope = beta_reference(returns, benchmark)
    assert slope is not None
    rf_period = math.pow(1.0 + risk_free_rate, 1.0 / periods_per_year) - 1.0
    excess_leg = [(x - rf_period) - slope * (y - rf_period) for x, y in pairs]
    base = 1.0 + sum(excess_leg) / len(excess_leg)
    try:
        growth = math.pow(base, periods_per_year)
    except OverflowError:
        growth = math.inf if (base > 0.0 or periods_per_year % 2 == 0) else -math.inf
    return growth - 1.0
