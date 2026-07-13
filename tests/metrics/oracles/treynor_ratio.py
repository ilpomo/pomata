"""
Naive reference oracle for ``pomata.metrics.treynor_ratio``.
"""

import math
from collections.abc import Sequence

from tests.metrics.oracles.beta import beta_reference


def treynor_ratio_reference(
    returns: Sequence[float | None],
    benchmark: Sequence[float | None],
    periods_per_year: int,
    risk_free_rate: float,
) -> float | None:
    """
    Naive annualized Treynor ratio over two Python lists.

    The annualized arithmetic excess return ``mean(r - rf) * P`` over the :func:`beta_reference` slope, where the
    per-period risk-free rate is ``(1 + risk_free_rate) ** (1 / P) - 1`` -- recomputed from scratch as the oracle for
    :func:`pomata.metrics.treynor_ratio`. The series are pairwise-complete: a pair contributes only where both legs are
    present; with fewer than two such pairs the result is ``None`` (taking precedence over poisoning); otherwise a
    ``nan`` in either leg of a retained pair poisons the result to ``nan``. A zero beta gives ``+/-inf`` (or ``nan``
    when the excess return is also zero), matching the implementation's division.
    """
    pairs = [(x, y) for x, y in zip(returns, benchmark, strict=True) if x is not None and y is not None]
    if len(pairs) < 2:
        return None
    if any(math.isnan(x) or math.isnan(y) for x, y in pairs):
        return math.nan
    slope = beta_reference(returns, benchmark)
    assert slope is not None
    rf_period = math.pow(1.0 + risk_free_rate, 1.0 / periods_per_year) - 1.0
    annualized_excess = (sum(x - rf_period for x, _ in pairs) / len(pairs)) * periods_per_year
    if slope == 0.0:
        return math.nan if annualized_excess == 0.0 else math.copysign(math.inf, annualized_excess)
    return annualized_excess / slope
