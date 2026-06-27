"""
Naive reference oracle for ``pomata.metrics.modigliani_risk_adjusted_performance``.
"""

import math
from collections.abc import Sequence

from tests.metrics.oracles.sharpe_ratio import sharpe_ratio_reference
from tests.metrics.oracles.volatility import volatility_reference


def modigliani_risk_adjusted_performance_reference(
    returns: Sequence[float | None],
    benchmark: Sequence[float | None],
    periods_per_year: int,
    risk_free_rate: float,
) -> float | None:
    """
    Naive Modigliani risk-adjusted performance (M-squared) over two Python lists.

    The risk-free rate plus the portfolio :func:`sharpe_ratio_reference` ratio scaled by the benchmark
    :func:`volatility_reference` -- ``rf + SR * sigma_b`` -- recomputed from scratch as the oracle for
    :func:`pomata.metrics.modigliani_risk_adjusted_performance` by composing the two independent single-input
    references. The series are pairwise-complete: a pair contributes only where both legs are present; with fewer than
    two such pairs the result is ``None`` (taking precedence over poisoning); otherwise a ``nan`` in either leg of a
    retained pair poisons the result to ``nan``. A constant portfolio gives an infinite Sharpe ratio, which propagates.
    """
    pairs = [(x, y) for x, y in zip(returns, benchmark, strict=True) if x is not None and y is not None]
    if len(pairs) < 2:
        return None
    if any(math.isnan(x) or math.isnan(y) for x, y in pairs):
        return math.nan
    sharpe_ratio = sharpe_ratio_reference([x for x, _ in pairs], periods_per_year, risk_free_rate)
    benchmark_volatility = volatility_reference([y for _, y in pairs], periods_per_year)
    assert sharpe_ratio is not None
    assert benchmark_volatility is not None
    return risk_free_rate + sharpe_ratio * benchmark_volatility
