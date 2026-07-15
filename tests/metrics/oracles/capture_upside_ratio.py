"""
Naive reference oracle for ``pomata.metrics.capture_upside_ratio``.
"""

import math
from collections.abc import Sequence


def capture_upside_ratio_reference(
    returns: Sequence[float | None], benchmark: Sequence[float | None], periods_per_year: int
) -> float | None:
    """
    Naive upside capture ratio over two Python lists.

    The geometric annualized portfolio return over the geometric annualized benchmark return, computed over the periods
    where the benchmark return is positive -- ``(prod(1 + r) ** (P / n) - 1) / (prod(1 + b) ** (P / n) - 1)`` over those
    ``n`` periods -- recomputed from scratch as the oracle for :func:`pomata.metrics.capture_upside_ratio`. The series
    are pairwise-complete: a pair contributes only where both legs are present; with no complete pairs the result is
    ``None``; otherwise a ``nan`` in either leg of a retained pair poisons the result to ``nan`` (taking precedence over
    an empty up-market). With no up-market period the result is ``None``. A selected pair with a wiped-out leg
    (``1 + x <= 0`` on either side) is outside the geometric domain and yields ``nan``, matching the implementation's
    domain guard. A zero annualized benchmark gain gives ``+/-inf`` (or ``nan``), matching the implementation's
    division.
    """
    pairs = [(x, y) for x, y in zip(returns, benchmark, strict=True) if x is not None and y is not None]
    if not pairs:
        return None
    if any(math.isnan(x) or math.isnan(y) for x, y in pairs):
        return math.nan
    selected = [(x, y) for x, y in pairs if y > 0.0]
    if not selected:
        return None
    if any(1.0 + x <= 0.0 or 1.0 + y <= 0.0 for x, y in selected):
        return math.nan
    count = len(selected)
    portfolio_growth = math.prod(1.0 + x for x, _ in selected) ** (periods_per_year / count) - 1.0
    benchmark_growth = math.prod(1.0 + y for _, y in selected) ** (periods_per_year / count) - 1.0
    if benchmark_growth == 0.0:
        return math.nan if portfolio_growth == 0.0 else math.copysign(math.inf, portfolio_growth)
    return portfolio_growth / benchmark_growth
