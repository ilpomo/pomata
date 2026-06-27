"""
Naive reference oracle for ``pomata.metrics.beta``.
"""

import math
from collections.abc import Sequence


def beta_reference(returns: Sequence[float | None], benchmark: Sequence[float | None]) -> float | None:
    """
    Naive beta (regression slope) over two Python lists.

    The population covariance of the portfolio and benchmark returns over the benchmark variance --
    ``cov(r, b) / var(b)`` -- recomputed from scratch as the oracle for :func:`pomata.metrics.beta`. The series are
    pairwise-complete: a pair contributes only where both legs are present; with fewer than two such pairs the result is
    ``None`` (taking precedence over poisoning); otherwise a ``nan`` in either leg of a retained pair poisons the result
    to ``nan``. A constant (zero-variance) benchmark gives ``0 / 0`` and is reported as ``nan``, detected exactly via
    ``min == max`` rather than the two-pass variance (whose float residual is not reliably zero for every constant).
    """
    pairs = [(x, y) for x, y in zip(returns, benchmark, strict=True) if x is not None and y is not None]
    if len(pairs) < 2:
        return None
    if any(math.isnan(x) or math.isnan(y) for x, y in pairs):
        return math.nan
    benchmark_values = [y for _, y in pairs]
    if max(benchmark_values) == min(benchmark_values):
        return math.nan
    count = len(pairs)
    mean_returns = sum(x for x, _ in pairs) / count
    mean_benchmark = sum(y for _, y in pairs) / count
    covariance = sum((x - mean_returns) * (y - mean_benchmark) for x, y in pairs) / count
    variance = sum((y - mean_benchmark) ** 2 for _, y in pairs) / count
    if variance == 0.0:
        return math.nan if covariance == 0.0 else math.copysign(math.inf, covariance)
    return covariance / variance
