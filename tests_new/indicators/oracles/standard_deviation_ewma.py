"""
Naive reference oracle for ``pomata.indicators.standard_deviation_ewma``.
"""

import math
from collections.abc import Sequence

from tests_new.indicators.oracles.variance_ewma import reference_variance_ewma


def reference_standard_deviation_ewma(
    expr: Sequence[float | None],
    window: int,
    *,
    adjust: bool = False,
    bias: bool = True,
) -> list[float | None]:
    """
    Naive Exponentially-Weighted Standard Deviation over a Python list.

    The square root of :func:`reference_variance_ewma`, recomputed as the oracle for
    :func:`pomata.indicators.standard_deviation_ewma`. The variance is itself a direct two-pass evaluation of the
    published weighted-variance definition (explicit per-observation weights, then a weighted mean and weighted sum of
    squared deviations), so this oracle is independent of the engine's internal running recurrence.

    Args:
        expr: The input series (may contain ``None`` and ``float('nan')``).
        window: Span of the exponential weighting, mapped to ``alpha = 2 / (window + 1)``. Must be ``>= 2``.
        adjust: When ``False`` (default) use the recursive (blend-fraction) weighting; when ``True`` the finite-window
            bias-corrected weighting.
        bias: When ``True`` (default) the population standard deviation; when ``False`` the unbiased sample one.

    Returns:
        A list the same length as ``expr``: the square root of the exponentially-weighted variance, ``None`` until
        ``window`` non-null observations have been seen and at each ``None`` row, ``nan`` once a ``nan`` has entered the
        weighted sums.

    Raises:
        ValueError: If ``window < 2``.
    """
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")
    variance = reference_variance_ewma(expr, window, adjust=adjust, bias=bias)
    results: list[float | None] = []
    for value in variance:
        if value is None:
            results.append(None)
        elif math.isnan(value) or value < 0.0:
            results.append(math.nan)
        else:
            results.append(math.sqrt(value))
    return results
