"""
Naive reference oracle for ``pomata.indicators.linear_regression_intercept``.
"""

import math
from collections.abc import Sequence

from tests.indicators.oracles.linear_regression_slope import linear_regression_slope_reference
from tests.indicators.oracles.sma import sma_reference


def linear_regression_intercept_reference(
    expr: Sequence[float | None],
    window: int,
) -> list[float | None]:
    """
    Naive Linear Regression intercept over a Python list.

    The window mean (:func:`sma_reference`) minus ``slope * (window - 1) / 2`` (the fitted line at the oldest bar),
    recomputed as the oracle for :func:`pomata.indicators.linear_regression_intercept`.

    Args:
        expr: The input series (may contain ``None`` and ``float('nan')``).
        window: Number of observations in the regression window. Must be ``>= 2``.

    Returns:
        A list the same length as ``expr``: the fitted intercept, ``None`` through the ``window - 1`` warm-up rows.

    Raises:
        ValueError: If ``window < 2``.
    """
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")
    mean = sma_reference(expr, window)
    slope = linear_regression_slope_reference(expr, window)
    results: list[float | None] = []
    for mean_value, slope_value in zip(mean, slope, strict=True):
        if mean_value is None or slope_value is None:
            results.append(None)
        elif math.isnan(mean_value) or math.isnan(slope_value):
            results.append(math.nan)
        else:
            results.append(mean_value - slope_value * (window - 1) / 2.0)
    return results
