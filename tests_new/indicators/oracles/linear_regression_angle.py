"""
Naive reference oracle for ``pomata.indicators.linear_regression_angle``.
"""

import math
from collections.abc import Sequence

from tests_new.indicators.oracles.linear_regression_slope import reference_linear_regression_slope


def reference_linear_regression_angle(
    expr: Sequence[float | None],
    window: int,
) -> list[float | None]:
    """
    Naive Linear Regression Angle over a Python list.

    The arctangent of the rolling slope (:func:`reference_linear_regression_slope`) in degrees, recomputed as the oracle
    for :func:`pomata.indicators.linear_regression_angle`.

    Args:
        expr: The input series (may contain ``None`` and ``float('nan')``).
        window: Number of observations in the regression window. Must be ``>= 2``.

    Returns:
        A list the same length as ``expr``: the slope angle in degrees, ``None`` through the ``window - 1`` warm-up
        rows.

    Raises:
        ValueError: If ``window < 2``.
    """
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")
    slope = reference_linear_regression_slope(expr, window)
    results: list[float | None] = []
    for slope_value in slope:
        if slope_value is None:
            results.append(None)
        elif math.isnan(slope_value):
            results.append(math.nan)
        else:
            results.append(math.degrees(math.atan(slope_value)))
    return results
