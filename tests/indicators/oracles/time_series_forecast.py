"""
Naive reference oracle for ``pomata.indicators.time_series_forecast``.
"""

import math
from collections.abc import Sequence

from tests.indicators.oracles.linear_regression_slope import reference_linear_regression_slope
from tests.indicators.oracles.sma import reference_sma


def reference_time_series_forecast(
    expr: Sequence[float | None],
    window: int,
) -> list[float | None]:
    """
    Naive Time Series Forecast over a Python list.

    The window mean (:func:`reference_sma`) plus ``slope * (window + 1) / 2`` (the fitted line one bar beyond the
    window), recomputed as the oracle for :func:`pomata.indicators.time_series_forecast`.

    Args:
        expr: The input series (may contain ``None`` and ``float('nan')``).
        window: Number of observations in the regression window. Must be ``>= 2``.

    Returns:
        A list the same length as ``expr``: the one-step-ahead forecast, ``None`` through the ``window - 1`` warm-up
        rows.

    Raises:
        ValueError: If ``window < 2``.
    """
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")
    mean = reference_sma(expr, window)
    slope = reference_linear_regression_slope(expr, window)
    results: list[float | None] = []
    for mean_value, slope_value in zip(mean, slope, strict=True):
        if mean_value is None or slope_value is None:
            results.append(None)
        elif math.isnan(mean_value) or math.isnan(slope_value):
            results.append(math.nan)
        else:
            results.append(mean_value + slope_value * (window + 1) / 2.0)
    return results
