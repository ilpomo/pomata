"""
Naive reference oracle for ``pomata.indicators.linear_regression_slope``.
"""

import math
from collections.abc import Sequence


def linear_regression_slope_reference(
    expr: Sequence[float | None],
    window: int,
) -> list[float | None]:
    """
    Naive rolling least-squares slope over a Python list.

    For each window the slope is computed from scratch in the mean-deviation form
    ``sum((i - i_mean) * (y - y_mean)) / sum((i - i_mean) ** 2)`` over the in-window positions ``i = 0 .. window - 1``,
    an execution independent of the implementation's summation form. The oracle for
    :func:`pomata.indicators.linear_regression_slope`.

    Args:
        expr: The input series (may contain ``None`` and ``float('nan')``).
        window: Number of observations in the regression window. Must be ``>= 2``.

    Returns:
        A list the same length as ``expr``: the rolling slope, ``None`` through the ``window - 1`` warm-up rows and for
        any window containing a ``None``, ``nan`` for a window containing a ``nan``.

    Raises:
        ValueError: If ``window < 2``.
    """
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")
    positions = list(range(window))
    position_mean = sum(positions) / window
    denominator = sum((position - position_mean) ** 2 for position in positions)
    results: list[float | None] = []
    for index in range(len(expr)):
        if index < window - 1:
            results.append(None)
            continue
        window_values = expr[index - window + 1 : index + 1]
        if any(value is None for value in window_values):
            results.append(None)
        elif any(isinstance(value, float) and math.isnan(value) for value in window_values):
            results.append(math.nan)
        else:
            value_mean = sum(value for value in window_values if value is not None) / window
            numerator = sum(
                (position - position_mean) * (value - value_mean)
                for position, value in zip(positions, window_values, strict=True)
                if value is not None
            )
            results.append(numerator / denominator)
    return results
