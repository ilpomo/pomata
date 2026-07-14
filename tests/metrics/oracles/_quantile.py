"""
Shared empirical-quantile helper for the tail-risk oracles.

The type-7 (linear-interpolation) quantile is the de-facto Python default -- it is what ``numpy.percentile``, pandas'
``Series.quantile``, and Polars' ``Expr.quantile(interpolation="linear")`` all compute -- so recomputing it here from
the textbook formula gives the tail-risk oracles that read a quantile (value-at-risk, tail ratio) one independent
definition to share (the conditional value-at-risk oracle averages its tail directly and does not consume it).
"""

import math
from collections.abc import Sequence


def type_seven_quantile(sorted_values: Sequence[float], probability: float) -> float:
    """
    The type-7 (linear) empirical quantile of a non-empty ascending sequence.

    Uses the virtual index ``h = probability * (n - 1)`` and linearly interpolates between the bracketing order
    statistics ``x[floor(h)]`` and ``x[ceil(h)]`` -- the Hyndman & Fan type-7 estimator shared by numpy, pandas, and
    Polars' ``"linear"`` interpolation.

    Args:
        sorted_values: The observations in ascending order; must be non-empty.
        probability: The quantile probability in ``[0, 1]``.

    Returns:
        The interpolated quantile value.
    """
    count = len(sorted_values)
    if count == 1:
        return sorted_values[0]
    position = probability * (count - 1)
    lower = math.floor(position)
    upper = min(lower + 1, count - 1)
    fraction = position - lower
    return sorted_values[lower] + fraction * (sorted_values[upper] - sorted_values[lower])
