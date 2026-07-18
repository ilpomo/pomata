"""
Naive reference oracle for ``pomata.indicators.trima``.
"""

import math
from collections.abc import Sequence


def reference_trima(
    expr: Sequence[float | None],
    window: int,
) -> list[float | None]:
    """
    Naive Triangular Moving Average over a Python list.

    The published triangular weighting evaluated directly: each in-window observation is weighted by the symmetric
    triangle ``1, 2, ..., ceil(window / 2), ..., 2, 1`` (a single peak for an odd window) or the trapezoid
    ``1, 2, ..., window / 2, window / 2, ..., 2, 1`` (a flat two-element ridge for an even window), and the row value is
    the weighted sum over the trailing ``window`` observations divided by the total weight. The weights are constructed
    from the triangle definition itself — counting up to the center and back down — rather than as a moving average of
    a moving average, so this oracle reaches the same numbers along an independent path.

    Args:
        expr: Input series, the observations to average (may contain ``None`` and ``float('nan')``).
        window: Number of observations in the moving window. Must be ``>= 1``.

    Returns:
        A list the same length as ``expr``: the first ``window - 1`` entries are ``None`` (warm-up); thereafter the
        triangular moving average.

    Raises:
        ValueError: If ``window < 1``.

    Note:
        Edge-case behavior mirrors the simple moving average's ``min_samples=window`` contract, since the triangular
        window spans the same trailing observations:

        - **Null** — a window that contains a ``None`` yields ``None``; ``None`` takes precedence over ``nan``.
        - **NaN** — a window free of ``None`` but containing a ``nan`` yields ``nan`` (the ``nan`` contaminates the
          weighted sum); the contamination is confined to the windows that span the missing observation.
        - **window == 1** — the single weight is ``1``, so the one-point weighted mean is the input itself and the
          TRIMA reproduces the input.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")

    # Build the symmetric triangle directly: ramp up to the center, then back down. Odd windows have a single peak
    # ``ceil(window / 2)``; even windows have a two-element ridge ``window / 2``. No sma-of-sma split is used here.
    peak = (window + 1) // 2
    ascending = list(range(1, peak + 1))
    descending = list(range(peak - 1 if window % 2 else peak, 0, -1))
    weights = ascending + descending
    total_weight = sum(weights)

    results: list[float | None] = []
    for index in range(len(expr)):
        if index + 1 < window:
            results.append(None)
            continue
        window_values = expr[index + 1 - window : index + 1]
        if any(value is None for value in window_values):
            results.append(None)
        elif any(isinstance(value, float) and math.isnan(value) for value in window_values):
            results.append(math.nan)
        else:
            defined_values = [value for value in window_values if value is not None]
            weighted_sum = sum(weight * value for weight, value in zip(weights, defined_values, strict=True))
            results.append(weighted_sum / total_weight)
    return results
