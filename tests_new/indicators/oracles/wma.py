"""
Naive reference oracle for ``pomata.indicators.wma``.
"""

import math
from collections.abc import Sequence


def reference_wma(
    expr: Sequence[float | None],
    window: int,
) -> list[float | None]:
    """
    Naive Weighted (Linear) Moving Average over a Python list.

    The linearly-weighted mean (weights ``1 .. window`` normalized by their sum), recomputed as the oracle for
    :func:`pomata.indicators.wma`. Its only subtlety is the same windowed missing-data contract as the simple mean,
    detailed below.

    Args:
        expr: Input series, the observations to average (may contain ``None`` and ``float('nan')``).
        window: Number of observations in the moving window. Must be ``>= 1``.

    Returns:
        A list the same length as ``expr``. The first ``window - 1`` entries are ``None`` (warm-up); thereafter the
        linearly-weighted mean of the trailing window.

    Raises:
        ValueError: If ``window < 1``.

    Note:
        Edge-case behavior:

        - **Null** — a window that contains a ``None`` yields ``None`` (a missing observation leaves the window short);
          ``None`` takes precedence over ``nan``.
        - **NaN** — a window that is free of ``None`` but contains a ``nan`` yields ``nan`` (the ``nan`` contaminates
          the weighted sum). Because the mean reads only the current window, a ``None`` or ``nan`` contaminates only the
          positions whose window spans it and never latches onto the rest of the series.
        - **window == 1** — the single weight normalizes to one, so the WMA reproduces the input.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")

    window_weights = list(range(1, window + 1))
    weight_total = sum(window_weights)
    results: list[float | None] = []
    for index in range(len(expr)):
        if index + 1 < window:
            results.append(None)
            continue
        window_values = expr[index + 1 - window : index + 1]
        if any(value is None for value in window_values):
            results.append(None)
            continue
        window_floats = [value for value in window_values if value is not None]  # no nulls survive the guard above
        if any(math.isnan(value) for value in window_floats):
            results.append(math.nan)
        else:
            weighted_sum = sum(weight * value for weight, value in zip(window_weights, window_floats, strict=True))
            results.append(weighted_sum / weight_total)
    return results
