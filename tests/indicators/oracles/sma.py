"""
Naive reference oracle for ``pomata.indicators.sma``.
"""

import math
from collections.abc import Sequence


def sma_reference(
    expr: Sequence[float | None],
    window: int,
) -> list[float | None]:
    """
    Naive Simple Moving Average over a Python list.

    The unweighted arithmetic mean of each trailing window of ``window`` observations, recomputed from scratch as the
    oracle for :func:`pomata.indicators.sma`. Its only subtlety is Polars' ``min_samples=window`` missing-data
    contract, detailed below.

    Args:
        expr: Input series, the observations to average (may contain ``None`` and ``float('nan')``).
        window: Number of observations in the moving window. Must be ``>= 1``.

    Returns:
        A list the same length as ``expr``. The first ``window - 1`` entries are ``None`` (warm-up); thereafter the
        unweighted arithmetic mean of the trailing window.

    Raises:
        ValueError: If ``window < 1``.

    Note:
        Edge-case behavior:

        - **Null** — a window that contains a ``None`` yields ``None`` (matching ``rolling_mean``'s default
          ``min_samples=window``, under which a missing observation leaves the window short); ``None`` takes precedence
          over ``nan``.
        - **NaN** — a window that is free of ``None`` but contains a ``nan`` yields ``nan`` (the ``nan`` contaminates
          the sum). Because the mean reads only the current window, a ``None`` or ``nan`` contaminates only the
          positions whose window spans it and never latches onto the rest of the series.
        - **window == 1** — the one-point mean is the input itself, so the SMA reproduces the input.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")

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
            results.append(sum(value for value in window_values if value is not None) / window)
    return results
