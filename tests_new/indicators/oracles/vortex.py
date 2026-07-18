"""
Naive reference oracle for ``pomata.indicators.vortex``.
"""

import math
from collections.abc import Sequence

from tests_new.indicators.oracles.true_range import reference_true_range


def _lagged_abs_diff(current: Sequence[float | None], previous: Sequence[float | None]) -> list[float | None]:
    """
    ``|current_t - previous_{t-1}|`` over aligned lists: ``None`` at row 0 (no previous) and wherever a side is
    ``None``, ``nan`` where a side is ``nan``.
    """
    result: list[float | None] = [None]
    for index in range(1, len(current)):
        a = current[index]
        b = previous[index - 1]
        if a is None or b is None:
            result.append(None)
        elif math.isnan(a) or math.isnan(b):
            result.append(math.nan)
        else:
            result.append(abs(a - b))
    return result


def _rolling_sum(values: Sequence[float | None], window: int) -> list[float | None]:
    """
    Matches Polars ``rolling_sum`` (``min_samples = window``): ``None`` for the warm-up and for any window holding a
    ``None``, ``nan`` for a window holding a ``nan``, else the window sum.
    """
    result: list[float | None] = []
    for index in range(len(values)):
        if index + 1 < window:
            result.append(None)
            continue
        window_values = values[index + 1 - window : index + 1]
        if any(value is None for value in window_values):
            result.append(None)
        elif any(isinstance(value, float) and math.isnan(value) for value in window_values):
            result.append(math.nan)
        else:
            result.append(sum(value for value in window_values if value is not None))
    return result


def _divide(numerator: float | None, denominator: float | None) -> float | None:
    """
    The ratio under Polars / IEEE rules: ``None`` if either side is ``None``, else ``nan`` if either is or it is 0/0.
    """
    if numerator is None or denominator is None:
        return None
    if math.isnan(numerator) or math.isnan(denominator):
        return math.nan
    if denominator == 0.0:
        return math.nan if numerator == 0.0 else math.copysign(math.inf, numerator)
    return numerator / denominator


def reference_vortex(
    high: Sequence[float | None],
    low: Sequence[float | None],
    close: Sequence[float | None],
    window: int,
) -> dict[str, list[float | None]]:
    """
    Naive Vortex Indicator over aligned Python lists.

    The summed positive (``|high_t - low_{t-1}|``) and negative (``|low_t - high_{t-1}|``) vortex movements over the
    summed :func:`reference_true_range`, recomputed from scratch as the oracle for :func:`pomata.indicators.vortex`. The
    rolling sums match Polars (the warm-up and any window holding a ``None`` yields ``None``, any holding a ``nan``
    yields ``nan``), and each ratio follows IEEE (``0 / 0`` is ``nan``).

    Args:
        high: High-price series (may contain ``None`` and ``float('nan')``).
        low: Low-price series (may contain ``None`` and ``float('nan')``).
        close: Close-price series (may contain ``None`` and ``float('nan')``).
        window: Number of observations in the moving window. Must be ``>= 1``.

    Returns:
        A dict with two lists the same length as the inputs — ``"plus"`` and ``"minus"`` — matching the fields of the
        indicator's struct. The first ``window`` entries of each are ``None`` (warm-up).

    Raises:
        ValueError: If ``window < 1``.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    if not high:
        return {"plus": [], "minus": []}
    range_sum = _rolling_sum(reference_true_range(high, low, close), window)
    plus_sum = _rolling_sum(_lagged_abs_diff(high, low), window)
    minus_sum = _rolling_sum(_lagged_abs_diff(low, high), window)
    plus = [_divide(movement, denominator) for movement, denominator in zip(plus_sum, range_sum, strict=True)]
    minus = [_divide(movement, denominator) for movement, denominator in zip(minus_sum, range_sum, strict=True)]
    return {"plus": plus, "minus": minus}
