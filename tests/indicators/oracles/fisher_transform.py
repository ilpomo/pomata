"""
Naive reference oracle for ``pomata.indicators.fisher_transform``.
"""

import math
from collections.abc import Sequence


def _median(high: float | None, low: float | None) -> float | None:
    """
    The median price ``(high + low) / 2`` under Polars arithmetic: ``None`` dominates, then ``nan``, else the mean.
    """
    if high is None or low is None:
        return None
    if math.isnan(high) or math.isnan(low):
        return math.nan
    return (high + low) / 2.0


def _window_extreme(window_values: Sequence[float | None], take_max: bool) -> float | None:
    """
    The window's extreme, matching Polars' ``rolling_max`` / ``rolling_min``: ``None`` if the window holds a ``None``,
    ``nan`` if it holds a ``nan`` (no ``None``), else the maximum (``take_max``) or minimum of the values.
    """
    if any(value is None for value in window_values):
        return None
    finite = [value for value in window_values if value is not None]
    if any(math.isnan(value) for value in finite):
        return math.nan
    return max(finite) if take_max else min(finite)


def _position(price: float | None, lowest: float | None, highest: float | None) -> float | None:
    """
    The channel position ``2 * (price - min) / (max - min) - 1`` in ``[-1, 1]``: ``None`` if any term is ``None``, else
    ``nan`` if any is ``nan`` or the channel is flat (``max == min``, a ``0/0``), else the normalized value.
    """
    if price is None or lowest is None or highest is None:
        return None
    if math.isnan(price) or math.isnan(lowest) or math.isnan(highest):
        return math.nan
    if highest == lowest:
        return math.nan
    return 2.0 * (price - lowest) / (highest - lowest) - 1.0


def fisher_transform_reference(
    high: Sequence[float | None],
    low: Sequence[float | None],
    window: int,
) -> dict[str, list[float | None]]:
    """
    Naive Ehlers Fisher Transform over Python lists.

    Built bottom-up in plain Python, sharing no code with the Polars implementation it certifies: the median price's
    position in its rolling ``[min, max]`` channel is mapped to ``[-1, 1]``, smoothed by the fixed ``0.33 / 0.67``
    recursion, clamped to ``[-0.999, 0.999]``, then run through ``0.5 * ln((1 + x) / (1 - x))`` with its own ``0.5``
    recursion. The ``signal`` line is ``fisher`` lagged one bar.

    Args:
        high: High-price series (may contain ``None`` and ``float('nan')``).
        low: Low-price series (may contain ``None`` and ``float('nan')``).
        window: Number of observations in the moving window. Must be ``>= 1``.

    Returns:
        A dict with two lists the same length as the inputs — ``"fisher"`` and ``"signal"`` — matching the fields of the
        indicator's struct. The first ``window - 1`` entries of ``fisher`` are ``None`` (warm-up); ``signal`` is
        ``None`` for one further row.

    Raises:
        ValueError: If ``window < 1``.

    Note:
        Both recursions seed at ``0``. A ``None`` position (warm-up, or a window touching a ``None``) is skipped with
        the running state bridging it; a ``nan`` position (a window touching a ``nan``, or a flat ``max == min`` window)
        yields ``nan`` and likewise leaves the state untouched, so a transient gap never latches.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")

    prices = [_median(high[index], low[index]) for index in range(len(high))]
    fisher: list[float | None] = []
    smoothed = 0.0
    transform = 0.0
    for index in range(len(prices)):
        if index + 1 < window:
            fisher.append(None)
            continue
        span = prices[index + 1 - window : index + 1]
        position = _position(
            prices[index],
            _window_extreme(span, take_max=False),
            _window_extreme(span, take_max=True),
        )
        if position is None:
            fisher.append(None)
            continue
        if math.isnan(position):
            fisher.append(math.nan)
            continue
        smoothed = 0.33 * position + 0.67 * smoothed
        smoothed = max(-0.999, min(0.999, smoothed))
        transform = 0.5 * math.log((1.0 + smoothed) / (1.0 - smoothed)) + 0.5 * transform
        fisher.append(transform)
    signal: list[float | None] = [None, *fisher[:-1]] if fisher else []
    return {"fisher": fisher, "signal": signal}
