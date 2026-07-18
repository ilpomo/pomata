"""
Naive reference oracle for ``pomata.indicators.donchian_channels``.
"""

import math
from collections.abc import Sequence


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


def _midline(top: float | None, bottom: float | None) -> float | None:
    """
    The mean of the two bands under Polars arithmetic: ``None`` if either band is ``None``; otherwise ``nan`` if either
    is ``nan``; otherwise their midpoint.
    """
    if top is None or bottom is None:
        return None
    if math.isnan(top) or math.isnan(bottom):
        return math.nan
    return (top + bottom) / 2


def reference_donchian_channels(
    high: Sequence[float | None],
    low: Sequence[float | None],
    window: int,
) -> dict[str, list[float | None]]:
    """
    Naive Donchian Channels over Python lists.

    The window's highest ``high`` (upper band) and lowest ``low`` (lower band), with their mean (middle band),
    recomputed from scratch as the oracle for :func:`pomata.indicators.donchian_channels`. Each band reads its own
    input's window independently, so missing values propagate per band rather than bar-wide.

    Args:
        high: High-price series (may contain ``None`` and ``float('nan')``).
        low: Low-price series (may contain ``None`` and ``float('nan')``).
        window: Number of observations in the moving window. Must be ``>= 1``.

    Returns:
        A dict with three lists the same length as the inputs — ``"lower"``, ``"middle"``, and ``"upper"`` — matching
        the fields of the indicator's struct. The first ``window - 1`` entries of each are ``None`` (warm-up).

    Raises:
        ValueError: If ``window < 1``.

    Note:
        ``None`` / ``nan`` propagate per band (``upper`` from the ``high`` window, ``lower`` from the ``low`` window),
        and ``middle`` inherits from both: ``None`` if either band is ``None``, else ``nan`` if either is ``nan``
        (``None`` always taking precedence over ``nan``).
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")

    lower: list[float | None] = []
    middle: list[float | None] = []
    upper: list[float | None] = []
    for index in range(len(high)):
        if index + 1 < window:
            lower.append(None)
            middle.append(None)
            upper.append(None)
            continue
        top = _window_extreme(high[index + 1 - window : index + 1], take_max=True)
        bottom = _window_extreme(low[index + 1 - window : index + 1], take_max=False)
        upper.append(top)
        lower.append(bottom)
        middle.append(_midline(top, bottom))
    return {"lower": lower, "middle": middle, "upper": upper}
