"""
Naive reference oracle for ``pomata.indicators.ichimoku``.
"""

import math
from collections.abc import Sequence


def _midline(top: float | None, bottom: float | None) -> float | None:
    """
    The mean of two values under Polars arithmetic: ``None`` if either is ``None``; otherwise ``nan`` if either is
    ``nan``; otherwise their midpoint.
    """
    if top is None or bottom is None:
        return None
    if math.isnan(top) or math.isnan(bottom):
        return math.nan
    return (top + bottom) / 2.0


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


def _midpoint(
    high: Sequence[float | None],
    low: Sequence[float | None],
    window: int,
    index: int,
) -> float | None:
    """
    The rolling high-low midpoint ending at ``index``: ``None`` through warm-up, else ``(max high + min low) / 2``.
    """
    if index + 1 < window:
        return None
    top = _window_extreme(high[index + 1 - window : index + 1], take_max=True)
    bottom = _window_extreme(low[index + 1 - window : index + 1], take_max=False)
    return _midline(top, bottom)


def ichimoku_reference(
    high: Sequence[float | None],
    low: Sequence[float | None],
    window_tenkan: int,
    window_kijun: int,
    window_senkou: int,
) -> dict[str, list[float | None]]:
    """
    Naive Ichimoku Kinkō Hyō over Python lists, recomputed from scratch as the oracle for
    :func:`pomata.indicators.ichimoku`.

    Each line is a rolling high-low midpoint: ``tenkan`` and ``kijun`` over their windows, ``senkou_b`` over the long
    window, and ``senkou_a`` the midpoint of the first two. Every line is aligned to its computation row -- no
    displacement -- so the reference, like the implementation, never reads a future bar.

    Args:
        high: High-price series (may contain ``None`` and ``float('nan')``).
        low: Low-price series (may contain ``None`` and ``float('nan')``).
        window_tenkan: Conversion-line window. Must be ``>= 1``.
        window_kijun: Base-line window. Must be ``>= 1`` and ``>= window_tenkan``.
        window_senkou: Leading-span-B window. Must be ``>= 1`` and ``>= window_kijun``.

    Returns:
        A dict with four lists the same length as the inputs — ``"tenkan"``, ``"kijun"``, ``"senkou_a"``,
        ``"senkou_b"`` — matching the fields of the indicator's struct.

    Raises:
        ValueError: If any window is ``< 1`` or the windows are not ordered ``window_tenkan <= window_kijun <=
            window_senkou``.

    Note:
        ``None`` / ``nan`` propagate per window through the rolling extremes, and ``senkou_a`` inherits from both
        ``tenkan`` and ``kijun``: ``None`` if either is ``None``, else ``nan`` if either is ``nan``.
    """
    if window_tenkan < 1:
        raise ValueError(f"window_tenkan must be >= 1, got {window_tenkan}")
    if window_kijun < 1:
        raise ValueError(f"window_kijun must be >= 1, got {window_kijun}")
    if window_senkou < 1:
        raise ValueError(f"window_senkou must be >= 1, got {window_senkou}")
    if not window_tenkan <= window_kijun <= window_senkou:
        raise ValueError(
            f"windows must be ordered window_tenkan <= window_kijun <= window_senkou, "
            f"got {window_tenkan}, {window_kijun}, and {window_senkou}"
        )

    tenkan: list[float | None] = []
    kijun: list[float | None] = []
    senkou_a: list[float | None] = []
    senkou_b: list[float | None] = []
    for index in range(len(high)):
        tenkan_value = _midpoint(high, low, window_tenkan, index)
        kijun_value = _midpoint(high, low, window_kijun, index)
        tenkan.append(tenkan_value)
        kijun.append(kijun_value)
        senkou_a.append(_midline(tenkan_value, kijun_value))
        senkou_b.append(_midpoint(high, low, window_senkou, index))
    return {"tenkan": tenkan, "kijun": kijun, "senkou_a": senkou_a, "senkou_b": senkou_b}
