"""
Naive reference oracle for ``pomata.indicators.keltner_channels``.
"""

import math
from collections.abc import Sequence

from tests_new.indicators.oracles.atr import reference_atr
from tests_new.indicators.oracles.ema import reference_ema


def reference_keltner_channels(
    high: Sequence[float | None],
    low: Sequence[float | None],
    close: Sequence[float | None],
    window: int,
    window_atr: int,
    multiplier: float,
) -> dict[str, list[float | None]]:
    """
    Naive Keltner Channels over Python lists.

    The center band is the :func:`reference_ema` of ``close``; the outer bands sit ``multiplier`` :func:`reference_atr`
    average true ranges away. Recomputed from the certified leg references as the oracle for
    :func:`pomata.indicators.keltner_channels`. The midline reads only ``close`` while the outer bands also read the
    ATR, so missing values propagate per band.

    Args:
        high: High-price series (may contain ``None`` and ``float('nan')``).
        low: Low-price series (may contain ``None`` and ``float('nan')``).
        close: Close-price series (may contain ``None`` and ``float('nan')``).
        window: EMA midline window. Must be ``>= 1``.
        window_atr: ATR window. Must be ``>= 1``.
        multiplier: ATRs between the midline and each band. Must be ``> 0``.

    Returns:
        A dict with three lists the same length as the inputs — ``"lower"``, ``"middle"``, and ``"upper"`` — matching
        the fields of the indicator's struct.

    Raises:
        ValueError: If ``window < 1``, ``window_atr < 1``, or ``multiplier <= 0``.

    Note:
        ``middle`` is the EMA of ``close`` alone; an outer band is ``None`` where either the EMA or the ATR is ``None``,
        else ``nan`` where either is ``nan`` (``None`` taking precedence).
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    if window_atr < 1:
        raise ValueError(f"window_atr must be >= 1, got {window_atr}")
    if not math.isfinite(multiplier) or multiplier <= 0.0:
        raise ValueError(f"multiplier must be a finite number > 0, got {multiplier}")

    center = reference_ema(close, window)
    ranges = reference_atr(high, low, close, window_atr)
    lower: list[float | None] = []
    middle: list[float | None] = []
    upper: list[float | None] = []
    for mid, band_atr in zip(center, ranges, strict=True):
        middle.append(mid)
        if mid is None or band_atr is None:
            lower.append(None)
            upper.append(None)
        elif math.isnan(mid) or math.isnan(band_atr):
            lower.append(math.nan)
            upper.append(math.nan)
        else:
            half = multiplier * band_atr
            lower.append(mid - half)
            upper.append(mid + half)
    return {"lower": lower, "middle": middle, "upper": upper}
