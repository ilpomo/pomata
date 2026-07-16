"""
Naive reference oracle for ``pomata.indicators.parabolic_sar``.
"""

import math
from collections.abc import Sequence


def parabolic_sar_reference(
    high: Sequence[float | None],
    low: Sequence[float | None],
    acceleration: float = 0.02,
    maximum: float = 0.20,
) -> list[float | None]:
    """
    Naive Parabolic SAR over aligned Python lists.

    A sequential re-implementation of Wilder's stop-and-reverse recurrence (seed from the first two valid bars,
    trail the stop, clamp to the two prior extremes, accelerate on new extremes, reverse on a crossing), recomputed as
    the oracle for :func:`pomata.indicators.parabolic_sar`. This is a one-shape path-dependent recurrence, so the
    transcription necessarily mirrors the production kernel's structure; the genuine correctness evidence is the
    spec's hand-derived golden master and pins, not the oracle agreement (see the documentation's Correctness page).

    Args:
        high: High-price series (may contain ``None`` and ``float('nan')``).
        low: Low-price series (may contain ``None`` and ``float('nan')``); same length as ``high``.
        acceleration: Starting acceleration factor and its per-extreme increment. A fraction in ``(0, 1]``, and never
            above ``maximum``.
        maximum: Cap on the acceleration factor. A fraction in ``(0, 1]``, and at least ``acceleration``.

    Returns:
        A list the same length as the inputs: the Parabolic SAR, ``None`` through the seed and on skipped rows.

    Raises:
        ValueError: If ``acceleration`` or ``maximum`` is not a finite number ``> 0``, if either is ``> 1``, or if
            ``acceleration > maximum``.
    """
    if not math.isfinite(acceleration) or acceleration <= 0.0:
        raise ValueError(f"acceleration must be a finite number > 0, got {acceleration}")
    if not math.isfinite(maximum) or maximum <= 0.0:
        raise ValueError(f"maximum must be a finite number > 0, got {maximum}")
    if acceleration > 1.0:
        raise ValueError(f"acceleration must be <= 1, got {acceleration}")
    if maximum > 1.0:
        raise ValueError(f"maximum must be <= 1, got {maximum}")
    if acceleration > maximum:
        raise ValueError(f"acceleration must be <= maximum, got acceleration={acceleration}, maximum={maximum}")
    result: list[float | None] = [None] * len(high)
    prior_highs: list[float] = []
    prior_lows: list[float] = []
    rising = True
    stop = 0.0
    extreme_point = 0.0
    step = acceleration
    started = False
    for index in range(len(high)):
        bar_high = high[index]
        bar_low = low[index]
        if bar_high is None or bar_low is None:
            continue
        if math.isnan(bar_high) or math.isnan(bar_low):
            result[index] = math.nan
            continue
        if not started:
            prior_highs.append(bar_high)
            prior_lows.append(bar_low)
            if len(prior_highs) < 2:
                continue
            rising = (prior_highs[1] - prior_highs[0]) >= (prior_lows[0] - prior_lows[1])
            step = acceleration
            stop = prior_lows[0] if rising else prior_highs[0]
            extreme_point = prior_highs[1] if rising else prior_lows[1]
            result[index] = stop
            started = True
            continue
        stop = stop + step * (extreme_point - stop)
        if rising:
            stop = min(stop, prior_lows[-1], prior_lows[-2])
            if bar_high > extreme_point:
                extreme_point = bar_high
                step = min(step + acceleration, maximum)
            if bar_low <= stop:
                rising = False
                stop = extreme_point
                extreme_point = bar_low
                step = acceleration
        else:
            stop = max(stop, prior_highs[-1], prior_highs[-2])
            if bar_low < extreme_point:
                extreme_point = bar_low
                step = min(step + acceleration, maximum)
            if bar_high >= stop:
                rising = True
                stop = extreme_point
                extreme_point = bar_high
                step = acceleration
        result[index] = stop
        prior_highs = [*prior_highs[-1:], bar_high]
        prior_lows = [*prior_lows[-1:], bar_low]
    return result
