"""
Naive reference oracle for ``pomata.indicators.hma``.
"""

import math
from collections.abc import Sequence

from tests_new.indicators.oracles.wma import reference_wma


def reference_hma(
    expr: Sequence[float | None],
    window: int,
) -> list[float | None]:
    """
    Naive Hull Moving Average over a Python list.

    Hull's moving average, ``WMA(2 * WMA(x, window / 2) - WMA(x, window), sqrt(window))``, recomputed as the oracle for
    :func:`pomata.indicators.hma` by composing :func:`reference_wma`. Its one non-obvious point is the
    round-half-**up** period reduction (``floor(x + 0.5)``, unlike Python's round-half-to-even); warm-up and
    ``null`` / ``NaN`` are inherited from the composing weighted means, detailed below.

    Args:
        expr: Input series, the observations to smooth (may contain ``None`` and ``float('nan')``).
        window: Number of observations in the moving window. Must be ``>= 2``.

    Returns:
        A list the same length as ``expr``. The first ``window + smoothing - 2`` entries are ``None`` (warm-up, where
        ``smoothing = floor(sqrt(window) + 0.5)``); thereafter the Hull Moving Average value.

    Raises:
        ValueError: If ``window < 2``. The half-period ``floor(window / 2 + 0.5)`` collapses to ``1`` at ``window == 1``
            and the Hull average degenerates there, so the smallest meaningful window is ``2``.

    Note:
        Edge-case behavior:

        - **Null** — a window containing a ``None`` yields ``None`` at that position, propagated through every
          composing :func:`reference_wma`.
        - **NaN** — a window containing a ``nan`` (and no ``None``) yields ``nan`` at that position.
    """
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")

    half_window = math.floor(window / 2 + 0.5)
    smoothing_window = math.floor(math.sqrt(window) + 0.5)
    wma_half_values = reference_wma(expr, half_window)
    wma_full_values = reference_wma(expr, window)

    raw_values: list[float | None] = []
    for value_half, value_full in zip(wma_half_values, wma_full_values, strict=True):
        if value_half is None or value_full is None:
            raw_values.append(None)
        elif math.isnan(value_half) or (math.isnan(value_full)):
            raw_values.append(math.nan)
        else:
            raw_values.append(2.0 * value_half - value_full)

    return reference_wma(raw_values, smoothing_window)
