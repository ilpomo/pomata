"""
Naive reference oracle for ``pomata.indicators.dema``.
"""

import math
from collections.abc import Sequence

from tests.indicators.oracles.ema import reference_ema


def reference_dema(
    expr: Sequence[float | None],
    window: int,
    *,
    adjust: bool = False,
) -> list[float | None]:
    """
    Naive Double Exponential Moving Average over a Python list.

    Mulloy's double EMA, ``2 * EMA(x) - EMA(EMA(x))`` with both passes at the same ``window``, recomputed as the oracle
    for :func:`pomata.indicators.dema`. Each pass is the EMA oracle (:func:`reference_ema`), SMA-seeded for the
    unadjusted form, so chaining two of them compounds the warm-up to ``2 * (window - 1)``; the null / NaN behavior is
    detailed below.

    Args:
        expr: The input observations (may contain ``None`` and ``float('nan')``).
        window: Number of observations in the moving window, mapped to the smoothing factor
            ``alpha = 2 / (window + 1)``. Must be ``>= 1``.
        adjust: Whether the underlying exponential passes use the bias-corrected (adjusted) weighting; forwarded
            unchanged to both passes, matching the implementation's single ``adjust`` flag.

    Returns:
        A list of the same length as ``expr``. The first ``2 * (window - 1)`` entries are ``None`` (warm-up, clamped to
        the length); thereafter ``2 * EMA(x) - EMA(EMA(x))``.

    Raises:
        ValueError: If ``window < 1``.

    Note:
        Edge-case behavior:

        - **Null** — a leading ``None`` run stays ``None`` until the first non-null seed; an interior ``None`` yields
          ``None`` at that position while the decay continues across the gap. The warm-up mask takes precedence over the
          ``nan`` latch, so a ``nan`` arriving inside the warm-up still emits ``None`` until the window is filled.
        - **NaN** — a ``nan`` contaminates the recursive state and yields ``nan`` for every subsequent non-null
          position.
        - **window == 1** — each EMA reduces to the identity, so the result reproduces the input.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")

    if window == 1:
        return list(expr)

    ema_once = reference_ema(expr, window, adjust=adjust)
    ema_twice = reference_ema(ema_once, window, adjust=adjust)
    results: list[float | None] = []
    for value_once, value_twice in zip(ema_once, ema_twice, strict=True):
        if value_once is None or value_twice is None:
            results.append(None)
        elif math.isnan(value_once) or (math.isnan(value_twice)):
            results.append(math.nan)
        else:
            results.append(2.0 * value_once - value_twice)
    return results
