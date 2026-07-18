"""
Naive reference oracle for ``pomata.indicators.tema``.
"""

import math
from collections.abc import Sequence

from tests_new.indicators.oracles.ema import reference_ema


def reference_tema(
    expr: Sequence[float | None],
    window: int,
    *,
    adjust: bool = False,
) -> list[float | None]:
    """
    Naive Triple Exponential Moving Average over a Python list.

    Mulloy's triple EMA, ``3 * EMA1 - 3 * EMA2 + EMA3`` over three same-``window`` passes, recomputed as the oracle for
    :func:`pomata.indicators.tema`. Each pass is the EMA oracle (:func:`reference_ema`), SMA-seeded for the unadjusted
    form, so chaining three compounds the warm-up to ``3 * (window - 1)``; the null / NaN behavior is detailed below.

    Args:
        expr: Input series, the observations to smooth (may contain ``None`` and ``float('nan')``).
        window: Number of observations in the moving window, mapped to the smoothing factor
            ``alpha = 2 / (window + 1)``. Must be ``>= 1``.
        adjust: When ``False`` (default) use the recursive technical-analysis form; when ``True`` use the finite-window
            unbiased weighting that divides by the decaying sum of weights at each step. The flag is forwarded unchanged
            to all three passes, matching the implementation's single ``adjust`` flag.

    Returns:
        A list the same length as ``expr``. The first ``3 * (window - 1)`` entries are ``None`` (warm-up, clamped to the
        length); thereafter the triple-EMA combination ``3 * EMA1 - 3 * EMA2 + EMA3``.

    Raises:
        ValueError: If ``window < 1``.

    Note:
        Edge-case behavior:

        - **Null** — a leading ``None`` run stays ``None`` until the first non-null seed; an interior ``None`` yields
          ``None`` at that position while the decay continues across the gap.
        - **NaN** — a ``nan`` contaminates the recursive state and yields ``nan`` for every subsequent non-null
          position.
        - **window == 1** — each EMA reduces to the identity, so the result reproduces the input.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")

    if window == 1:
        return list(expr)

    ema_first = reference_ema(expr, window, adjust=adjust)
    ema_second = reference_ema(ema_first, window, adjust=adjust)
    ema_third = reference_ema(ema_second, window, adjust=adjust)

    results: list[float | None] = []
    for value_first, value_second, value_third in zip(ema_first, ema_second, ema_third, strict=True):
        if value_first is None or value_second is None or value_third is None:
            results.append(None)
        elif math.isnan(value_first) or math.isnan(value_second) or math.isnan(value_third):
            results.append(math.nan)
        else:
            results.append(3.0 * value_first - 3.0 * value_second + value_third)
    return results
