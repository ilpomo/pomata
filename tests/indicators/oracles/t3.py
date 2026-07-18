"""
Naive reference oracle for ``pomata.indicators.t3``.
"""

import math
from collections.abc import Sequence

from tests.indicators.oracles.ema import reference_ema


def reference_t3(
    expr: Sequence[float | None],
    window: int,
    *,
    volume_factor: float = 0.7,
    adjust: bool = False,
) -> list[float | None]:
    """
    Naive Tillson T3 Moving Average over a Python list.

    Tillson's T3, the coefficient combination ``c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3`` of six chained EMAs
    (coefficients in the volume factor ``v``, summing to ``1``), recomputed as the oracle for
    :func:`pomata.indicators.t3` by composing :func:`reference_ema`. Chaining six passes compounds the warm-up to
    ``6 * (window - 1)``; the null / NaN behavior is detailed below.

    Args:
        expr: Input series, the observations to smooth (may contain ``None`` and ``float('nan')``).
        window: Number of observations in the moving window, mapped to the smoothing factor
            ``alpha = 2 / (window + 1)``. Must be ``>= 1``.
        volume_factor: The Tillson volume factor ``v`` controlling smoothing versus responsiveness; the canonical
            default is ``0.7``. It is forwarded unchanged to the coefficients ``c1 .. c4``.
        adjust: Whether the underlying exponential passes use the bias-corrected (adjusted) weighting; forwarded
            unchanged to all six passes, matching the implementation's single ``adjust`` flag.

    Returns:
        A list the same length as ``expr``. The first ``6 * (window - 1)`` entries are ``None`` (warm-up, clamped to the
        length); thereafter the T3 value ``c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3``.

    Raises:
        ValueError: If ``window < 1``.

    Note:
        Edge-case behavior:

        - **Null** — a leading ``None`` run stays ``None`` until the first non-null seed; an interior ``None`` yields
          ``None`` at that position while the decay continues across the gap (it propagates through all six passes).
        - **NaN** — a ``nan`` poisons the recursive state and latches ``nan`` for every subsequent non-null position.
        - **window == 1** — each EMA reduces to the identity, so the result reproduces the input.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")

    if window == 1:
        return list(expr)

    factor = volume_factor
    c1 = -(factor**3)
    c2 = 3.0 * factor**2 + 3.0 * factor**3
    c3 = -6.0 * factor**2 - 3.0 * factor - 3.0 * factor**3
    c4 = 1.0 + 3.0 * factor + 3.0 * factor**2 + factor**3

    e1 = reference_ema(expr, window, adjust=adjust)
    e2 = reference_ema(e1, window, adjust=adjust)
    e3 = reference_ema(e2, window, adjust=adjust)
    e4 = reference_ema(e3, window, adjust=adjust)
    e5 = reference_ema(e4, window, adjust=adjust)
    e6 = reference_ema(e5, window, adjust=adjust)

    results: list[float | None] = []
    for value_e3, value_e4, value_e5, value_e6 in zip(e3, e4, e5, e6, strict=True):
        if value_e3 is None or value_e4 is None or value_e5 is None or value_e6 is None:
            results.append(None)
        elif math.isnan(value_e3) or math.isnan(value_e4) or math.isnan(value_e5) or math.isnan(value_e6):
            results.append(math.nan)
        else:
            results.append(c4 * value_e3 + c3 * value_e4 + c2 * value_e5 + c1 * value_e6)
    return results
