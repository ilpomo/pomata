"""
Naive reference oracle for ``pomata.indicators.bollinger_bands``.
"""

import math
from collections.abc import Sequence

from tests_new.indicators.oracles.sma import reference_sma
from tests_new.indicators.oracles.standard_deviation_rolling import reference_standard_deviation_rolling


def reference_bollinger_bands(
    close: Sequence[float | None],
    window: int,
    multiplier: float = 2.0,
) -> dict[str, list[float | None]]:
    """
    Naive Bollinger Bands over a Python list.

    The center band is :func:`reference_sma`; the outer bands sit ``multiplier`` population standard deviations
    (:func:`reference_standard_deviation_rolling`) away. Recomputed from scratch as the oracle for
    :func:`pomata.indicators.bollinger_bands`.

    Args:
        close: Close-price series (may contain ``None`` and ``float('nan')``).
        window: Number of observations in the moving window. Must be ``>= 1``.
        multiplier: Number of standard deviations between the center band and each outer band.

    Returns:
        A dict with three lists the same length as ``close`` — ``"lower"``, ``"middle"``, and ``"upper"`` — matching the
        fields of the indicator's struct. The first ``window - 1`` entries of each are ``None`` (warm-up).

    Raises:
        ValueError: If ``window < 1``.

    Note:
        A ``None`` or ``nan`` in a window propagates to all three bands at that row (``None`` taking precedence), since
        the center and the deviation read the same window.
    """
    middle = reference_sma(close, window)
    sigma = reference_standard_deviation_rolling(close, window)
    lower: list[float | None] = []
    center: list[float | None] = []
    upper: list[float | None] = []
    for mean, deviation in zip(middle, sigma, strict=True):
        if mean is None or deviation is None:
            lower.append(None)
            center.append(None)
            upper.append(None)
        elif math.isnan(mean) or math.isnan(deviation):
            lower.append(math.nan)
            center.append(math.nan)
            upper.append(math.nan)
        else:
            lower.append(mean - multiplier * deviation)
            center.append(mean)
            upper.append(mean + multiplier * deviation)
    return {"lower": lower, "middle": center, "upper": upper}
