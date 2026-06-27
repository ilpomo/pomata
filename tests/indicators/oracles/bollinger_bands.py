"""
Naive reference oracle for ``pomata.indicators.bollinger_bands``.
"""

import math
from collections.abc import Sequence

from tests.indicators.oracles.sma import sma_reference
from tests.indicators.oracles.standard_deviation_rolling import standard_deviation_rolling_reference


def bollinger_bands_reference(
    close: Sequence[float | None],
    window: int,
    num_std: float = 2.0,
) -> dict[str, list[float | None]]:
    """
    Naive Bollinger Bands over a Python list.

    The center band is :func:`sma_reference`; the outer bands sit ``num_std`` population standard deviations
    (:func:`standard_deviation_rolling_reference`) away. Recomputed from scratch as the oracle for
    :func:`pomata.indicators.bollinger_bands`.

    Args:
        close: Close-price series (may contain ``None`` and ``float('nan')``).
        window: Number of observations in the moving window. Must be ``>= 1``.
        num_std: Number of standard deviations between the center band and each outer band.

    Returns:
        A dict with three lists the same length as ``close`` — ``"lower"``, ``"middle"``, and ``"upper"`` — matching the
        fields of the indicator's struct. The first ``window - 1`` entries of each are ``None`` (warm-up).

    Raises:
        ValueError: If ``window < 1``.

    Note:
        A ``None`` or ``nan`` in a window propagates to all three bands at that row (``None`` taking precedence), since
        the center and the deviation read the same window.
    """
    middle = sma_reference(close, window)
    sigma = standard_deviation_rolling_reference(close, window)
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
            lower.append(mean - num_std * deviation)
            center.append(mean)
            upper.append(mean + num_std * deviation)
    return {"lower": lower, "middle": center, "upper": upper}
