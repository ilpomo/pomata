"""
Naive reference oracle for ``pomata.indicators.vwma``.
"""

import math
from collections.abc import Sequence


def reference_vwma(
    price: Sequence[float | None],
    volume: Sequence[float | None],
    window: int,
) -> list[float | None]:
    """
    Naive Volume-Weighted Moving Average over two Python lists.

    The volume-weighted mean ``sum(price * volume) / sum(volume)`` over each window, recomputed as the oracle for
    :func:`pomata.indicators.vwma`. Its subtleties are the windowed missing-data contract and the IEEE-754 zero-volume
    degeneracy, detailed below.

    Args:
        price: Input price observations (may contain ``None`` and ``float('nan')``).
        volume: Input traded-volume observations (may contain ``None`` and ``float('nan')``); must be the same length
            as ``price``.
        window: Number of observations in the moving window. Must be ``>= 1``.

    Returns:
        A list the same length as the inputs. The first ``window - 1`` entries are ``None`` (warm-up); thereafter the
        volume-weighted mean ``sum(price * volume) / sum(volume)``.

    Raises:
        ValueError: If ``window < 1`` or if the two input lists do not have the same length.

    Note:
        Edge-case behavior:

        - **Null** — a window in which ``price`` or ``volume`` contains a ``None`` yields ``None`` (matching
          ``rolling_sum``'s default ``min_samples=window``, under which a missing observation leaves the window short);
          ``None`` takes precedence over ``nan``.
        - **NaN** — a window that is free of ``None`` but contains a ``nan`` yields ``nan`` (the ``nan`` contaminates a
          windowed sum). Because each output reads only the current window, a ``None`` or ``nan`` contaminates only the
          positions whose window spans it and never latches onto the rest of the series.
        - **Zero volume** — when the windowed volume sums to zero the division follows IEEE-754: a zero weighted-price
          sum over zero (``0 / 0``) is ``nan``, and a non-zero numerator over zero is ``+/-inf`` with the sign of the
          numerator.
        - **window == 1** — with non-zero volume the single ``(price, volume)`` pair reduces to ``price`` itself, so
          the VWMA reproduces the price.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    if len(price) != len(volume):
        raise ValueError(f"price and volume must have equal length, got {len(price)} and {len(volume)}")

    results: list[float | None] = []
    for index in range(len(price)):
        if index + 1 < window:
            results.append(None)
            continue
        price_window = price[index + 1 - window : index + 1]
        volume_window = volume[index + 1 - window : index + 1]
        pairs = list(zip(price_window, volume_window, strict=True))
        if any(price is None or volume is None for price, volume in pairs):
            results.append(None)
            continue
        clean_pairs = [(price, volume) for price, volume in pairs if price is not None and volume is not None]
        if any(math.isnan(price) or math.isnan(volume) for price, volume in clean_pairs):
            results.append(math.nan)
        else:
            numerator = sum(price * volume for price, volume in clean_pairs)
            denominator = sum(volume for _, volume in clean_pairs)
            if denominator == 0.0:
                results.append(math.nan if numerator == 0.0 else math.copysign(math.inf, numerator))
            else:
                results.append(numerator / denominator)
    return results
