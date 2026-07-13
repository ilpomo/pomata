"""
Naive reference oracle for ``pomata.indicators.trix``.
"""

from collections.abc import Sequence

from tests.indicators.oracles.ema import ema_reference
from tests.indicators.oracles.roc import roc_reference


def trix_reference(
    close: Sequence[float | None],
    window: int,
) -> list[float | None]:
    """
    Naive TRIX over a Python list.

    The one-period rate of change (:func:`roc_reference`) of a triple-smoothed exponential moving average
    (:func:`ema_reference`), recomputed from scratch as the oracle for :func:`pomata.indicators.trix`.

    Args:
        close: Close-price series (may contain ``None`` and ``float('nan')``).
        window: Span of each of the three EMA passes. Must be ``>= 1``.

    Returns:
        A list the same length as ``close``: the one-period rate of change of the triple EMA, ``None`` through the
        ``3 * (window - 1) + 1`` warm-up rows.

    Raises:
        ValueError: If ``window < 1``.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    triple_ema = ema_reference(ema_reference(ema_reference(close, window), window), window)
    return roc_reference(triple_ema, 1)
