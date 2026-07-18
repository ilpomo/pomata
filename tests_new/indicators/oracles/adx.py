"""
Naive reference oracle for ``pomata.indicators.adx``.
"""

from collections.abc import Sequence

from tests_new.indicators.oracles.dx import reference_dx
from tests_new.indicators.oracles.rma import reference_rma


def reference_adx(
    high: Sequence[float | None],
    low: Sequence[float | None],
    close: Sequence[float | None],
    window: int,
) -> list[float | None]:
    """
    Naive Average Directional Index over aligned Python lists.

    The Wilder moving average (:func:`reference_rma`) of the directional index (:func:`reference_dx`), recomputed as the
    oracle for :func:`pomata.indicators.adx`.

    Args:
        high: High-price series (may contain ``None`` and ``float('nan')``).
        low: Low-price series (may contain ``None`` and ``float('nan')``); same length as ``high``.
        close: Close-price series (may contain ``None`` and ``float('nan')``); same length as ``high``.
        window: Number of observations in the Wilder moving window. Must be ``>= 1``.

    Returns:
        A list the same length as the inputs: the average directional index, ``None`` through the warm-up.

    Raises:
        ValueError: If ``window < 1``.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    return reference_rma(reference_dx(high, low, close, window), window)
