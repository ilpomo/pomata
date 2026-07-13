"""
Naive reference oracle for ``pomata.indicators.stochastic_slow``.
"""

from collections.abc import Sequence

from tests_new.indicators.oracles.sma import sma_reference
from tests_new.indicators.oracles.stochastic_fast import stochastic_fast_reference


def stochastic_slow_reference(
    high: Sequence[float | None],
    low: Sequence[float | None],
    close: Sequence[float | None],
    window_k: int,
    window_slowing: int,
    window_d: int,
) -> dict[str, list[float | None]]:
    """
    Naive Slow Stochastic Oscillator over aligned Python lists.

    The raw %K of :func:`stochastic_fast_reference` smoothed by :func:`sma_reference` into the slow %K, then smoothed
    once more into %D, recomputed as the oracle for :func:`pomata.indicators.stochastic_slow`.

    Args:
        high: High-price series (may contain ``None`` and ``float('nan')``).
        low: Low-price series (may contain ``None`` and ``float('nan')``); same length as ``high``.
        close: Close-price series (may contain ``None`` and ``float('nan')``); same length as ``high``.
        window_k: Number of observations in the raw %K look-back range. Must be ``>= 1``.
        window_slowing: Number of observations in the slowing average of the raw %K. Must be ``>= 1``.
        window_d: Number of observations in the %D moving average of the slow %K. Must be ``>= 1``.

    Returns:
        A dict with keys ``"k"`` and ``"d"``, each a list the same length as the inputs.

    Raises:
        ValueError: If ``window_k < 1``, ``window_slowing < 1``, or ``window_d < 1``.
    """
    if window_k < 1:
        raise ValueError(f"window_k must be >= 1, got {window_k}")
    if window_slowing < 1:
        raise ValueError(f"window_slowing must be >= 1, got {window_slowing}")
    if window_d < 1:
        raise ValueError(f"window_d must be >= 1, got {window_d}")
    raw_k = stochastic_fast_reference(high, low, close, window_k, 1)["k"]
    slow_k = sma_reference(raw_k, window_slowing)
    return {"k": slow_k, "d": sma_reference(slow_k, window_d)}
