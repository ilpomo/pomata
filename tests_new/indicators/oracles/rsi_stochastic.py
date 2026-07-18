"""
Naive reference oracle for ``pomata.indicators.rsi_stochastic``.
"""

from collections.abc import Sequence

from tests_new.indicators.oracles.rsi import reference_rsi
from tests_new.indicators.oracles.stochastic_fast import reference_stochastic_fast


def reference_rsi_stochastic(
    values: Sequence[float | None],
    window_rsi: int,
    window_k: int,
    window_d: int,
) -> dict[str, list[float | None]]:
    """
    Naive Stochastic RSI over aligned Python lists.

    The :func:`reference_rsi` over ``window_rsi``, fed as the high / low / close of :func:`reference_stochastic_fast`
    (so its %K places the RSI within its own ``window_k`` range, and %D is the average of that), recomputed as the
    oracle for :func:`pomata.indicators.rsi_stochastic`.

    Args:
        values: Input series (may contain ``None`` and ``float('nan')``).
        window_rsi: Number of observations in the underlying RSI. Must be ``>= 1``.
        window_k: Number of observations in the %K look-back range over the RSI. Must be ``>= 1``.
        window_d: Number of observations in the %D moving average of %K. Must be ``>= 1``.

    Returns:
        A dict with keys ``"k"`` and ``"d"``, each a list the same length as the input.

    Raises:
        ValueError: If ``window_rsi < 1``, ``window_k < 1``, or ``window_d < 1``.
    """
    if window_rsi < 1:
        raise ValueError(f"window_rsi must be >= 1, got {window_rsi}")
    if window_k < 1:
        raise ValueError(f"window_k must be >= 1, got {window_k}")
    if window_d < 1:
        raise ValueError(f"window_d must be >= 1, got {window_d}")
    strength = reference_rsi(values, window_rsi)
    return reference_stochastic_fast(strength, strength, strength, window_k, window_d)
