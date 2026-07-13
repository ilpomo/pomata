"""
Naive reference oracle for ``pomata.indicators.atr``.
"""

from collections.abc import Sequence

from tests_new.indicators.oracles.rma import rma_reference
from tests_new.indicators.oracles.true_range import true_range_reference


def atr_reference(
    high: Sequence[float | None],
    low: Sequence[float | None],
    close: Sequence[float | None],
    window: int,
) -> list[float | None]:
    """
    Naive Average True Range over three Python lists.

    The Average True Range ``RMA(true_range, window)``, recomputed as the oracle for :func:`pomata.indicators.atr` by
    composing :func:`true_range_reference` and :func:`rma_reference`. Its behavior — the ``max_horizontal`` true
    range, then Wilder smoothing's seed, warm-up, and null-bridging — is inherited from those two oracles, detailed
    below.

    Args:
        high: The high-price observations (may contain ``None`` and ``float('nan')``).
        low: The low-price observations (may contain ``None`` and ``float('nan')``); must be the same length as
            ``high``.
        close: The close-price observations (may contain ``None`` and ``float('nan')``); must be the same length as
            ``high``. The previous close seeds the two gap terms of each true range.
        window: Window length, mapped to the Wilder smoothing factor ``alpha = 1 / window``. Must be ``>= 1``.

    Returns:
        A list of the same length as the inputs. The first ``window - 1`` entries are ``None`` (warm-up), until
        ``window`` non-null true ranges have been counted; thereafter the Wilder-smoothed true range.

    Raises:
        ValueError: If ``window < 1`` or if the three input lists do not all have the same length.

    Note:
        Edge-case behavior:

        - **Null** — null handling follows ``pl.max_horizontal``: a ``None`` in a single ``high``, ``low``, or
          ``close`` drops only the candidate terms that reference it, leaving the true range as the maximum of the
          remaining terms. The true range is ``None`` only when every candidate term is ``None`` (e.g. the first bar
          with both ``high`` and ``low`` ``None``); a ``None`` true range yields ``None`` at that position while the
          Wilder recursion preserves its state and bridges the gap.
        - **NaN** — a ``nan`` in any active term poisons that true range and then the recursion, latching ``nan`` for
          every subsequent value.
        - **window == 1** — the smoothing factor is ``1`` and the warm-up vanishes, so the result reproduces the
          (``max_horizontal``-reduced) true range exactly.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    if not (len(high) == len(low) == len(close)):
        raise ValueError(f"high, low, close must have equal length, got {len(high)}, {len(low)} and {len(close)}")
    true_range_values = true_range_reference(high, low, close)
    return rma_reference(true_range_values, window)
