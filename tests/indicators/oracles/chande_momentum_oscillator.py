"""
Naive reference oracle for ``pomata.indicators.chande_momentum_oscillator``.
"""

import math
from collections.abc import Sequence


def chande_momentum_oscillator_reference(
    close: Sequence[float | None],
    window: int,
) -> list[float | None]:
    """
    Naive Chande Momentum Oscillator over a Python list.

    The net of gains over losses summed across ``window`` one-step changes, as a percentage of total movement,
    recomputed from scratch as the oracle for :func:`pomata.indicators.chande_momentum_oscillator`.

    Args:
        close: Close-price series (may contain ``None`` and ``float('nan')``).
        window: Number of one-step changes summed in the window. Must be ``>= 1``.

    Returns:
        A list the same length as ``close``: the oscillator in percent, ``None`` through the ``window`` warm-up rows,
        and ``NaN`` for an exactly-flat window (zero total movement, the ``0 / 0`` degenerate).

    Raises:
        ValueError: If ``window < 1``.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")

    length = len(close)
    gain: list[float | None] = [None] * length
    loss: list[float | None] = [None] * length
    for index in range(1, length):
        previous = close[index - 1]
        current = close[index]
        if previous is None or current is None:
            continue
        change = current - previous
        if math.isnan(change):
            gain[index] = math.nan
            loss[index] = math.nan
        else:
            gain[index] = max(change, 0.0)
            loss[index] = max(-change, 0.0)

    results: list[float | None] = []
    for index in range(length):
        if index < window - 1:
            results.append(None)
            continue
        gain_window = gain[index - window + 1 : index + 1]
        loss_window = loss[index - window + 1 : index + 1]
        window_values = gain_window + loss_window
        if any(value is None for value in window_values):
            results.append(None)
        elif any(isinstance(value, float) and math.isnan(value) for value in window_values):
            results.append(math.nan)
        else:
            sum_gain = sum(value for value in gain_window if value is not None)
            sum_loss = sum(value for value in loss_window if value is not None)
            total = sum_gain + sum_loss
            # An exactly-flat window (total movement zero) is the 0/0 degenerate -> NaN, matching the implementation's
            # residual-free rolling-maximum guard; otherwise the bounded quotient, mathematically within [-100, 100].
            results.append(math.nan if total == 0.0 else 100.0 * (sum_gain - sum_loss) / total)
    return results
