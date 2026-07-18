"""
Naive reference oracle for ``pomata.indicators.aroon``.
"""

import math
from collections.abc import Sequence


def _aroon_line(
    window_values: Sequence[float | None],
    window: int,
    *,
    seek_maximum: bool,
) -> float | None:
    """
    The Aroon value for one ``window + 1``-bar look-back: ``100 * (window - bars since the extreme) / window``.

    ``None`` if any bar is ``None``; ``nan`` if any is ``nan`` (it propagates rather than counting as an extreme); on
    ties the most recent extreme (smallest bars-since) is used.
    """
    if any(value is None for value in window_values):
        return None
    if any(isinstance(value, float) and math.isnan(value) for value in window_values):
        return math.nan
    extreme = (
        max(value for value in window_values if value is not None)
        if seek_maximum
        else min(value for value in window_values if value is not None)
    )
    span = window + 1
    periods_since = min(window - position for position in range(span) if window_values[position] == extreme)
    return 100.0 * (window - periods_since) / window


def reference_aroon(
    high: Sequence[float | None],
    low: Sequence[float | None],
    window: int,
) -> dict[str, list[float | None]]:
    """
    Naive Aroon over aligned Python lists.

    Each line measures how recently the extreme of the last ``window + 1`` bars occurred, as a percentage of the window,
    recomputed from scratch as the oracle for :func:`pomata.indicators.aroon`.

    Args:
        high: High-price series (may contain ``None`` and ``float('nan')``).
        low: Low-price series (may contain ``None`` and ``float('nan')``); same length as ``high``.
        window: Look-back length; the extreme is sought over the last ``window + 1`` bars. Must be ``>= 1``.

    Returns:
        A dict with two lists the same length as the inputs — ``"up"`` and ``"down"`` — matching the fields of the
        indicator's struct. The first ``window`` entries of each are ``None`` (warm-up).

    Raises:
        ValueError: If ``window < 1`` or the inputs differ in length.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    if len(high) != len(low):
        raise ValueError("high and low must have equal length")
    up: list[float | None] = []
    down: list[float | None] = []
    for index in range(len(high)):
        if index < window:
            up.append(None)
            down.append(None)
            continue
        up.append(_aroon_line(high[index - window : index + 1], window, seek_maximum=True))
        down.append(_aroon_line(low[index - window : index + 1], window, seek_maximum=False))
    return {"up": up, "down": down}
