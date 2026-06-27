"""
Naive reference oracle for ``pomata.indicators.price_typical``.
"""

from collections.abc import Sequence


def price_typical_reference(
    high: Sequence[float | None],
    low: Sequence[float | None],
    close: Sequence[float | None],
) -> list[float | None]:
    """
    Naive Typical Price over three aligned Python lists.

    The equal-weighted mean of high, low, and close ``(high + low + close) / 3``, recomputed as the oracle for
    :func:`pomata.indicators.price_typical`. It is elementwise; its one subtlety is the missing-data rule of plain
    arithmetic — ``None`` propagates and a ``nan`` propagates to ``nan`` — detailed below.

    Args:
        high: High-price series for the bar (may contain ``None`` and ``float('nan')``).
        low: Low-price series for the bar (may contain ``None`` and ``float('nan')``); same length as ``high``.
        close: Close-price series for the bar (may contain ``None`` and ``float('nan')``); same length as ``high``.

    Returns:
        A list of the same length as the inputs. There is no warm-up: every row is the mean of its ``high``, ``low``,
        and ``close``.

    Raises:
        ValueError: If the three input lists do not all have the same length.

    Note:
        Edge-case behavior:

        - **Null** — a ``None`` in any of ``high`` / ``low`` / ``close`` makes that row ``None`` (``None`` takes
          precedence over ``nan``).
        - **NaN** — a ``nan`` in any input (with no ``None`` at that row) propagates to ``nan``.
    """
    if not len(high) == len(low) == len(close):
        raise ValueError(f"high, low, and close must have equal length, got {len(high)}, {len(low)}, and {len(close)}")

    results: list[float | None] = []
    for value_high, value_low, value_close in zip(high, low, close, strict=True):
        if value_high is None or value_low is None or value_close is None:
            results.append(None)
        else:
            results.append((value_high + value_low + value_close) / 3.0)
    return results
