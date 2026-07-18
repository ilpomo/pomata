"""
Naive reference oracle for ``pomata.indicators.price_average``.
"""

from collections.abc import Sequence


def reference_price_average(
    open: Sequence[float | None],
    high: Sequence[float | None],
    low: Sequence[float | None],
    close: Sequence[float | None],
) -> list[float | None]:
    """
    Naive Average Price over four aligned Python lists.

    The equal-weighted OHLC mean ``(open + high + low + close) / 4``, recomputed as the oracle for
    :func:`pomata.indicators.price_average`. It is elementwise; its one subtlety is the missing-data rule of plain
    arithmetic — ``None`` propagates and a ``nan`` propagates to ``nan`` — detailed below.

    Args:
        open: Open-price series for the bar (may contain ``None`` and ``float('nan')``).
        high: High-price series for the bar (may contain ``None`` and ``float('nan')``); same length as ``open``.
        low: Low-price series for the bar (may contain ``None`` and ``float('nan')``); same length as ``open``.
        close: Close-price series for the bar (may contain ``None`` and ``float('nan')``); same length as ``open``.

    Returns:
        A list of the same length as the inputs. There is no warm-up: every row is the mean of its four prices.

    Raises:
        ValueError: If the four input lists do not all have the same length.

    Note:
        Edge-case behavior:

        - **Null** — a ``None`` in any of the four inputs makes that row ``None`` (``None`` takes precedence over
          ``nan``).
        - **NaN** — a ``nan`` in any input (with no ``None`` at that row) propagates to ``nan``.
    """
    if not len(open) == len(high) == len(low) == len(close):
        raise ValueError("open, high, low, and close must have equal length")

    results: list[float | None] = []
    for value_open, value_high, value_low, value_close in zip(open, high, low, close, strict=True):
        if value_open is None or value_high is None or value_low is None or value_close is None:
            results.append(None)
        else:
            results.append((value_open + value_high + value_low + value_close) / 4.0)
    return results
