"""
Naive reference oracle for ``pomata.indicators.price_median``.
"""

from collections.abc import Sequence


def reference_price_median(
    high: Sequence[float | None],
    low: Sequence[float | None],
) -> list[float | None]:
    """
    Naive Median Price over two aligned Python lists.

    The midpoint of the bar's range ``(high + low) / 2``, recomputed as the oracle for
    :func:`pomata.indicators.price_median`. It is elementwise; its one subtlety is the missing-data rule of plain
    arithmetic — ``None`` propagates and a ``nan`` propagates to ``nan`` — detailed below.

    Args:
        high: High-price series for the bar (may contain ``None`` and ``float('nan')``).
        low: Low-price series for the bar (may contain ``None`` and ``float('nan')``); same length as ``high``.

    Returns:
        A list of the same length as the inputs. There is no warm-up: every row is the midpoint of its ``high``
        and ``low``.

    Raises:
        ValueError: If the two input lists do not have the same length.

    Note:
        Edge-case behavior:

        - **Null** — a ``None`` in ``high`` or ``low`` makes that row ``None`` (``None`` takes precedence over ``nan``).
        - **NaN** — a ``nan`` in either input (with no ``None`` at that row) propagates to ``nan``.
    """
    if len(high) != len(low):
        raise ValueError(f"high and low must have equal length, got {len(high)} and {len(low)}")

    results: list[float | None] = []
    for value_high, value_low in zip(high, low, strict=True):
        if value_high is None or value_low is None:
            results.append(None)
        else:
            results.append((value_high + value_low) / 2.0)
    return results
