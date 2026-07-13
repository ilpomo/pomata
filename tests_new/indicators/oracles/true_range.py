"""
Naive reference oracle for ``pomata.indicators.true_range``.
"""

import math
from collections.abc import Sequence


def true_range_reference(
    high: Sequence[float | None],
    low: Sequence[float | None],
    close: Sequence[float | None],
) -> list[float | None]:
    """
    Naive True Range over three aligned Python lists.

    The true range ``max(high - low, |high - prev_close|, |low - prev_close|)``, recomputed as the oracle for
    :func:`pomata.indicators.true_range`. Its one subtlety is ``pl.max_horizontal`` 's asymmetric missing-data rule —
    it *skips* ``None`` candidates but lets a ``nan`` dominate the maximum — so the first row, with no previous close,
    reduces to ``high - low``; detailed below.

    Args:
        high: High-price series for the bar (may contain ``None`` and ``float('nan')``).
        low: Low-price series for the bar (may contain ``None`` and ``float('nan')``); must be the same length as
            ``high``.
        close: Close-price series for the bar (may contain ``None`` and ``float('nan')``); must be the same length as
            ``high``. The previous close supplies the two gap terms of each row.

    Returns:
        A list of the same length as the inputs. There is no warm-up: every row is the maximum of its surviving
        candidate distances. On well-formed OHLC data (``high >= low``) every value is non-negative.

    Raises:
        ValueError: If the three input lists do not all have the same length.

    Note:
        Edge-case behavior:

        - **Null** — null handling follows ``pl.max_horizontal``, which **skips** ``None`` candidates rather than
          propagating them: a ``None`` in ``high`` or ``low`` (or a ``None`` previous ``close``) simply drops that
          candidate, so the row still resolves from whichever distances remain. The result is ``None`` only when all
          three candidates are ``None`` (``high`` and ``low`` both ``None`` at the row, and no usable previous close).
        - **NaN** — a ``nan`` is **not** skipped: it dominates the maximum, so any row whose surviving candidates
          include a ``nan`` yields ``nan`` (a ``nan`` ``close`` therefore contaminates the two gap terms of the **next**
          row only, not the whole series).
    """
    if not len(high) == len(low) == len(close):
        raise ValueError(f"high, low, and close must have equal length, got {len(high)}, {len(low)}, and {len(close)}")

    results: list[float | None] = []
    previous_close: float | None = None  # None for the first row, so its two gap terms drop, leaving high - low
    for value_high, value_low, value_close in zip(high, low, close, strict=True):
        # Build only the candidates whose operands are all non-null, mirroring pl.max_horizontal skipping null inputs.
        candidates: list[float] = []
        if value_high is not None and value_low is not None:
            candidates.append(value_high - value_low)
        if value_high is not None and previous_close is not None:
            candidates.append(abs(value_high - previous_close))
        if value_low is not None and previous_close is not None:
            candidates.append(abs(value_low - previous_close))
        if not candidates:
            results.append(None)  # all three candidates dropped (high and low both null, no usable previous close)
        elif any(math.isnan(candidate) for candidate in candidates):
            results.append(math.nan)  # a nan is never skipped: it dominates the maximum
        else:
            results.append(max(candidates))
        previous_close = value_close  # carry this row's close forward as the next row's previous close
    return results
