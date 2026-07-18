"""
Naive reference oracle for ``pomata.indicators.balance_of_power``.
"""

from collections.abc import Sequence


def reference_balance_of_power(
    open: Sequence[float | None],
    high: Sequence[float | None],
    low: Sequence[float | None],
    close: Sequence[float | None],
) -> list[float | None]:
    """
    Naive Balance of Power over four aligned Python lists.

    The per-bar ``(close - open) / (high - low)``, recomputed as the oracle for
    :func:`pomata.indicators.balance_of_power`. A flat bar (a finite zero range) yields ``0`` by convention (the
    degenerate ``0 / 0``).

    Args:
        open: Open-price series for the bar (may contain ``None`` and ``float('nan')``).
        high: High-price series for the bar (may contain ``None`` and ``float('nan')``); same length as ``open``.
        low: Low-price series for the bar (may contain ``None`` and ``float('nan')``); same length as ``open``.
        close: Close-price series for the bar (may contain ``None`` and ``float('nan')``); same length as ``open``.

    Returns:
        A list of the same length as the inputs. There is no warm-up: every row is the bar's balance of power.

    Raises:
        ValueError: If the four input lists do not all have the same length.

    Note:
        Edge-case behavior (mirroring the ``pl.when(high - low == 0)`` guard of the implementation):

        - **Flat bar** — a finite zero range (``high - low == 0``) yields ``0``, taking precedence over ``open`` /
          ``close`` being missing.
        - **Null** — otherwise a ``None`` in any input makes that row ``None``.
        - **NaN** — otherwise a ``nan`` in any input (with a non-zero range) propagates to ``nan``.
    """
    if not len(open) == len(high) == len(low) == len(close):
        raise ValueError("open, high, low, and close must have equal length")

    results: list[float | None] = []
    for value_open, value_high, value_low, value_close in zip(open, high, low, close, strict=True):
        if value_high is None or value_low is None:
            results.append(None)
        elif value_high - value_low == 0:
            results.append(0.0)
        elif value_open is None or value_close is None:
            results.append(None)
        else:
            results.append((value_close - value_open) / (value_high - value_low))
    return results
