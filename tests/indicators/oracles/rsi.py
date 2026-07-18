"""
Naive reference oracle for ``pomata.indicators.rsi``.
"""

import math
from collections.abc import Sequence

from tests.indicators.oracles.rma import reference_rma


def reference_rsi(
    expr: Sequence[float | None],
    window: int,
) -> list[float | None]:
    """
    Naive Relative Strength Index over a Python list.

    Wilder's RSI, ``100 - 100 / (1 + RMA(gain) / RMA(loss))`` over the one-step gains and losses, recomputed as the
    oracle for :func:`pomata.indicators.rsi` by composing :func:`reference_rma`. Its delicate points are the ``window``
    -row warm-up (the index-0 difference is ``None``) and the relative-strength saturations (``0 / 0`` is ``nan``, no
    losses gives ``100``, no gains gives ``0``), detailed below.

    Args:
        expr: The input observations, typically a price series (may contain ``None`` and ``float('nan')``).
        window: Number of observations in the moving window, mapped to the Wilder smoothing factor
            ``alpha = 1 / window``. Must be ``>= 1``.

    Returns:
        A list the same length as ``expr``. The first ``window`` entries are ``None`` (warm-up, clamped to the length);
        thereafter ``100 - 100 / (1 + avg_gain / avg_loss)``.

    Raises:
        ValueError: If ``window < 1``.

    Note:
        Edge-case behavior:

        - **Null** — a leading ``None`` run is skipped: the warm-up counts only non-``null`` observations, so the
          ``window`` warm-up is measured from the first non-``null`` value. An interior ``None`` yields ``None`` at that
          position while the Wilder recursion bridges the gap.
        - **NaN** — a ``nan`` poisons the recursion and latches ``nan`` for every subsequent non-warm-up position.
        - **Flat window** — both Wilder averages ``0`` is the indeterminate ``0 / 0`` relative strength, surfaced as
          ``nan`` (genuinely undefined, not a conventional ``50`` or ``100``); a zero loss average with a non-zero gain
          average saturates to ``100`` and a zero gain average with a non-zero loss average to ``0``.
        - **window == 1** — the smoothing vanishes: each position reports ``100`` on an up move, ``0`` on a down move,
          and ``nan`` on no move.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")

    # One-step differences: None at index 0 and wherever either endpoint is None (matching Polars' diff).
    delta: list[float | None] = []
    for index in range(len(expr)):
        if index == 0:
            delta.append(None)
            continue
        value_previous = expr[index - 1]
        value_current = expr[index]
        if value_previous is None or value_current is None:
            delta.append(None)
        else:
            delta.append(value_current - value_previous)

    gain: list[float | None] = []
    loss: list[float | None] = []
    for difference in delta:
        if difference is None:
            gain.append(None)
            loss.append(None)
        elif math.isnan(difference):
            gain.append(math.nan)
            loss.append(math.nan)
        else:
            gain.append(max(0.0, difference))
            loss.append(-difference if difference < 0.0 else 0.0)

    average_gain = reference_rma(gain, window)
    average_loss = reference_rma(loss, window)

    results: list[float | None] = []
    for value_gain, value_loss in zip(average_gain, average_loss, strict=True):
        if value_gain is None or value_loss is None:
            results.append(None)
        elif math.isnan(value_gain) or (math.isnan(value_loss)):
            results.append(math.nan)
        elif value_loss == 0.0:
            if value_gain == 0.0:
                results.append(math.nan)
            else:
                relative_strength = math.copysign(math.inf, value_gain)
                results.append(100.0 - 100.0 / (1.0 + relative_strength))
        else:
            relative_strength = value_gain / value_loss
            results.append(100.0 - 100.0 / (1.0 + relative_strength))
    return results
