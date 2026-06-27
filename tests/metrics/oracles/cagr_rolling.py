"""
Naive reference oracle for ``pomata.metrics.cagr_rolling``.
"""

import math
from collections.abc import Sequence


def cagr_rolling_reference(
    equity_curve: Sequence[float | None], window: int, periods_per_year: int
) -> list[float | None]:
    """
    Naive rolling compound annual growth rate: the window's endpoint ratio annualized, recomputed from scratch.

    An endpoint quantity: row ``i`` is ``(E[i] / E[i - window + 1]) ** (periods_per_year / window) - 1``. The first
    ``window - 1`` rows are warm-up ``None``; a ``None`` at either endpoint yields ``None``; a ``NaN`` at either
    endpoint yields ``nan``. An interior ``None`` / ``NaN`` does not affect the result.
    """
    output: list[float | None] = []
    for index in range(len(equity_curve)):
        if index < window - 1:
            output.append(None)
            continue
        first = equity_curve[index - window + 1]
        last = equity_curve[index]
        if first is None or last is None:
            output.append(None)
        elif math.isnan(first) or math.isnan(last):
            output.append(math.nan)
        else:
            output.append(math.pow(last / first, periods_per_year / window) - 1.0)
    return output
