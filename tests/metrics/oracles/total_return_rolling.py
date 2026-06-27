"""
Naive reference oracle for ``pomata.metrics.total_return_rolling``.
"""

import math
from collections.abc import Sequence


def total_return_rolling_reference(equity_curve: Sequence[float | None], window: int) -> list[float | None]:
    """
    Naive rolling total return: the window's last equity over its first, less one, recomputed from scratch.

    An endpoint quantity: row ``i`` is ``E[i] / E[i - window + 1] - 1``. The first ``window - 1`` rows are warm-up
    ``None``; a ``None`` at either endpoint yields ``None``; a ``NaN`` at either endpoint yields ``nan``. An interior
    ``None`` / ``NaN`` does not affect the result.
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
            output.append(last / first - 1.0)
    return output
