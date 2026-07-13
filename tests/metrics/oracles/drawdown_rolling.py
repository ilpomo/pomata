"""
Naive reference oracle for ``pomata.metrics.drawdown_rolling``.
"""

import math
from collections.abc import Sequence


def drawdown_rolling_reference(equity_curve: Sequence[float | None], window: int) -> list[float | None]:
    """
    Naive rolling drawdown: the current equity over the trailing window's peak, less one, recomputed from scratch.

    Row ``i`` is ``E[i] / max(E[i - window + 1 : i + 1]) - 1`` over the window. The first ``window - 1`` rows are
    warm-up ``None``; a window holding any ``None`` is ``None`` (the window must hold ``window`` non-null values);
    otherwise a ``NaN`` anywhere in the window yields ``nan``.
    """
    output: list[float | None] = []
    for index in range(len(equity_curve)):
        if index < window - 1:
            output.append(None)
            continue
        window_slice = equity_curve[index - window + 1 : index + 1]
        finite = [value for value in window_slice if value is not None]
        if len(finite) < window:
            output.append(None)
        elif any(math.isnan(value) for value in finite):
            output.append(math.nan)
        else:
            output.append(finite[-1] / max(finite) - 1.0)
    return output
