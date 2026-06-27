"""
Naive reference oracle for ``pomata.metrics.omega_ratio``.
"""

import math
from collections.abc import Sequence


def omega_ratio_reference(returns: Sequence[float | None], threshold: float) -> float | None:
    """
    Naive omega ratio over a Python list.

    The mean gain above the threshold divided by the mean loss below it -- ``E[max(r - tau, 0)] / E[max(tau - r, 0)]``
    over the non-null returns, recomputed from scratch as the oracle for :func:`pomata.metrics.omega_ratio`. ``None``
    returns are skipped; a ``nan`` anywhere poisons the result to ``nan``; with no observations the result is ``None``.
    With no downside the ratio is ``+inf`` (or ``nan`` when there is also no upside), matching the implementation's
    division.
    """
    observations = [value for value in returns if value is not None]
    if any(math.isnan(value) for value in observations):
        return math.nan
    if not observations:
        return None
    excess = [value - threshold for value in observations]
    count = len(excess)
    mean_gain = sum(value for value in excess if value > 0.0) / count
    mean_loss = sum(-value for value in excess if value < 0.0) / count
    if mean_loss == 0.0:
        return math.nan if mean_gain == 0.0 else math.inf
    return mean_gain / mean_loss
