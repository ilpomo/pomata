"""
Naive reference oracle for ``pomata.metrics.gain_to_pain_ratio``.
"""

import math
from collections.abc import Sequence


def gain_to_pain_ratio_reference(returns: Sequence[float | None]) -> float | None:
    """
    Naive gain to pain ratio over a Python list.

    The sum of all returns over the magnitude of the sum of the negative returns, recomputed from scratch as the oracle
    for :func:`pomata.metrics.gain_to_pain_ratio`. ``None`` returns are skipped; a ``nan`` anywhere poisons the result
    to ``nan``; with no observations the result is ``None``; with no losses the result is ``+inf`` (or ``nan`` when the
    net return is also zero), matching the implementation's division.
    """
    observations = [value for value in returns if value is not None]
    if any(math.isnan(value) for value in observations):
        return math.nan
    if not observations:
        return None
    net = sum(observations)
    pain = sum(-value for value in observations if value < 0.0)
    if pain == 0.0:
        return math.inf if net > 0.0 else math.nan
    return net / pain
