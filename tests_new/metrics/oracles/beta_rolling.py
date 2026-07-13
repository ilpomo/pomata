"""
Naive reference oracle for ``pomata.metrics.beta_rolling``.
"""

from collections.abc import Sequence

from tests_new.metrics.oracles._rolling import rolling_reference_pair
from tests_new.metrics.oracles.beta import beta_reference


def beta_rolling_reference(
    returns: Sequence[float | None], benchmark: Sequence[float | None], window: int
) -> list[float | None]:
    """
    The reducing reference applied to each trailing returns/benchmark window (warm-up / any-null windows are ``None``).
    """
    return rolling_reference_pair(beta_reference, returns, benchmark, window)
