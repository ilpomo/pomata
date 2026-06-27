"""
Naive reference oracle for ``pomata.metrics.volatility_rolling``.
"""

from collections.abc import Sequence

from tests.metrics.oracles._rolling import rolling_reference
from tests.metrics.oracles.volatility import volatility_reference


def volatility_rolling_reference(
    values: Sequence[float | None], window: int, periods_per_year: int
) -> list[float | None]:
    """
    The reducing reference applied to each trailing window (warm-up and any-null windows are ``None``).
    """
    return rolling_reference(lambda window_slice: volatility_reference(window_slice, periods_per_year), values, window)
