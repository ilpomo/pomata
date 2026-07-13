"""
Naive reference oracle for ``pomata.metrics.sortino_ratio_rolling``.
"""

from collections.abc import Sequence

from tests.metrics.oracles._rolling import rolling_reference
from tests.metrics.oracles.sortino_ratio import sortino_ratio_reference


def sortino_ratio_rolling_reference(
    values: Sequence[float | None], window: int, periods_per_year: int, risk_free_rate: float = 0.0
) -> list[float | None]:
    """
    The reducing reference applied to each trailing window (warm-up and any-null windows are ``None``).
    """
    return rolling_reference(
        lambda window_slice: sortino_ratio_reference(window_slice, periods_per_year, risk_free_rate), values, window
    )
