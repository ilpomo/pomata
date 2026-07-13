"""
Naive reference oracle for ``pomata.metrics.downside_deviation_rolling``.
"""

from collections.abc import Sequence

from tests_new.metrics.oracles._rolling import rolling_reference
from tests_new.metrics.oracles.downside_deviation import downside_deviation_reference


def downside_deviation_rolling_reference(
    values: Sequence[float | None], window: int, periods_per_year: int, threshold: float = 0.0
) -> list[float | None]:
    """
    The reducing reference applied to each trailing window (warm-up and any-null windows are ``None``).
    """
    return rolling_reference(
        lambda window_slice: downside_deviation_reference(window_slice, periods_per_year, threshold), values, window
    )
