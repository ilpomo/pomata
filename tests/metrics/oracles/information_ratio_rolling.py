"""
Naive reference oracle for ``pomata.metrics.information_ratio_rolling``.
"""

from collections.abc import Sequence

from tests.metrics.oracles._rolling import rolling_reference_pair
from tests.metrics.oracles.information_ratio import information_ratio_reference


def information_ratio_rolling_reference(
    returns: Sequence[float | None], benchmark: Sequence[float | None], window: int, periods_per_year: int
) -> list[float | None]:
    """
    The reducing reference applied to each trailing returns/benchmark window (warm-up / any-null windows are ``None``).
    """
    return rolling_reference_pair(
        lambda window_returns, window_benchmark: information_ratio_reference(
            window_returns, window_benchmark, periods_per_year
        ),
        returns,
        benchmark,
        window,
    )
