"""
Naive reference oracle for ``pomata.metrics.alpha_rolling``.
"""

from collections.abc import Sequence

from tests.metrics.oracles._rolling import rolling_reference_pair
from tests.metrics.oracles.alpha import alpha_reference


def alpha_rolling_reference(
    returns: Sequence[float | None],
    benchmark: Sequence[float | None],
    window: int,
    periods_per_year: int,
    risk_free_rate: float = 0.0,
) -> list[float | None]:
    """
    The reducing reference applied to each trailing returns/benchmark window (warm-up / any-null windows are ``None``).
    """
    return rolling_reference_pair(
        lambda window_returns, window_benchmark: alpha_reference(
            window_returns, window_benchmark, periods_per_year, risk_free_rate
        ),
        returns,
        benchmark,
        window,
    )
