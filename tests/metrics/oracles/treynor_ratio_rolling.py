"""
Naive reference oracle for ``pomata.metrics.treynor_ratio_rolling``.
"""

from collections.abc import Sequence

from tests.metrics.oracles._rolling import rolling_reference_pair
from tests.metrics.oracles.treynor_ratio import treynor_ratio_reference


def treynor_ratio_rolling_reference(
    returns: Sequence[float | None],
    benchmark: Sequence[float | None],
    window: int,
    periods_per_year: int,
    risk_free_rate: float = 0.0,
) -> list[float | None]:
    """
    Naive rolling Treynor ratio: the reducing reference applied to each trailing returns/benchmark window
    (warm-up / any-null windows are ``None``).
    """
    return rolling_reference_pair(
        lambda window_returns, window_benchmark: treynor_ratio_reference(
            window_returns, window_benchmark, periods_per_year, risk_free_rate
        ),
        returns,
        benchmark,
        window,
    )
