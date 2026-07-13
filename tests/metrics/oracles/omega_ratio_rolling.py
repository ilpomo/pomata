"""
Naive reference oracle for ``pomata.metrics.omega_ratio_rolling``.
"""

from collections.abc import Sequence

from tests.metrics.oracles._rolling import rolling_reference
from tests.metrics.oracles.omega_ratio import omega_ratio_reference


def omega_ratio_rolling_reference(
    values: Sequence[float | None], window: int, threshold: float = 0.0
) -> list[float | None]:
    """
    The reducing reference applied to each trailing window (warm-up and any-null windows are ``None``).
    """
    return rolling_reference(lambda window_slice: omega_ratio_reference(window_slice, threshold), values, window)
