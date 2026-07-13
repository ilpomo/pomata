"""
Naive reference oracle for ``pomata.metrics.kurtosis_rolling``.
"""

from collections.abc import Sequence

from tests.metrics.oracles._rolling import rolling_reference
from tests.metrics.oracles.kurtosis import kurtosis_reference


def kurtosis_rolling_reference(values: Sequence[float | None], window: int) -> list[float | None]:
    """
    The reducing reference applied to each trailing window (warm-up and any-null windows are ``None``).
    """
    return rolling_reference(kurtosis_reference, values, window)
