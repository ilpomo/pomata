"""
Naive reference oracle for ``pomata.metrics.skewness_rolling``.
"""

from collections.abc import Sequence

from tests_new.metrics.oracles._rolling import rolling_reference
from tests_new.metrics.oracles.skewness import skewness_reference


def skewness_rolling_reference(values: Sequence[float | None], window: int) -> list[float | None]:
    """
    The reducing reference applied to each trailing window (warm-up and any-null windows are ``None``).
    """
    return rolling_reference(skewness_reference, values, window)
