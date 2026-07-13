"""
Naive reference oracle for ``pomata.metrics.tail_ratio_rolling``.
"""

from collections.abc import Sequence

from tests_new.metrics.oracles._rolling import rolling_reference
from tests_new.metrics.oracles.tail_ratio import tail_ratio_reference


def tail_ratio_rolling_reference(values: Sequence[float | None], window: int) -> list[float | None]:
    """
    The reducing reference applied to each trailing window (warm-up and any-null windows are ``None``).
    """
    return rolling_reference(tail_ratio_reference, values, window)
