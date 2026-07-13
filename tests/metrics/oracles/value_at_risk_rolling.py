"""
Naive reference oracle for ``pomata.metrics.value_at_risk_rolling``.
"""

from collections.abc import Sequence

from tests.metrics.oracles._rolling import rolling_reference
from tests.metrics.oracles.value_at_risk import value_at_risk_reference


def value_at_risk_rolling_reference(
    values: Sequence[float | None], window: int, confidence: float = 0.95
) -> list[float | None]:
    """
    The reducing reference applied to each trailing window (warm-up and any-null windows are ``None``).
    """
    return rolling_reference(lambda window_slice: value_at_risk_reference(window_slice, confidence), values, window)
