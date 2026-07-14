"""
This pins the test infrastructure: the scaling guard's derived bound must follow its documented formula, so the
benchmark tier's discrimination margins hold exactly as ``tests/support/benchmarks.py`` derives them.
"""

from tests.support import SCALING_OVERHEAD_MULTIPLE, scaling_threshold


def test_threshold_follows_the_derived_formula() -> None:
    """Verifies the one-decade bound is exactly 3 * 10**degree at the first two polynomial degrees."""
    assert scaling_threshold(1) == 30.0
    assert scaling_threshold(2) == 300.0


def test_threshold_grows_a_decade_per_degree() -> None:
    """Verifies each declared degree widens the bound by exactly one decade, mirroring the cost model."""
    for degree in range(1, 5):
        assert scaling_threshold(degree + 1) == 10.0 * scaling_threshold(degree)


def test_stop_rule_multiple_guarantees_the_derived_signal() -> None:
    """Verifies the stop rule's multiple leaves a signal of at least 3 (t >= 4h implies c * n**k >= 3h)."""
    assert SCALING_OVERHEAD_MULTIPLE - 1.0 >= 3.0
