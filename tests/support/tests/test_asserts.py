"""
Meta-tests for ``tests.support.asserts`` — the three-way ``assert_matches`` comparator.

This pins the test infrastructure: ``assert_matches`` is the comparison every correctness tier runs through, so a bug
that blurred the ``null`` / ``NaN`` / finite distinction would silently weaken the whole suite.
"""

import math

import pytest
from tests.support import assert_matches, assert_scale_homogeneous


class TestAssertMatches:
    """
    The element-wise comparator enforces the ``null`` / ``NaN`` / infinity / finite distinction.
    """

    def test_passes_on_equal_including_none_and_nan(self) -> None:
        """
        Verifies that exactly-equal lists (with ``None`` and ``NaN`` in matching positions) pass.
        """
        assert_matches([1.0, None, math.nan], [1.0, None, math.nan])

    def test_passes_within_tolerance(self) -> None:
        """
        Verifies that a finite value within the default tolerance passes.
        """
        assert_matches([1.0 + 1e-13], [1.0])

    def test_fails_on_length_mismatch(self) -> None:
        """
        Verifies that differing lengths fail.
        """
        with pytest.raises(AssertionError):
            assert_matches([1.0], [1.0, 2.0])

    def test_fails_none_vs_nan(self) -> None:
        """
        Verifies that an actual ``None`` against an expected ``NaN`` fails (the distinction is enforced).
        """
        with pytest.raises(AssertionError):
            assert_matches([None], [math.nan])

    def test_fails_nan_vs_none(self) -> None:
        """
        Verifies that an actual ``NaN`` against an expected ``None`` fails.
        """
        with pytest.raises(AssertionError):
            assert_matches([math.nan], [None])

    def test_fails_finite_vs_nan(self) -> None:
        """
        Verifies that a finite actual against an expected ``NaN`` fails.
        """
        with pytest.raises(AssertionError):
            assert_matches([1.0], [math.nan])

    def test_fails_out_of_tolerance(self) -> None:
        """
        Verifies that a finite value outside the tolerance band fails.
        """
        with pytest.raises(AssertionError):
            assert_matches([1.1], [1.0])

    def test_infinity_matches_by_sign(self) -> None:
        """
        Verifies that an expected infinity requires a same-sign infinity (and rejects the opposite sign).
        """
        assert_matches([math.inf], [math.inf])
        assert_matches([-math.inf], [-math.inf])
        with pytest.raises(AssertionError):
            assert_matches([-math.inf], [math.inf])


class TestAssertScaleHomogeneous:
    """
    ``assert_scale_homogeneous`` checks element-wise degree-``degree`` homogeneity with a magnitude-sized floor.
    """

    def test_passes_on_exact_homogeneity(self) -> None:
        """
        Verifies that an exactly rescaled output (including ``None`` and matching infinities) passes.
        """
        assert_scale_homogeneous([None, 2.0, 4.0, math.inf], [None, 1.0, 2.0, math.inf], k=2.0, degree=1)

    def test_fails_out_of_band(self) -> None:
        """
        Verifies that a value off the rescaling relation fails.
        """
        with pytest.raises(AssertionError):
            assert_scale_homogeneous([2.0, 5.0], [1.0, 2.0], k=2.0, degree=1)

    def test_infinite_base_row_does_not_disarm(self) -> None:
        """
        Verifies that an ``inf`` in the base output cannot size the absolute floor to ``inf`` and silently pass every
        other (wholly wrong) row of the example.
        """
        with pytest.raises(AssertionError):
            assert_scale_homogeneous([math.inf, 999999.0, -42.0], [math.inf, 2.0, 3.0], k=2.0, degree=1)
