"""
Self-tests of :mod:`tests_new.support.compare` — the three-way comparator and its non-raising probe.

This pins the test infrastructure: ``assert_matches`` is the comparison every correctness rung runs through, so a bug
that blurred the ``null`` / ``NaN`` / finite distinction would silently weaken the whole suite; ``first_mismatch`` is
what the rungs use to point a failure message at the exact diverging row.
"""

import math

import pytest

from tests_new.support.compare import assert_matches, assert_scale_homogeneous, first_mismatch


class TestAssertMatches:
    """The element-wise comparator enforces the ``null`` / ``NaN`` / infinity / finite distinction."""

    def test_passes_on_equal_including_none_and_nan(self) -> None:
        """Exactly-equal lists (with ``None`` and ``NaN`` in matching positions) pass."""
        assert_matches([1.0, None, math.nan], [1.0, None, math.nan])

    def test_passes_within_tolerance(self) -> None:
        """A finite value within the default tolerance passes."""
        assert_matches([1.0 + 1e-13], [1.0])

    def test_fails_on_length_mismatch(self) -> None:
        """Differing lengths fail."""
        with pytest.raises(AssertionError):
            assert_matches([1.0], [1.0, 2.0])

    def test_fails_none_vs_nan(self) -> None:
        """An actual ``None`` against an expected ``NaN`` fails (the distinction is enforced)."""
        with pytest.raises(AssertionError):
            assert_matches([None], [math.nan])

    def test_fails_nan_vs_none(self) -> None:
        """An actual ``NaN`` against an expected ``None`` fails."""
        with pytest.raises(AssertionError):
            assert_matches([math.nan], [None])

    def test_fails_finite_vs_nan(self) -> None:
        """A finite actual against an expected ``NaN`` fails."""
        with pytest.raises(AssertionError):
            assert_matches([1.0], [math.nan])

    def test_fails_out_of_tolerance(self) -> None:
        """A finite value outside the tolerance band fails."""
        with pytest.raises(AssertionError):
            assert_matches([1.1], [1.0])

    def test_infinity_matches_by_sign(self) -> None:
        """An expected infinity requires a same-sign infinity (and rejects the opposite sign)."""
        assert_matches([math.inf], [math.inf])
        assert_matches([-math.inf], [-math.inf])
        with pytest.raises(AssertionError):
            assert_matches([-math.inf], [math.inf])


class TestFirstMismatch:
    """``first_mismatch`` returns the first diverging index (``None`` when the lists agree)."""

    def test_none_when_equal(self) -> None:
        """Agreeing lists (including ``None`` / ``NaN``) return ``None``."""
        assert first_mismatch([1.0, None, math.nan], [1.0, None, math.nan]) is None

    def test_points_at_first_divergence(self) -> None:
        """The first index whose kind or value diverges is returned."""
        assert first_mismatch([1.0, 2.0, 9.0], [1.0, 2.0, 3.0]) == 2

    def test_kind_divergence_is_caught(self) -> None:
        """A finite value where a ``NaN`` is expected diverges at that index."""
        assert first_mismatch([1.0, 2.0], [1.0, math.nan]) == 1

    def test_length_mismatch_reports_the_shorter_length(self) -> None:
        """A length mismatch reports the first index past the shorter list."""
        assert first_mismatch([1.0], [1.0, 2.0]) == 1


class TestAssertScaleHomogeneous:
    """``assert_scale_homogeneous`` checks element-wise degree-``degree`` homogeneity with a magnitude-sized floor."""

    def test_passes_on_exact_homogeneity(self) -> None:
        """An exactly rescaled output (including ``None`` and matching infinities) passes."""
        assert_scale_homogeneous([None, 2.0, 4.0, math.inf], [None, 1.0, 2.0, math.inf], k=2.0, degree=1)

    def test_fails_out_of_band(self) -> None:
        """A value off the rescaling relation fails."""
        with pytest.raises(AssertionError):
            assert_scale_homogeneous([2.0, 5.0], [1.0, 2.0], k=2.0, degree=1)

    def test_infinite_base_row_does_not_disarm(self) -> None:
        """An ``inf`` in the base output cannot size the absolute floor to ``inf`` and silently pass wrong rows."""
        with pytest.raises(AssertionError):
            assert_scale_homogeneous([math.inf, 999999.0, -42.0], [math.inf, 2.0, 3.0], k=2.0, degree=1)
