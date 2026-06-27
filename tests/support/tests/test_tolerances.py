"""
Meta-tests for ``tests.support.tolerances`` — the ``input_scale`` sizer and the named tolerance ladder.

These pin the test infrastructure: ``input_scale`` sizes every magnitude-relative tolerance, so a bug there would
silently loosen or tighten the property tiers, and the ladder's ordering encodes the conditioning rationale.
"""

import math

from tests.support import (
    ABSOLUTE_TOLERANCE_PROPERTY,
    ABSOLUTE_TOLERANCE_REFERENCE,
    ABSOLUTE_TOLERANCE_SCALE,
    ABSOLUTE_TOLERANCE_STREAMING,
    BOUND_MARGIN,
    EXACT_TOLERANCE_FACTOR,
    RELATIVE_TOLERANCE_PROPERTY,
    RELATIVE_TOLERANCE_REFERENCE,
    RELATIVE_TOLERANCE_SCALE,
    STREAMING_TOLERANCE_FACTOR,
    VARIANCE_TOLERANCE_FACTOR,
    input_scale,
)

ALL_TOLERANCES = (
    ABSOLUTE_TOLERANCE_PROPERTY,
    ABSOLUTE_TOLERANCE_REFERENCE,
    ABSOLUTE_TOLERANCE_SCALE,
    ABSOLUTE_TOLERANCE_STREAMING,
    BOUND_MARGIN,
    EXACT_TOLERANCE_FACTOR,
    RELATIVE_TOLERANCE_PROPERTY,
    RELATIVE_TOLERANCE_REFERENCE,
    RELATIVE_TOLERANCE_SCALE,
    STREAMING_TOLERANCE_FACTOR,
    VARIANCE_TOLERANCE_FACTOR,
)


class TestInputScale:
    """
    ``input_scale`` returns the largest absolute finite value (``1.0`` when there are none).
    """

    def test_largest_absolute_finite(self) -> None:
        """
        Verifies that the magnitude is the maximum absolute value, regardless of sign.
        """
        assert input_scale([1.0, -5.0, 3.0]) == 5.0

    def test_skips_none_and_nan(self) -> None:
        """
        Verifies that ``None`` and ``NaN`` entries are ignored when sizing the magnitude.
        """
        assert input_scale([1.0, None, math.nan, -2.0]) == 2.0

    def test_empty_and_all_missing_default_to_one(self) -> None:
        """
        Verifies the ``1.0`` fallback when there is no finite value to size from.
        """
        assert input_scale([]) == 1.0
        assert input_scale([None, math.nan]) == 1.0


class TestToleranceLadder:
    """
    The named tolerances are positive and ordered as their conditioning rationale documents.
    """

    def test_all_positive(self) -> None:
        """
        Verifies that every tolerance / factor is a positive float (a non-positive band would accept anything or
        nothing).
        """
        for tolerance in ALL_TOLERANCES:
            assert tolerance > 0.0

    def test_magnitude_relative_factor_ordering(self) -> None:
        """
        Verifies the conditioning order: degree-2 variance is tightest, the well-conditioned degree-1 kernels next, the
        square-root-amplified streaming factor loosest.
        """
        assert VARIANCE_TOLERANCE_FACTOR < EXACT_TOLERANCE_FACTOR < STREAMING_TOLERANCE_FACTOR

    def test_tier_ordering(self) -> None:
        """
        Verifies the per-tier band order: reference / property (tightest) <= scale <= streaming.
        """
        assert ABSOLUTE_TOLERANCE_REFERENCE == ABSOLUTE_TOLERANCE_PROPERTY
        assert ABSOLUTE_TOLERANCE_PROPERTY <= ABSOLUTE_TOLERANCE_SCALE <= ABSOLUTE_TOLERANCE_STREAMING
        assert RELATIVE_TOLERANCE_REFERENCE == RELATIVE_TOLERANCE_PROPERTY
        assert RELATIVE_TOLERANCE_PROPERTY <= RELATIVE_TOLERANCE_SCALE
