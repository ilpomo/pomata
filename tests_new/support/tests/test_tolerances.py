"""
Self-tests of :mod:`tests_new.support.tolerances` — the ``input_scale`` sizer and the named tolerance ladder.

These pin the test infrastructure: ``input_scale`` sizes every magnitude-relative tolerance, so a bug there would
silently loosen or tighten the property tiers, and the ladder's ordering encodes the conditioning rationale.
"""

import math

from tests_new.support import tolerances
from tests_new.support.tolerances import (
    TOLERANCE_ABSOLUTE_PROPERTY,
    TOLERANCE_ABSOLUTE_REFERENCE,
    TOLERANCE_ABSOLUTE_ROLLING_ORACLE,
    TOLERANCE_ABSOLUTE_STREAMING,
    TOLERANCE_FACTOR_EXACT,
    TOLERANCE_RELATIVE_PROPERTY,
    TOLERANCE_RELATIVE_REFERENCE,
    TOLERANCE_RELATIVE_SCALE,
    input_scale,
)

# Every UPPER-cased float in the module: a new tolerance constant is swept in the moment it lands, so the positivity
# check can never skip one.
ALL_TOLERANCES: tuple[float, ...] = tuple(
    value for name, value in vars(tolerances).items() if isinstance(value, float) and name.isupper()
)


class TestInputScale:
    """``input_scale`` returns the largest absolute finite value (``1.0`` when there are none)."""

    def test_infinity_is_not_finite(self) -> None:
        """An ``inf`` never sets the scale: it would size every magnitude-relative tolerance to ``inf``."""
        assert input_scale([1.0, math.inf, 2.0]) == 2.0
        assert input_scale([-math.inf]) == 1.0

    def test_largest_absolute_finite(self) -> None:
        """The magnitude is the maximum absolute value, regardless of sign."""
        assert input_scale([1.0, -5.0, 3.0]) == 5.0

    def test_skips_none_and_nan(self) -> None:
        """``None`` and ``NaN`` entries are ignored when sizing the magnitude."""
        assert input_scale([1.0, None, math.nan, -2.0]) == 2.0

    def test_empty_and_all_missing_default_to_one(self) -> None:
        """The ``1.0`` fallback holds when there is no finite value to size from."""
        assert input_scale([]) == 1.0
        assert input_scale([None, math.nan]) == 1.0


class TestToleranceLadder:
    """The named tolerances are positive and ordered as their conditioning rationale documents."""

    def test_all_positive(self) -> None:
        """Every tolerance / factor is a positive float (a non-positive band would accept anything or nothing)."""
        for tolerance in ALL_TOLERANCES:
            assert tolerance > 0.0

    def test_magnitude_relative_factor_ordering(self) -> None:
        """The magnitude-relative factor stays far below any per-tier band (it multiplies the input scale)."""
        assert TOLERANCE_FACTOR_EXACT < TOLERANCE_ABSOLUTE_STREAMING

    def test_tier_ordering(self) -> None:
        """The per-tier band order: reference / property (tightest) <= rolling <= streaming."""
        assert TOLERANCE_ABSOLUTE_REFERENCE == TOLERANCE_ABSOLUTE_PROPERTY
        assert TOLERANCE_ABSOLUTE_PROPERTY <= TOLERANCE_ABSOLUTE_ROLLING_ORACLE <= TOLERANCE_ABSOLUTE_STREAMING
        assert TOLERANCE_RELATIVE_REFERENCE == TOLERANCE_RELATIVE_PROPERTY
        assert TOLERANCE_RELATIVE_PROPERTY <= TOLERANCE_RELATIVE_SCALE
