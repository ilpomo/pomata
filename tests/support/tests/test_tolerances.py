"""
Self-tests of :mod:`tests.support.tolerances` — the ``input_scale`` sizer and the named tolerance ladder.

These pin the test infrastructure: ``input_scale`` sizes every magnitude-relative tolerance, so a bug there would
silently loosen or tighten the property tiers, and the ladder's ordering encodes the conditioning rationale. The
consumer sweep keeps the ladder honest the other way: a named band nothing consumes reads as live protection while
guarding nothing, so every constant must be load-bearing somewhere outside the ladder and this self-test.
"""

import math
import re
from pathlib import Path

from tests.support import tolerances
from tests.support.tolerances import (
    TOLERANCE_ABSOLUTE_PROPERTY,
    TOLERANCE_ABSOLUTE_REFERENCE,
    TOLERANCE_ABSOLUTE_ROLLING_ORACLE,
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
        assert TOLERANCE_FACTOR_EXACT < TOLERANCE_ABSOLUTE_ROLLING_ORACLE

    def test_tier_ordering(self) -> None:
        """The per-tier band order: reference / property (tightest) <= rolling."""
        assert TOLERANCE_ABSOLUTE_REFERENCE == TOLERANCE_ABSOLUTE_PROPERTY
        assert TOLERANCE_ABSOLUTE_PROPERTY <= TOLERANCE_ABSOLUTE_ROLLING_ORACLE
        assert TOLERANCE_RELATIVE_REFERENCE == TOLERANCE_RELATIVE_PROPERTY
        assert TOLERANCE_RELATIVE_PROPERTY <= TOLERANCE_RELATIVE_SCALE


class TestToleranceConsumers:
    """Every named tolerance is load-bearing: something outside the ladder must consume it."""

    def test_every_constant_has_a_consumer(self) -> None:
        """Each UPPER-cased constant is referenced outside ``tolerances.py`` and this self-test — a named band nothing
        consumes reads as live protection while guarding nothing.
        """
        own = {Path("tests/support/tolerances.py").resolve(), Path(__file__).resolve()}
        names = [name for name, value in vars(tolerances).items() if name.isupper() and isinstance(value, float)]
        tree = [path for path in Path("tests").rglob("*.py") if path.resolve() not in own]
        sources = [path.read_text(encoding="utf-8") for path in tree]
        unconsumed = [name for name in names if not any(re.search(rf"\b{name}\b", source) for source in sources)]
        assert not unconsumed, f"tolerance constants with no consumer outside the ladder: {unconsumed}"
