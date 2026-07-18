"""
Self-tests of :mod:`tests_new.support.strategies` — the live element domains the single-input fuzz path draws from.

These pin the test infrastructure: ``fuzz_frames`` builds the single-input property-tier frames from ``finite_floats``
and ``missing_data_floats``, so a domain that overran its bound or a missing-data domain that stopped mixing in ``None``
and ``NaN`` would silently feed every single-input factory data outside the regime the tier claims to cover. The
contracts are checked the same way a factory is — with Hypothesis.
"""

import math

from hypothesis import given

from tests_new.support.strategies import FLOOR_SUBNORMAL, finite_floats, missing_data_floats


class TestFiniteFloats:
    """``finite_floats`` draws a finite value inside its symmetric bound."""

    @given(value=finite_floats(bound=1e3))
    def test_finite_and_bounded(self, value: float) -> None:
        """A draw is finite and within the requested bound."""
        assert math.isfinite(value)
        assert -1e3 <= value <= 1e3


class TestMissingDataFloats:
    """``missing_data_floats`` interleaves ``None`` / ``NaN`` / finite draws and honors the magnitude floor."""

    @given(value=missing_data_floats(min_magnitude=FLOOR_SUBNORMAL))
    def test_kind_and_floor(self, value: float | None) -> None:
        """A draw is ``None`` / ``NaN`` / finite, and a finite draw stays in band and above the magnitude floor."""
        assert value is None or isinstance(value, float)
        if value is not None and not math.isnan(value):
            assert -1e6 <= value <= 1e6
            assert abs(value) >= FLOOR_SUBNORMAL
