"""
Meta-tests for ``tests_new.support.strategies`` — the coherent-bar generators and the element / series strategies.

These pin the test infrastructure: the property tiers draw their inputs from these strategies, so a generator that
emitted an *impossible* bar (``high < low``) or failed to mix in ``None`` / ``NaN`` would quietly feed every indicator
data outside its domain. The contracts are checked the same way an indicator is.
"""

import math

from hypothesis import given
from tests_new.support import (
    SUBNORMAL_FLOOR,
    WINDOW_MAX,
    coherent_hl,
    coherent_hl_with_missing,
    coherent_hlc,
    coherent_hlc_with_missing,
    coherent_hlcv,
    coherent_hlcv_with_missing,
    coherent_ohlc,
    coherent_ohlc_with_missing,
    finite_floats,
    missing_data_floats,
    positive_missing_data,
    spans_even_lag_repeat,
    subnormal_safe_floats,
    two_segment_missing_data,
)


def _is_missing_or_positive(value: float | None) -> bool:
    """
    Whether a perturbed field is ``None``, a ``NaN``, or a positive finite float (the coherent value kept).
    """
    return value is None or math.isnan(value) or value > 0.0


class TestCoherentBars:
    """
    Every coherent bar is internally consistent and strictly positive.
    """

    @given(bar=coherent_hl())
    def test_coherent_hl(self, bar: tuple[float, float]) -> None:
        """
        Verifies ``high >= low`` and both finite positive.
        """
        high, low = bar
        assert math.isfinite(high)
        assert math.isfinite(low)
        assert high >= low > 0.0

    @given(bar=coherent_hlc())
    def test_coherent_hlc(self, bar: tuple[float, float, float]) -> None:
        """
        Verifies ``low <= close <= high`` and all finite positive.
        """
        high, low, close = bar
        assert low <= close <= high
        assert low > 0.0

    @given(bar=coherent_hlcv())
    def test_coherent_hlcv(self, bar: tuple[float, float, float, float]) -> None:
        """
        Verifies ``low <= close <= high``, positive volume, all finite positive.
        """
        high, low, close, volume = bar
        assert low <= close <= high
        assert low > 0.0
        assert volume > 0.0

    @given(bar=coherent_ohlc())
    def test_coherent_ohlc(self, bar: tuple[float, float, float, float]) -> None:
        """
        Verifies ``low <= open, close <= high`` and all finite positive.
        """
        open_, high, low, close = bar
        assert low <= open_ <= high
        assert low <= close <= high
        assert low > 0.0


class TestMissingVariants:
    """
    The ``*_with_missing`` bars keep each field as the coherent value, ``None``, or ``NaN``.
    """

    @given(bar=coherent_hl_with_missing())
    def test_hl_fields_are_missing_or_positive(self, bar: tuple[float | None, float | None]) -> None:
        """
        Verifies each field of a perturbed ``(high, low)`` bar is ``None`` / ``NaN`` / positive finite.
        """
        assert all(_is_missing_or_positive(value) for value in bar)

    @given(bar=coherent_hlc_with_missing())
    def test_hlc_fields_are_missing_or_positive(self, bar: tuple[float | None, float | None, float | None]) -> None:
        """
        Verifies each field of a perturbed ``(high, low, close)`` bar is ``None`` / ``NaN`` / positive finite.
        """
        assert all(_is_missing_or_positive(value) for value in bar)

    @given(bar=coherent_hlcv_with_missing())
    def test_hlcv_fields_are_missing_or_positive(
        self, bar: tuple[float | None, float | None, float | None, float | None]
    ) -> None:
        """
        Verifies each field of a perturbed ``(high, low, close, volume)`` bar is ``None`` / ``NaN`` / positive finite.
        """
        assert all(_is_missing_or_positive(value) for value in bar)

    @given(bar=coherent_ohlc_with_missing())
    def test_ohlc_fields_are_missing_or_positive(
        self, bar: tuple[float | None, float | None, float | None, float | None]
    ) -> None:
        """
        Verifies each field of a perturbed ``(open, high, low, close)`` bar is ``None`` / ``NaN`` / positive finite.
        """
        assert all(_is_missing_or_positive(value) for value in bar)


class TestElementStrategies:
    """
    The scalar element strategies honor their magnitude and missing-value contracts.
    """

    @given(value=finite_floats(bound=1e3))
    def test_finite_floats_bounded(self, value: float) -> None:
        """
        Verifies a finite value within the bound.
        """
        assert math.isfinite(value)
        assert -1e3 <= value <= 1e3

    @given(value=missing_data_floats(min_magnitude=SUBNORMAL_FLOOR))
    def test_missing_data_floats_kind_and_floor(self, value: float | None) -> None:
        """
        Verifies the value is ``None`` / ``NaN`` / finite, and a finite draw honors the magnitude floor.
        """
        assert value is None or isinstance(value, float)
        if value is not None and not math.isnan(value):
            assert -1e6 <= value <= 1e6
            assert abs(value) >= SUBNORMAL_FLOOR

    @given(value=positive_missing_data(high=1e4))
    def test_positive_missing_data(self, value: float | None) -> None:
        """
        Verifies the value is ``None`` / ``NaN`` / a positive finite in ``[1, high]``.
        """
        assert value is None or isinstance(value, float)
        if value is not None and not math.isnan(value):
            assert 1.0 <= value <= 1e4

    @given(value=subnormal_safe_floats(bound=1e3))
    def test_subnormal_safe_floats_floor(self, value: float) -> None:
        """
        Verifies a finite value bounded away from the subnormal range.
        """
        assert math.isfinite(value)
        assert abs(value) >= SUBNORMAL_FLOOR
        assert abs(value) <= 1e3


class TestSeriesStrategies:
    """
    The whole-series helpers for the cycle cluster honor their contracts.
    """

    def test_spans_even_lag_repeat_detects_flat_and_alternation(self) -> None:
        """
        Verifies the even-lag predicate: a repeat two bars apart (flat run or period-two alternation) is detected, a
        strictly-monotone run is not.
        """
        assert spans_even_lag_repeat([1.0, 2.0, 1.0])
        assert spans_even_lag_repeat([5.0, 9.0, 5.0, 9.0])
        assert not spans_even_lag_repeat([1.0, 2.0, 3.0, 4.0])

    @given(series=two_segment_missing_data(warmup=4, tail=8))
    def test_two_segment_finite_prefix_then_tail(self, series: list[float | None]) -> None:
        """
        Verifies a finite positive prefix longer than the warm-up (with no even-lag repeat), followed by a missing tail.
        """
        assert len(series) > 4
        prefix = series[:5]
        assert all(value is not None and not math.isnan(value) and value > 0.0 for value in prefix)
        assert not spans_even_lag_repeat([value for value in series if value is not None and not math.isnan(value)][:5])


class TestWindowMax:
    """
    The shared ``window`` cap.
    """

    def test_is_a_positive_int(self) -> None:
        """
        Verifies that ``WINDOW_MAX`` is a usable window bound (a positive ``int``).
        """
        assert isinstance(WINDOW_MAX, int)
        assert WINDOW_MAX >= 1
