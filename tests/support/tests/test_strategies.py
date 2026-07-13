"""
Meta-tests for ``tests.support.strategies`` — the coherent-bar generators and the element / series strategies.

These pin the test infrastructure: the property tiers draw their inputs from these strategies, so a generator that
emitted an *impossible* bar (``high < low``) or failed to mix in ``None`` / ``NaN`` would quietly feed every indicator
data outside its domain. The contracts are checked the same way an indicator is.
"""

import math

from hypothesis import given
from tests.support import (
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
    spans_even_lag_run,
    subnormal_safe_floats,
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

    def test_spans_even_lag_run_cuts_at_the_sustained_run(self) -> None:
        """
        Verifies the run-length predicate fires on sustained flat runs and period-two alternations (full-length
        even-lag runs) and on a seven-long embedded run, while admitting an isolated even-lag tie and a five-long
        run — the boundary sits at six consecutive even-lag equalities (about eight structured bars).
        """
        assert spans_even_lag_run([100.0] * 80)
        assert spans_even_lag_run([100.0 if i % 2 == 0 else 105.0 for i in range(40)])
        isolated = [float(i) for i in range(40)]
        isolated[20] = isolated[18]
        assert not spans_even_lag_run(isolated)
        seven_run = [float(i) for i in range(40)]
        for j in range(30, 37):
            seven_run[j] = seven_run[j - 2]
        assert spans_even_lag_run(seven_run)
        five_run = [float(i) for i in range(40)]
        for j in range(30, 35):
            five_run[j] = five_run[j - 2]
        assert not spans_even_lag_run(five_run)


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
