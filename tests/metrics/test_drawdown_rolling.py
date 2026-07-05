"""
Tests for ``pomata.metrics.drawdown_rolling`` — the rolling (windowed) twin of :func:`pomata.metrics.drawdown`.

``drawdown_rolling`` is single-input and WINDOWED-SERIES-VALUED (an equity series → a series the same length, one value
per trailing window), so tests use the shared ``apply_expr`` helper; ``assert_matches`` and the naive
``drawdown_rolling_reference`` oracle (the current equity over the window peak, less one, recomputed over each window)
are shared across the suite. The rolling null/NaN policy differs from the running :func:`drawdown`: a window holding any
``null`` is ``null`` (it must hold ``window`` non-null values), and a ``NaN`` inside a window propagates.

The ladder is the canonical one: contract (type / length-preserving / lazy-eager / ``.over`` per-group warm-up), edge
(validation / empty / warm-up / null-in-window / NaN / window peak), correctness (vs the closed-form reference and a
frozen golden master), and properties (reference agreement for any input and under missing data, scale invariance).
Categories are split
into classes; cross-cutting categories use markers.
"""

import math

import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st
from tests.metrics.oracles import drawdown_rolling_reference
from tests.support import (
    ABSOLUTE_TOLERANCE_REFERENCE,
    COLUMN_X,
    RELATIVE_TOLERANCE_REFERENCE,
    WINDOW_MAX,
    apply_expr,
    assert_matches,
    positive_missing_data,
)

from pomata.metrics import drawdown_rolling

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- drawdown_rolling is WINDOWED and series-valued. Facts (mirroring the windowed indicators):
#   1. shape   length-preserving: one output row per input row; the first ``window - 1`` rows are warm-up ``null``
#   2. domain  positive equities (a growth factor is > 0); the missing variant mixes null / NaN
#   3. window  window_min = 1 (a single-row window is its own peak, so the drawdown is 0) .. WINDOW_MAX
# Each case carries (window - 1) warm-up rows + a window of defined output, so no example is all warm-up. The endpoint
# and rolling-max operations are exact, so the implementation matches the oracle to the reference tolerance directly.
# ----------------------------------------------------------------------------------------------------------------------
_EQUITY = st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False)


@st.composite
def _cases[T](draw: st.DrawFn, values: st.SearchStrategy[T], window_min: int = 1) -> tuple[list[T], int]:
    """A (series, window) pair sized so every example has a window of defined output past the warm-up."""
    window = draw(st.integers(min_value=window_min, max_value=WINDOW_MAX))
    defined = draw(st.integers(min_value=window, max_value=2 * window))
    length = (window - 1) + defined
    return draw(st.lists(values, min_size=length, max_size=length)), window


class TestDrawdownRollingContract:
    """
    Type, shape, and lazy/eager guarantees.
    """


class TestDrawdownRollingEdge:
    """
    Validation, boundaries, and null / NaN handling.
    """

    def test_window_below_one_raises(self) -> None:
        """
        Verifies that ``window < 1`` raises ``ValueError``.
        """
        with pytest.raises(ValueError, match="window must be >= 1"):
            drawdown_rolling(pl.col(COLUMN_X), 0)

    def test_warmup_null_count(self) -> None:
        """
        Verifies that the first ``window - 1`` rows are ``null`` and the rest match the reference.
        """
        values = [1.0, 1.1, 1.05, 1.2, 1.15]
        assert_matches(
            apply_expr(values, drawdown_rolling(pl.col(COLUMN_X), 3)),
            drawdown_rolling_reference(values, 3),
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
        )

    def test_null_in_window_is_null(self) -> None:
        """
        Verifies that a window containing a ``null`` yields ``null`` (the window must hold ``window`` non-null values).
        """
        values = [1.0, None, 1.05, 1.2, 1.1]
        assert_matches(
            apply_expr(values, drawdown_rolling(pl.col(COLUMN_X), 3)),
            drawdown_rolling_reference(values, 3),
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
        )

    def test_nan_propagates(self) -> None:
        """
        Verifies that a ``NaN`` inside a window propagates to ``NaN`` for the windows that touch it.
        """
        values = [1.0, math.nan, 1.05, 1.2, 1.1]
        assert_matches(
            apply_expr(values, drawdown_rolling(pl.col(COLUMN_X), 3)),
            drawdown_rolling_reference(values, 3),
        )

    def test_window_peak_is_zero(self) -> None:
        """
        Verifies that at a monotonically rising window's peak (the current equity) the drawdown is ``0``.
        """
        assert_matches(
            apply_expr([1.0, 1.1, 1.2, 1.3], drawdown_rolling(pl.col(COLUMN_X), 3)),
            [None, None, 0.0, 0.0],
        )


class TestDrawdownRollingCorrectness:
    """
    Against the naive reference oracle and frozen golden-master values.
    """

    def test_matches_reference(self) -> None:
        """
        Verifies agreement with the naive closed-form reference over a representative equity curve.
        """
        values = [1.0, 1.05, 1.2, 1.1, 1.3, 0.95, 1.0, 1.4]
        assert_matches(
            apply_expr(values, drawdown_rolling(pl.col(COLUMN_X), 4)),
            drawdown_rolling_reference(values, 4),
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference: the rolling drawdown over a seven-bar equity curve with a window of three.
        """
        values = [1.0, 1.1, 1.05, 1.2, 1.15, 1.3, 1.25]
        assert_matches(
            apply_expr(values, drawdown_rolling(pl.col(COLUMN_X), 3).round(4)),
            [None, None, -0.0455, 0.0, -0.0417, 0.0, -0.0385],
        )


class TestDrawdownRollingProperties:
    """
    Invariants that must hold for all inputs (property-based).
    """

    @given(case=_cases(_EQUITY))
    def test_matches_reference_for_any_input(self, case: tuple[list[float], int]) -> None:
        """
        Verifies that, for any positive equity series and window, the implementation matches the naive reference.
        """
        values, window = case
        assert_matches(
            apply_expr(values, drawdown_rolling(pl.col(COLUMN_X), window)),
            drawdown_rolling_reference(values, window),
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    @given(case=_cases(positive_missing_data(high=1e4)))
    def test_matches_reference_under_missing_data(self, case: tuple[list[float | None], int]) -> None:
        """
        Verifies that, for inputs freely mixing null / NaN / finite, the implementation matches the naive reference.
        """
        values, window = case
        assert_matches(
            apply_expr(values, drawdown_rolling(pl.col(COLUMN_X), window)),
            drawdown_rolling_reference(values, window),
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    @given(case=_cases(_EQUITY), exponent=st.sampled_from([-4, -3, -2, -1, 1, 2, 3, 4]))
    def test_scale_invariance(self, case: tuple[list[float], int], exponent: int) -> None:
        """
        Verifies that ``drawdown_rolling`` is scale-invariant: scaling every input value by a constant ``k`` leaves
        the output unchanged -- ``drawdown_rolling(k * x) == drawdown_rolling(x)``. ``k`` is a power of two, so the
        rescale is exact and adds no floating-point error.
        """
        values, window = case
        k = 2.0**exponent
        base = apply_expr(values, drawdown_rolling(pl.col(COLUMN_X), window))
        scaled = apply_expr([value * k for value in values], drawdown_rolling(pl.col(COLUMN_X), window))
        assert_matches(scaled, base, rel_tol=RELATIVE_TOLERANCE_REFERENCE, abs_tol=ABSOLUTE_TOLERANCE_REFERENCE)
