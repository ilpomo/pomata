"""
Tests for ``pomata.metrics.tail_ratio_rolling`` — the rolling (windowed) twin of :func:`pomata.metrics.tail_ratio`.

``tail_ratio_rolling`` is single-input and WINDOWED-SERIES-VALUED (a return series → a series the same length, one value
per trailing window), so tests use the shared ``apply_expr`` helper; ``assert_matches`` and the naive
``tail_ratio_rolling_reference`` oracle (the reducing :func:`tail_ratio` recomputed over each window) are shared across
the suite. The rolling null/NaN policy differs from the reducing one: a window holding any ``null`` is ``null`` (it must
hold ``window`` non-null values), and a ``NaN`` inside a window propagates.

The ladder is the canonical one: contract (type / length-preserving / lazy-eager / ``.over`` per-group warm-up), edge
(validation / empty / warm-up / null-in-window / NaN), correctness (vs the closed-form reference and a frozen golden
master), and properties (reference agreement for any input and under missing data). It is a scale-invariant ratio of two
quantiles, so it matches the oracle tightly. Categories are split into classes; cross-cutting categories use markers.
"""

import math

import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st
from tests.metrics.oracles import tail_ratio_rolling_reference
from tests.support import (
    ABSOLUTE_TOLERANCE_REFERENCE,
    COLUMN_X,
    RELATIVE_TOLERANCE_REFERENCE,
    RELATIVE_TOLERANCE_SCALE,
    WINDOW_MAX,
    apply_expr,
    assert_matches,
    missing_data_floats,
    subnormal_safe_floats,
)

from pomata.metrics import tail_ratio_rolling

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- tail_ratio_rolling is WINDOWED and series-valued. Facts (mirroring the windowed indicators):
#   1. shape   length-preserving: one output row per input row; the first ``window - 1`` rows are warm-up ``null``
#   2. domain  subnormal_safe_floats: finite returns floored off subnormal so the power-of-two rescale stays bit-exact;
#              the missing variant mixes null / NaN
#   3. window  window_min = 1 (a single observation has equal tails, so the ratio is 1) .. WINDOW_MAX
# Each case carries (window - 1) warm-up rows + a window of defined output, so no example is all warm-up. As a ratio of
# two quantiles the one-pass primitive agrees with the two-pass oracle to the reference band (no per-window guard).
# ----------------------------------------------------------------------------------------------------------------------


@st.composite
def _cases[T](draw: st.DrawFn, values: st.SearchStrategy[T], window_min: int = 1) -> tuple[list[T], int]:
    """A (series, window) pair sized so every example has a window of defined output past the warm-up."""
    window = draw(st.integers(min_value=window_min, max_value=WINDOW_MAX))
    defined = draw(st.integers(min_value=window, max_value=2 * window))
    length = (window - 1) + defined
    return draw(st.lists(values, min_size=length, max_size=length)), window


class TestTailRatioRollingContract:
    """
    Type, shape, and lazy/eager guarantees.
    """


class TestTailRatioRollingEdge:
    """
    Validation, boundaries, and null / NaN handling.
    """

    def test_window_below_one_raises(self) -> None:
        """
        Verifies that ``window < 1`` raises ``ValueError``.
        """
        with pytest.raises(ValueError, match="window must be >= 1"):
            tail_ratio_rolling(pl.col(COLUMN_X), 0)

    def test_null_in_window_is_null(self) -> None:
        """
        Verifies that a window containing a ``null`` yields ``null`` (the window must hold ``window`` non-null values).
        """
        values = [0.01, None, 0.03, -0.01, 0.02]
        assert_matches(
            apply_expr(values, tail_ratio_rolling(pl.col(COLUMN_X), 3)),
            tail_ratio_rolling_reference(values, 3),
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
        )

    def test_nan_propagates(self) -> None:
        """
        Verifies that a ``NaN`` inside a window propagates to ``NaN`` for the windows that touch it.
        """
        values = [0.01, math.nan, 0.03, -0.01, 0.02]
        assert_matches(
            apply_expr(values, tail_ratio_rolling(pl.col(COLUMN_X), 3)),
            tail_ratio_rolling_reference(values, 3),
        )

    def test_warmup_null_count(self) -> None:
        """
        Verifies that the first ``window - 1`` rows are ``null`` and the rest match the reference.
        """
        values = [0.01, -0.02, 0.03, -0.01, 0.02]
        assert_matches(
            apply_expr(values, tail_ratio_rolling(pl.col(COLUMN_X), 3)),
            tail_ratio_rolling_reference(values, 3),
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
        )

    def test_window_exceeds_length(self) -> None:
        """
        Verifies that a window exceeding the series length yields an all-null output.
        """
        values = [0.01, -0.02, 0.03, -0.01, 0.02]
        assert_matches(apply_expr(values, tail_ratio_rolling(pl.col(COLUMN_X), 7)), [None, None, None, None, None])

    def test_window_equals_length(self) -> None:
        """
        Verifies that when ``window`` equals the series length only the last row is defined, matching the reference.
        """
        values = [0.01, -0.02, 0.03, -0.01, 0.02]
        assert_matches(
            apply_expr(values, tail_ratio_rolling(pl.col(COLUMN_X), 5)),
            tail_ratio_rolling_reference(values, 5),
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
        )

    def test_zero_left_tail_window_is_inf(self) -> None:
        """
        Verifies that a window whose 5th-percentile return is zero with a non-zero 95th-percentile gives ``+inf``
        (reported, not clipped).
        """
        assert_matches(
            apply_expr([0.0, 0.0, 0.0, 0.0, 0.02], tail_ratio_rolling(pl.col(COLUMN_X), 5)),
            [None, None, None, None, math.inf],
        )

    def test_all_zero_window_is_nan(self) -> None:
        """
        Verifies that an all-zero window gives ``0 / 0``, so the ratio is ``NaN``.
        """
        assert_matches(apply_expr([0.0, 0.0, 0.0], tail_ratio_rolling(pl.col(COLUMN_X), 3)), [None, None, math.nan])


class TestTailRatioRollingCorrectness:
    """
    Against the naive reference oracle and frozen golden-master values.
    """

    def test_matches_reference(self) -> None:
        """
        Verifies agreement with the naive closed-form reference over a representative series.
        """
        values = [0.012, -0.008, 0.02, -0.015, 0.005, 0.0, -0.02, 0.018]
        assert_matches(
            apply_expr(values, tail_ratio_rolling(pl.col(COLUMN_X), 4)),
            tail_ratio_rolling_reference(values, 4),
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference: the rolling tail ratio over a window of five.
        """
        values = [0.01, -0.02, 0.03, -0.01, 0.02, 0.0, -0.015]
        assert_matches(
            apply_expr(values, tail_ratio_rolling(pl.col(COLUMN_X), 5).round(4)),
            [None, None, None, None, 1.5556, 1.5556, 2.0],
        )


class TestTailRatioRollingProperties:
    """
    Invariants that must hold for all inputs (property-based).
    """

    @given(case=_cases(subnormal_safe_floats(bound=1e3)))
    def test_matches_reference_for_any_input(self, case: tuple[list[float], int]) -> None:
        """
        Verifies that, for any return series and window, the implementation matches the naive reference.
        """
        values, window = case
        assert_matches(
            apply_expr(values, tail_ratio_rolling(pl.col(COLUMN_X), window)),
            tail_ratio_rolling_reference(values, window),
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    @given(case=_cases(missing_data_floats(min_magnitude=1e-3)))
    def test_matches_reference_under_missing_data(self, case: tuple[list[float | None], int]) -> None:
        """
        Verifies that, for inputs freely mixing null / NaN / finite, the implementation matches the naive reference.
        """
        values, window = case
        assert_matches(
            apply_expr(values, tail_ratio_rolling(pl.col(COLUMN_X), window)),
            tail_ratio_rolling_reference(values, window),
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    @given(case=_cases(subnormal_safe_floats(bound=1e3)), exponent=st.sampled_from([-4, -3, -2, -1, 1, 2, 3, 4]))
    def test_scale_invariance(self, case: tuple[list[float], int], exponent: int) -> None:
        """
        Verifies that ``tail_ratio_rolling`` is scale-invariant: scaling every input value by a constant ``k``
        leaves the output unchanged -- ``tail_ratio_rolling(k * x) == tail_ratio_rolling(x)``. ``k`` is a power of
        two, so the rescale is exact and adds no floating-point error.
        """
        values, window = case
        k = 2.0**exponent
        base = apply_expr(values, tail_ratio_rolling(pl.col(COLUMN_X), window))
        scaled = apply_expr([value * k for value in values], tail_ratio_rolling(pl.col(COLUMN_X), window))
        assert_matches(scaled, base, rel_tol=RELATIVE_TOLERANCE_SCALE, abs_tol=ABSOLUTE_TOLERANCE_REFERENCE)
