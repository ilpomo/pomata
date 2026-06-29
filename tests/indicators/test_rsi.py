"""
Tests for ``pomata.indicators.rsi`` — Wilder's Relative Strength Index.

Categories are split into classes; cross-cutting categories elsewhere use markers (see ``tests/README.md``).
"""

import math

import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st
from polars.testing import assert_frame_equal
from tests.indicators.oracles import rsi_reference
from tests.support import (
    ABSOLUTE_TOLERANCE_REFERENCE,
    BOUND_MARGIN,
    COLUMN_X,
    GROUP_KEY,
    RELATIVE_TOLERANCE_PROPERTY,
    RELATIVE_TOLERANCE_REFERENCE,
    apply_expr,
    assert_matches,
    assert_scale_homogeneous,
    count_leading_nulls,
    missing_data_floats,
)

from pomata.indicators import rsi

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- derived, not chosen; the rationale is the shared method in CORRECTNESS.md, the numbers are this
# indicator's. To add an indicator, set its three facts here; the property tier below is then the same shape as every
# other indicator's.
#   1. warmup  W(window) = window   (Wilder's recursion needs ``window`` non-null one-step changes -- ``window + 1``
#              prices -- before emitting; the warm-up counts only non-null observations)
#   2. memory  the oracle shares pomata's recursive Wilder seeding, so the property holds from the first defined row
#              (M = 0); each example carries D in [window, 2 * window] defined values -- one window of output, never all
#              warm-up
#   3. domain  finite floats; RSI is scale-INVARIANT, so the scale tier draws strictly positive values (so a power-of-
#              two rescaling cannot flip a sign) and the magnitude is widened per test below
# RSI is a bounded scale-INVARIANT ratio in ``[0, 100]``: its value is O(1) whatever the input magnitude, so its
# tolerance is ABSOLUTE (never input_scale-sized), and it carries a scale-INVARIANCE property in place of the
# homogeneity / large-magnitude tests of a scale-dependent indicator -- a large-magnitude test would be vacuous because
# the common scale cancels in the ratio. Repetitions N are the shared CI profile (tests/conftest.py); override per-test
# only if its parameter space is larger.
# ----------------------------------------------------------------------------------------------------------------------
WINDOW_MAX = 50


@st.composite
def _cases[T](draw: st.DrawFn, values: st.SearchStrategy[T]) -> tuple[list[T], int]:
    """
    A (series, window) pair sized from the facts above: ``window`` over its regimes, length = warm-up + a window of
    defined values, so every example has output to check (never an all-warm-up series, the waste a ``window`` decoupled
    from the length would cause).
    """
    window = draw(st.integers(min_value=1, max_value=WINDOW_MAX))
    defined = draw(st.integers(min_value=window, max_value=2 * window))
    length = window + defined
    return draw(st.lists(values, min_size=length, max_size=length)), window


class TestRsiContract:
    """
    Type, shape, and lazy/eager guarantees.
    """

    def test_returns_expr(self) -> None:
        """
        Verifies that the factory returns a ``pl.Expr`` without touching a frame.
        """
        assert isinstance(rsi(pl.col(COLUMN_X), 3), pl.Expr)

    def test_preserves_length_and_dtype(self) -> None:
        """
        Verifies that the output has one value per input row and is ``Float64``.
        """
        frame = pl.DataFrame({COLUMN_X: pl.Series(COLUMN_X, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])})
        result = frame.select(rsi(pl.col(COLUMN_X), 2).alias("y"))
        assert result.height == frame.height
        assert result.schema["y"] == pl.Float64

    def test_lazy_eager_parity(self) -> None:
        """
        Verifies that eager and lazy application produce identical materialized output.
        """
        frame = pl.DataFrame({COLUMN_X: pl.Series(COLUMN_X, [44.0, 45.0, 43.0, 46.0, 47.0, 44.0, 48.0])})
        result_eager = frame.select(rsi(pl.col(COLUMN_X), 3).alias("y"))
        result_lazy = frame.lazy().select(rsi(pl.col(COLUMN_X), 3).alias("y")).collect()
        assert_frame_equal(result_eager, result_lazy)

    def test_over_partitions_independently(self) -> None:
        """
        Verifies that under ``.over`` the differencing and recursion reset per group and never span group boundaries.
        """
        values_a = [44.0, 45.0, 43.0, 46.0, 47.0]
        values_b = [10.0, 8.0, 12.0, 6.0, 14.0]
        frame = pl.DataFrame({GROUP_KEY: ["a"] * 5 + ["b"] * 5, COLUMN_X: values_a + values_b})
        result_over = frame.select(rsi(pl.col(COLUMN_X), 2).over(GROUP_KEY).alias("y"))["y"].to_list()
        result_a = apply_expr(values_a, rsi(pl.col(COLUMN_X), 2))
        result_b = apply_expr(values_b, rsi(pl.col(COLUMN_X), 2))
        assert_matches(result_over, result_a + result_b)


class TestRsiEdge:
    """
    Boundaries, warm-up, and null / NaN handling.
    """

    def test_window_below_one_raises(self) -> None:
        """
        Verifies that ``window < 1`` raises ``ValueError``.
        """
        with pytest.raises(ValueError, match="window must be >= 1"):
            rsi(pl.col(COLUMN_X), 0)

    def test_warmup_null_count(self) -> None:
        """
        Verifies that the first ``window`` rows are null and the next is defined (``window + 1`` prices are needed).
        """
        result = apply_expr([2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0], rsi(pl.col(COLUMN_X), 3))
        assert result[:3] == [None, None, None]
        assert result[3] is not None

    def test_window_one(self) -> None:
        """
        Verifies that ``window == 1`` reports ``100`` on an up move, ``0`` on a down move, and ``NaN`` on no move.
        """
        assert_matches(apply_expr([1.0, 3.0, 2.0, 5.0], rsi(pl.col(COLUMN_X), 1)), [None, 100.0, 0.0, 100.0])

    def test_window_equals_length(self) -> None:
        """
        Verifies the whole output is null when ``window`` equals the series length (only ``window`` differences exist).
        """
        assert_matches(apply_expr([1.0, 2.0, 3.0], rsi(pl.col(COLUMN_X), 3)), [None, None, None])

    def test_window_exceeds_length(self) -> None:
        """
        Verifies the whole output is null when ``window`` exceeds the series length.
        """
        assert_matches(apply_expr([1.0, 2.0, 3.0], rsi(pl.col(COLUMN_X), 5)), [None, None, None])

    def test_single_row(self) -> None:
        """
        Verifies behavior on a one-element series (no difference exists, so the value is null).
        """
        assert_matches(apply_expr([42.0], rsi(pl.col(COLUMN_X), 1)), [None])
        assert_matches(apply_expr([42.0], rsi(pl.col(COLUMN_X), 2)), [None])

    def test_empty(self) -> None:
        """
        Verifies that an empty series yields an empty result.
        """
        assert_matches(apply_expr([], rsi(pl.col(COLUMN_X), 3)), [])

    def test_all_null(self) -> None:
        """
        Verifies that an all-null series stays null (no difference ever seeds the recursion).
        """
        assert_matches(apply_expr([None, None, None], rsi(pl.col(COLUMN_X), 2)), [None, None, None])

    def test_constant_series_is_nan(self) -> None:
        """
        Verifies the all-flat convention: a constant series has no gain and no loss, so the relative strength is the
        indeterminate ``0 / 0`` and the RSI is ``NaN`` once warmed up.
        """
        assert_matches(apply_expr([5.0, 5.0, 5.0, 5.0], rsi(pl.col(COLUMN_X), 2)), [None, None, math.nan, math.nan])

    def test_monotone_increasing_is_hundred(self) -> None:
        """
        Verifies that a strictly increasing series (no losses) yields exactly ``100`` (relative strength ``+inf``).
        """
        assert_matches(
            apply_expr([2.0, 4.0, 6.0, 8.0, 10.0], rsi(pl.col(COLUMN_X), 2)),
            [None, None, 100.0, 100.0, 100.0],
        )

    def test_monotone_decreasing_is_zero(self) -> None:
        """
        Verifies that a strictly decreasing series (no gains) yields exactly ``0`` (relative strength ``0``).
        """
        assert_matches(
            apply_expr([10.0, 8.0, 6.0, 4.0, 2.0], rsi(pl.col(COLUMN_X), 2)),
            [None, None, 0.0, 0.0, 0.0],
        )

    def test_leading_null_defers_first_difference(self) -> None:
        """
        Verifies that a leading ``null`` defers the first difference and does not consume warm-up budget early.
        """
        assert_matches(
            apply_expr([None, 2.0, 4.0, 6.0, 8.0], rsi(pl.col(COLUMN_X), 2)),
            [None, None, None, 100.0, 100.0],
        )

    def test_interior_null_bridged(self) -> None:
        """
        Verifies that an interior ``null`` yields ``null`` at its position while the recursion bridges the gap.
        """
        assert_matches(
            apply_expr([2.0, 4.0, None, 8.0, 10.0, 12.0], rsi(pl.col(COLUMN_X), 2)),
            [None, None, None, None, 100.0, 100.0],
        )

    def test_nan_propagates(self) -> None:
        """
        Verifies that a ``NaN`` poisons the recursion and latches for every subsequent value.
        """
        assert_matches(
            apply_expr([1.0, math.nan, 3.0, 4.0, 5.0], rsi(pl.col(COLUMN_X), 2)),
            [None, None, math.nan, math.nan, math.nan],
        )


class TestRsiCorrectness:
    """
    Against the naive reference oracle and frozen golden-master values.
    """

    def test_matches_reference(self) -> None:
        """
        Verifies agreement with the naive closed-form reference across several windows.
        """
        values = [44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08, 45.89, 46.03]
        for window in (1, 2, 3, 5, 7):
            assert_matches(
                apply_expr(values, rsi(pl.col(COLUMN_X), window)),
                rsi_reference(values, window),
                rel_tol=RELATIVE_TOLERANCE_REFERENCE,
                abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
            )

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference: RSI(window=3) over a short price series, against hand-checked golden values.
        """
        assert_matches(
            apply_expr([44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42], rsi(pl.col(COLUMN_X), 3)),
            [
                None,
                None,
                None,
                7.058823529411242,
                59.06735751295326,
                74.1407528641571,
                80.08194138039714,
                85.85813381069593,
            ],
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )


class TestRsiProperties:
    """
    Invariants that must hold for all inputs (property-based).
    """

    @given(case=_cases(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)))
    def test_matches_reference_for_any_input(
        self,
        case: tuple[list[float], int],
    ) -> None:
        """
        Verifies that, for any series and window, the implementation matches the naive reference.
        """
        values, window = case
        assert_matches(
            apply_expr(values, rsi(pl.col(COLUMN_X), window)),
            rsi_reference(values, window),
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    @given(case=_cases(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)))
    def test_bounded_in_unit_interval(
        self,
        case: tuple[list[float], int],
    ) -> None:
        """
        Verifies that every defined, non-NaN RSI value lies in ``[0, 100]`` (a perfectly flat window yields ``NaN``).
        """
        values, window = case
        result = apply_expr(values, rsi(pl.col(COLUMN_X), window))
        for value in result:
            if value is not None and not math.isnan(value):
                assert -BOUND_MARGIN <= value <= 100.0 + BOUND_MARGIN

    @given(case=_cases(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)))
    def test_warmup_null_count_property(
        self,
        case: tuple[list[float], int],
    ) -> None:
        """
        Verifies that the leading-null run is exactly ``min(window, len(values))``.
        """
        values, window = case
        result = apply_expr(values, rsi(pl.col(COLUMN_X), window))
        leading_nulls = count_leading_nulls(result)
        assert leading_nulls == min(window, len(values))

    @given(
        case=_cases(st.floats(min_value=1.0, max_value=1e3, allow_nan=False, allow_infinity=False)),
        exponent=st.sampled_from([-4, -3, -2, -1, 1, 2, 3, 4]),
    )
    def test_scale_invariance(
        self,
        case: tuple[list[float], int],
        exponent: int,
    ) -> None:
        """
        Verifies that RSI is invariant under a positive rescaling of the series: ``rsi(k * x) == rsi(x)`` for ``k > 0``
        (both gains and losses scale by ``k``, so the relative strength — and thus the RSI — is unchanged). ``k`` is a
        power of two so the rescaling is lossless and cannot introduce a floating-point artifact.
        """
        k = 2.0**exponent
        values, window = case
        result_base = apply_expr(values, rsi(pl.col(COLUMN_X), window))
        result_scaled = apply_expr([value * k for value in values], rsi(pl.col(COLUMN_X), window))
        assert_scale_homogeneous(result_scaled, result_base, k=k, degree=0)

    @given(case=_cases(missing_data_floats()))
    def test_matches_reference_under_missing_data(
        self,
        case: tuple[list[float | None], int],
    ) -> None:
        """
        Verifies that, for inputs freely mixing null / NaN / finite, the implementation matches the naive reference.
        """
        values, window = case
        assert_matches(
            apply_expr(values, rsi(pl.col(COLUMN_X), window)),
            rsi_reference(values, window),
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )
