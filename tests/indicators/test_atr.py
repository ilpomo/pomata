"""
Tests for ``pomata.indicators.atr`` — Wilder's Average True Range.

Categories are split into classes; cross-cutting categories elsewhere use markers (see ``tests/README.md``).
"""

import math
from collections.abc import Sequence

import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st
from polars.testing import assert_frame_equal
from tests.indicators.oracles import atr_reference
from tests.support import (
    ABSOLUTE_TOLERANCE_REFERENCE,
    BOUND_MARGIN,
    CLOSE,
    EXACT_TOLERANCE_FACTOR,
    GROUP_KEY,
    HIGH,
    LOW,
    RELATIVE_TOLERANCE_PROPERTY,
    RELATIVE_TOLERANCE_REFERENCE,
    RELATIVE_TOLERANCE_SCALE,
    WINDOW_MAX,
    assert_matches,
    assert_scale_homogeneous,
    coherent_hlc,
    coherent_hlc_with_missing,
    count_leading_nulls,
    input_scale,
    materialize,
    split_triples,
)

from pomata.indicators import atr

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- derived, not chosen; the rationale is the shared method in CORRECTNESS.md, the numbers are this
# indicator's. To add an indicator, set its three facts here; the property tier below is then the same shape as every
# other indicator's.
#   1. warmup  W(window) = window - 1   (the rma over the true-range series emits only once ``window`` non-null true
#              ranges have accrued; the true range itself is defined from row 0)
#   2. memory  the oracle shares pomata's recursive Wilder seeding, so the property holds from the first defined row
#              (M = 0); each example carries D in [window, 2 * window] defined bars -- one window of output, never all
#              warm-up
#   3. domain  coherent_hlc(): coherent (high >= low, low <= close <= high) positive-finite bars -- the ATR is only
#              non-negative on well-formed OHLC; windows span 1 .. WINDOW_MAX
# Repetitions N are the shared CI profile (tests/conftest.py); override per-test only if its parameter space is larger.
# ----------------------------------------------------------------------------------------------------------------------


@st.composite
def _cases[T](draw: st.DrawFn, bars: st.SearchStrategy[T]) -> tuple[list[T], int]:
    """
    A (series, window) pair sized from the facts above: ``window`` over its regimes, length = warm-up + a window of
    defined bars, so every example has output to check (never an all-warm-up series, the waste a ``window`` decoupled
    from the length would cause).
    """
    window = draw(st.integers(min_value=1, max_value=WINDOW_MAX))
    defined = draw(st.integers(min_value=window, max_value=2 * window))
    length = (window - 1) + defined
    return draw(st.lists(bars, min_size=length, max_size=length)), window


def apply_atr(
    high: Sequence[float | None],
    low: Sequence[float | None],
    close: Sequence[float | None],
    window: int,
) -> list[float | None]:
    """
    Materialize ``atr`` over a three-column ``Float64`` frame built from the aligned ``high``, ``low``, and ``close``
    lists.
    """
    return materialize({HIGH: high, LOW: low, CLOSE: close}, atr(pl.col(HIGH), pl.col(LOW), pl.col(CLOSE), window))


class TestAtrContract:
    """
    Type, shape, and lazy/eager guarantees.
    """

    def test_returns_expr(self) -> None:
        """
        Verifies that the factory returns a ``pl.Expr`` without touching a frame.
        """
        assert isinstance(atr(pl.col(HIGH), pl.col(LOW), pl.col(CLOSE), 3), pl.Expr)

    def test_preserves_length_and_dtype(self) -> None:
        """
        Verifies that the output has one value per input row and is ``Float64``.
        """
        frame = pl.DataFrame(
            {
                HIGH: [10.0, 12.0, 11.0, 13.0, 15.0],
                LOW: [8.0, 9.0, 9.5, 10.0, 12.0],
                CLOSE: [9.0, 11.0, 10.0, 12.0, 14.0],
            }
        )
        result = frame.select(atr(pl.col(HIGH), pl.col(LOW), pl.col(CLOSE), 3).alias("y"))
        assert result.height == frame.height
        assert result.schema["y"] == pl.Float64

    def test_empty_frame_preserves_shape_and_dtype(self) -> None:
        """
        Verifies that an empty input yields an empty ``Float64`` column rather than raising.
        """
        frame = pl.DataFrame(
            {
                HIGH: pl.Series(HIGH, [], dtype=pl.Float64),
                LOW: pl.Series(LOW, [], dtype=pl.Float64),
                CLOSE: pl.Series(CLOSE, [], dtype=pl.Float64),
            }
        )
        result = frame.select(atr(pl.col(HIGH), pl.col(LOW), pl.col(CLOSE), 3).alias("y"))
        assert result.height == 0
        assert result.schema["y"] == pl.Float64

    def test_lazy_eager_parity(self) -> None:
        """
        Verifies that eager and lazy application produce identical materialized output.
        """
        frame = pl.DataFrame(
            {
                HIGH: [10.0, 12.0, 11.0, 13.0, 15.0, 14.0],
                LOW: [8.0, 9.0, 9.5, 10.0, 12.0, 11.0],
                CLOSE: [9.0, 11.0, 10.0, 12.0, 14.0, 13.0],
            }
        )
        expr = atr(pl.col(HIGH), pl.col(LOW), pl.col(CLOSE), 3).alias("y")
        result_eager = frame.select(expr)
        result_lazy = frame.lazy().select(expr).collect()
        assert_frame_equal(result_eager, result_lazy)

    def test_over_partitions_independently(self) -> None:
        """
        Verifies that under ``.over`` the previous-close shift and Wilder recursion reset per group.
        """
        high_a = [10.0, 12.0, 11.0, 13.0, 15.0]
        low_a = [8.0, 9.0, 9.5, 10.0, 12.0]
        close_a = [9.0, 11.0, 10.0, 12.0, 14.0]
        high_b = [100.0, 120.0, 110.0, 130.0, 150.0]
        low_b = [80.0, 90.0, 95.0, 100.0, 120.0]
        close_b = [90.0, 110.0, 100.0, 120.0, 140.0]
        frame = pl.DataFrame(
            {
                GROUP_KEY: ["a"] * 5 + ["b"] * 5,
                HIGH: high_a + high_b,
                LOW: low_a + low_b,
                CLOSE: close_a + close_b,
            }
        )
        result_over = frame.select(atr(pl.col(HIGH), pl.col(LOW), pl.col(CLOSE), 2).over(GROUP_KEY).alias("y"))[
            "y"
        ].to_list()
        result_a = apply_atr(high_a, low_a, close_a, 2)
        result_b = apply_atr(high_b, low_b, close_b, 2)
        assert_matches(result_over, result_a + result_b)


class TestAtrEdge:
    """
    Boundaries, warm-up, and null / NaN handling.
    """

    def test_window_below_one_raises(self) -> None:
        """
        Verifies that ``window < 1`` raises ``ValueError``.
        """
        with pytest.raises(ValueError, match="window must be >= 1"):
            atr(pl.col(HIGH), pl.col(LOW), pl.col(CLOSE), 0)

    def test_warmup_null_count(self) -> None:
        """
        Verifies that the first ``window - 1`` rows are null and the first full window is defined.
        """
        result = apply_atr(
            [10.0, 12.0, 11.0, 13.0, 15.0, 14.0],
            [8.0, 9.0, 9.5, 10.0, 12.0, 11.0],
            [9.0, 11.0, 10.0, 12.0, 14.0, 13.0],
            3,
        )
        assert result[:2] == [None, None]
        assert result[2] is not None

    def test_window_one_is_true_range(self) -> None:
        """
        Verifies that ``window == 1`` reproduces the true range (the Wilder smoothing is the identity).
        """
        high = [10.0, 12.0, 11.0, 13.0]
        low = [8.0, 9.0, 9.5, 10.0]
        close = [9.0, 11.0, 10.0, 12.0]
        assert_matches(apply_atr(high, low, close, 1), atr_reference(high, low, close, 1))
        assert_matches(apply_atr(high, low, close, 1), [2.0, 3.0, 1.5, 3.0])

    def test_window_equals_length(self) -> None:
        """
        Verifies the single defined value when ``window`` equals the series length.
        """
        high = [10.0, 12.0, 11.0]
        low = [8.0, 9.0, 9.5]
        close = [9.0, 11.0, 10.0]
        assert_matches(apply_atr(high, low, close, 3), atr_reference(high, low, close, 3))
        assert_matches(apply_atr(high, low, close, 3), [None, None, 2.1666666666666665])

    def test_single_row(self) -> None:
        """
        Verifies behavior on a one-element series: the lone true range degenerates to ``high - low``.
        """
        assert_matches(apply_atr([10.0], [8.0], [9.0], 1), [2.0])
        assert_matches(apply_atr([10.0], [8.0], [9.0], 2), [None])

    def test_window_exceeds_length(self) -> None:
        """
        Verifies that a window longer than the series yields an all-null result (the warm-up never completes).
        """
        assert_matches(apply_atr([10.0, 12.0, 13.0], [8.0, 9.0, 10.0], [9.0, 11.0, 12.0], 5), [None, None, None])

    def test_empty(self) -> None:
        """
        Verifies that an empty series yields an empty result.
        """
        assert_matches(apply_atr([], [], [], 3), [])

    def test_interior_null_in_high_is_absorbed(self) -> None:
        """
        Verifies that a null in only ``high`` drops the ``high - low`` and ``|high - prev_close|`` candidate terms, so
        ``max_horizontal`` falls back to ``|low - prev_close|`` and the true range stays non-null at that bar.
        """
        high = [10.0, 12.0, None, 13.0, 15.0]
        low = [8.0, 9.0, 9.5, 10.0, 12.0]
        close = [9.0, 11.0, 10.0, 12.0, 14.0]
        assert_matches(apply_atr(high, low, close, 2), atr_reference(high, low, close, 2))
        assert_matches(apply_atr(high, low, close, 2), [None, 2.5, 2.0, 2.5, 2.75])

    def test_interior_null_in_low_changes_result(self) -> None:
        """
        Verifies that a null in only ``low`` (vs in ``high``) yields a different, smaller true range: it drops
        ``high - low`` and ``|low - prev_close|``, leaving ``|high - prev_close| = 0`` here, distinguishing the
        per-input roles.
        """
        high = [10.0, 12.0, 11.0, 13.0, 15.0]
        low = [8.0, 9.0, None, 10.0, 12.0]
        close = [9.0, 11.0, 10.0, 12.0, 14.0]
        assert_matches(apply_atr(high, low, close, 2), atr_reference(high, low, close, 2))
        assert_matches(apply_atr(high, low, close, 2), [None, 2.5, 1.25, 2.125, 2.5625])

    def test_interior_null_in_close_affects_next_bar(self) -> None:
        """
        Verifies that a null in only ``close`` leaves the current bar's ``high - low`` term intact but drops the two
        gap terms of the following bar (whose previous close is null), so the next true range degenerates to its
        ``high - low``.
        """
        high = [10.0, 12.0, 11.0, 13.0, 15.0]
        low = [8.0, 9.0, 9.5, 10.0, 12.0]
        close = [9.0, 11.0, None, 12.0, 14.0]
        assert_matches(apply_atr(high, low, close, 2), atr_reference(high, low, close, 2))
        assert_matches(apply_atr(high, low, close, 2), [None, 2.5, 2.0, 2.5, 2.75])

    def test_leading_null_run_defers_warmup(self) -> None:
        """
        Verifies that a leading all-null run does not consume warm-up budget: the rma counts only non-null true ranges,
        so the first defined ATR appears once ``window`` non-null true ranges have accrued.
        """
        high = [None, None, 11.0, 13.0, 15.0]
        low = [None, None, 9.5, 10.0, 12.0]
        close = [None, None, 10.0, 12.0, 14.0]
        assert_matches(apply_atr(high, low, close, 2), atr_reference(high, low, close, 2))
        assert_matches(apply_atr(high, low, close, 2), [None, None, None, 2.25, 2.625])

    def test_all_null_is_all_null(self) -> None:
        """
        Verifies that an all-null OHLC frame yields an all-null result (every true range is null, the rma never seeds).
        """
        high = [None, None, None]
        low = [None, None, None]
        close = [None, None, None]
        assert_matches(apply_atr(high, low, close, 2), atr_reference(high, low, close, 2))
        assert_matches(apply_atr(high, low, close, 2), [None, None, None])

    def test_all_nan_latches(self) -> None:
        """
        Verifies that an all-NaN OHLC frame poisons every true range, so the result is NaN past the warm-up.
        """
        high = [math.nan, math.nan, math.nan, math.nan]
        low = [math.nan, math.nan, math.nan, math.nan]
        close = [math.nan, math.nan, math.nan, math.nan]
        assert_matches(apply_atr(high, low, close, 2), atr_reference(high, low, close, 2))
        assert_matches(apply_atr(high, low, close, 2), [None, math.nan, math.nan, math.nan])

    def test_nan_propagates(self) -> None:
        """
        Verifies that a ``NaN`` input poisons the true range and latches for every subsequent value.
        """
        high = [10.0, 12.0, 11.0, 13.0, 15.0]
        low = [8.0, 9.0, 9.5, 10.0, 12.0]
        close = [9.0, math.nan, 10.0, 12.0, 14.0]
        assert_matches(apply_atr(high, low, close, 2), atr_reference(high, low, close, 2))
        assert_matches(apply_atr(high, low, close, 2), [None, 2.5, math.nan, math.nan, math.nan])

    def test_constant_bars_have_zero_atr(self) -> None:
        """
        Verifies that identical high == low == close bars give a zero true range and hence a zero ATR after warm-up.
        """
        assert_matches(
            apply_atr([5.0] * 6, [5.0] * 6, [5.0] * 6, 3),
            [None, None, 0.0, 0.0, 0.0, 0.0],
        )


class TestAtrCorrectness:
    """
    Against the naive reference oracle and frozen golden-master values.
    """

    def test_matches_reference(self) -> None:
        """
        Verifies agreement with the naive closed-form reference across several windows.
        """
        high = [10.0, 12.0, 11.0, 13.0, 15.0, 14.0, 16.0, 18.0]
        low = [8.0, 9.0, 9.5, 10.0, 12.0, 11.0, 13.0, 15.0]
        close = [9.0, 11.0, 10.0, 12.0, 14.0, 13.0, 15.0, 17.0]
        for window in (1, 2, 3, 4, 5):
            assert_matches(
                apply_atr(high, low, close, window),
                atr_reference(high, low, close, window),
                rel_tol=RELATIVE_TOLERANCE_REFERENCE,
                abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
            )

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference: ATR(window=3) over an eight-bar OHLC series.
        """
        assert_matches(
            apply_atr(
                [10.0, 12.0, 11.0, 13.0, 15.0, 14.0, 16.0, 18.0],
                [8.0, 9.0, 9.5, 10.0, 12.0, 11.0, 13.0, 15.0],
                [9.0, 11.0, 10.0, 12.0, 14.0, 13.0, 15.0, 17.0],
                3,
            ),
            [
                None,
                None,
                2.1666666666666665,
                2.444444444444444,
                2.6296296296296293,
                2.753086419753086,
                2.8353909465020575,
                2.8902606310013716,
            ],
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )


class TestAtrProperties:
    """
    Invariants that must hold for all inputs (property-based).
    """

    @given(case=_cases(coherent_hlc()))
    def test_matches_reference_for_any_input(
        self,
        case: tuple[list[tuple[float, float, float]], int],
    ) -> None:
        """
        Verifies that, for any well-formed OHLC series and window, the implementation matches the naive reference.
        """
        rows, window = case
        high, low, close = split_triples(rows)
        assert_matches(
            apply_atr(high, low, close, window),
            atr_reference(high, low, close, window),
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=input_scale(high) * EXACT_TOLERANCE_FACTOR,
        )

    @given(
        case=_cases(coherent_hlc()),
        exponent=st.sampled_from([-4, -3, -2, -1, 1, 2, 3, 4]),
    )
    def test_scale_homogeneity_positive(
        self,
        case: tuple[list[tuple[float, float, float]], int],
        exponent: int,
    ) -> None:
        """
        Verifies that ATR is homogeneous of degree 1 under a positive scale: ``atr(k * ohlc) == k * atr(ohlc)``. ``k``
        is a power of two so the rescaling is lossless and cannot perturb the Wilder recursion through rounding.
        """
        k = 2.0**exponent
        rows, window = case
        high, low, close = split_triples(rows)
        scaled_high = [value * k for value in high]
        result_base = apply_atr(high, low, close, window)
        result_scaled = apply_atr(scaled_high, [value * k for value in low], [value * k for value in close], window)
        assert_scale_homogeneous(result_scaled, result_base, k=k, degree=1)

    @given(case=_cases(coherent_hlc()))
    def test_non_negative(
        self,
        case: tuple[list[tuple[float, float, float]], int],
    ) -> None:
        """
        Verifies that the ATR of a well-formed OHLC series (``high >= low``) is never negative.
        """
        rows, window = case
        high, low, close = split_triples(rows)
        result = apply_atr(high, low, close, window)
        for value in result:
            if value is not None:
                assert value >= -BOUND_MARGIN

    @given(case=_cases(coherent_hlc()))
    def test_warmup_null_count_property(
        self,
        case: tuple[list[tuple[float, float, float]], int],
    ) -> None:
        """
        Verifies that the leading-null run is exactly ``min(window - 1, len(rows))`` for clean OHLC input.
        """
        rows, window = case
        high, low, close = split_triples(rows)
        result = apply_atr(high, low, close, window)
        leading_nulls = count_leading_nulls(result)
        # NOTE: ``_cases`` couples length >= window, so ``min`` always resolves to ``window - 1``; the form is kept
        # to state the exact warm-up rule (the leading-null run is never clamped by a too-short series here).
        assert leading_nulls == min(window - 1, len(rows))

    @given(case=_cases(coherent_hlc_with_missing()))
    def test_matches_reference_under_missing_data(
        self,
        case: tuple[list[tuple[float | None, float | None, float | None]], int],
    ) -> None:
        """
        Verifies that, for inputs freely mixing null / NaN / finite, the implementation matches the naive reference.
        """
        rows, window = case
        high, low, close = split_triples(rows)
        assert_matches(
            apply_atr(high, low, close, window),
            atr_reference(high, low, close, window),
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=input_scale(high) * EXACT_TOLERANCE_FACTOR,
        )

    @given(
        case=_cases(coherent_hlc()),
        scale=st.sampled_from([1e-6, 1e6, 1e9]),
    )
    def test_matches_reference_at_large_magnitude(
        self,
        case: tuple[list[tuple[float, float, float]], int],
        scale: float,
    ) -> None:
        """
        Verifies that at extreme positive magnitudes the implementation stays finite where the reference is and agrees.
        """
        rows, window = case
        high = [row[0] * scale for row in rows]
        low = [row[1] * scale for row in rows]
        close = [row[2] * scale for row in rows]
        assert_matches(
            apply_atr(high, low, close, window),
            atr_reference(high, low, close, window),
            rel_tol=RELATIVE_TOLERANCE_SCALE,
            abs_tol=input_scale(high) * EXACT_TOLERANCE_FACTOR,
        )
