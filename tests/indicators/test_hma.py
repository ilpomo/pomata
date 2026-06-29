"""
Tests for ``pomata.indicators.hma`` — the Hull Moving Average.

Categories are split into classes; cross-cutting categories elsewhere use markers (see ``tests/README.md``).
"""

import math

import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st
from polars.testing import assert_frame_equal
from tests.indicators.oracles import hma_reference
from tests.support import (
    ABSOLUTE_TOLERANCE_REFERENCE,
    ABSOLUTE_TOLERANCE_SCALE,
    COLUMN_X,
    EXACT_TOLERANCE_FACTOR,
    GROUP_KEY,
    RELATIVE_TOLERANCE_PROPERTY,
    RELATIVE_TOLERANCE_REFERENCE,
    RELATIVE_TOLERANCE_SCALE,
    WINDOW_MAX,
    apply_expr,
    assert_matches,
    assert_scale_homogeneous,
    input_scale,
    missing_data_floats,
)

from pomata.indicators import hma

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- derived, not chosen; the rationale is the shared method in CORRECTNESS.md, the numbers are this
# indicator's. To add an indicator, set its three facts here; the property tier below is then the same shape as every
# other indicator's.
#   1. warmup  W(window) = window + s - 2, with the smoothing period s = floor(sqrt(window) + 0.5) (round-half-up): the
#              inner ``WMA(x, window)`` needs ``window`` observations, after which the final ``WMA(., s)`` needs s - 1
#              more. The smallest meaningful window is 2 (the half-period collapses to 1 at window == 1)
#   2. memory  the oracle shares pomata's windowed WMA construction, so the property holds from the first defined row
#              (M = 0); each example carries D in [window, 2 * window] defined values past the warm-up -- always output
#              to check, never an all-warm-up series
#   3. domain  finite floats; HMA is a degree-1 weighted mean (no squaring), so the ordinary finite domain needs no
#              underflow floor; the bound is widened per test below
# Windows span ``window_min`` (2) .. WINDOW_MAX. Repetitions N are the shared CI profile (tests/conftest.py); override
# per-test only if its parameter space is larger.
# ----------------------------------------------------------------------------------------------------------------------


def _warmup(window: int) -> int:
    """
    The HMA leading-null count ``window + floor(sqrt(window) + 0.5) - 2`` (round-half-up smoothing period).
    """
    return window + math.floor(math.sqrt(window) + 0.5) - 2


@st.composite
def _cases[T](
    draw: st.DrawFn,
    values: st.SearchStrategy[T],
    window_min: int = 2,
) -> tuple[list[T], int]:
    """
    A (series, window) pair sized from the facts above: ``window`` over its regimes, length = warm-up + a window of
    defined values, so every example has output to check (never an all-warm-up series, the waste a ``window`` decoupled
    from the length would cause).
    """
    window = draw(st.integers(min_value=window_min, max_value=WINDOW_MAX))
    defined = draw(st.integers(min_value=window, max_value=2 * window))
    length = _warmup(window) + defined
    return draw(st.lists(values, min_size=length, max_size=length)), window


class TestHmaContract:
    """
    Type, shape, and lazy/eager guarantees.
    """

    def test_returns_expr(self) -> None:
        """
        Verifies that the factory returns a ``pl.Expr`` without touching a frame.
        """
        assert isinstance(hma(pl.col(COLUMN_X), 4), pl.Expr)

    def test_preserves_length_and_dtype(self) -> None:
        """
        Verifies that the output has one value per input row and is ``Float64``.
        """
        frame = pl.DataFrame({COLUMN_X: pl.Series(COLUMN_X, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])})
        result = frame.select(hma(pl.col(COLUMN_X), 4).alias("y"))
        assert result.height == frame.height
        assert result.schema["y"] == pl.Float64

    def test_lazy_eager_parity(self) -> None:
        """
        Verifies that eager and lazy application produce identical materialized output.
        """
        frame = pl.DataFrame({COLUMN_X: pl.Series(COLUMN_X, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])})
        result_eager = frame.select(hma(pl.col(COLUMN_X), 4).alias("y"))
        result_lazy = frame.lazy().select(hma(pl.col(COLUMN_X), 4).alias("y")).collect()
        assert_frame_equal(result_eager, result_lazy)

    def test_over_partitions_independently(self) -> None:
        """
        Verifies that under ``.over`` the windows reset per group and never span group boundaries.
        """
        frame = pl.DataFrame(
            {
                GROUP_KEY: ["a", "a", "a", "a", "a", "a", "b", "b", "b", "b", "b", "b"],
                COLUMN_X: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            }
        )
        result = frame.select(hma(pl.col(COLUMN_X), 4).over(GROUP_KEY).alias("y"))["y"].to_list()
        assert_matches(
            result,
            [None, None, None, None, 5.0, 6.0, None, None, None, None, 50.0, 60.0],
        )


class TestHmaEdge:
    """
    Boundaries, warm-up, and null / NaN handling.
    """

    def test_window_below_two_raises(self) -> None:
        """
        Verifies that ``window < 2`` raises ``ValueError``: the half-period ``floor(window / 2 + 0.5)`` collapses to
        ``1`` at ``window == 1`` and the Hull average degenerates there, so the smallest meaningful window is ``2``.
        """
        with pytest.raises(ValueError, match="window must be >= 2"):
            hma(pl.col(COLUMN_X), 1)
        with pytest.raises(ValueError, match="window must be >= 2"):
            hma(pl.col(COLUMN_X), 0)

    def test_warmup_null_count(self) -> None:
        """
        Verifies that the first ``window + round_half_up(sqrt(window)) - 2`` rows are null and the next is defined.
        """
        values = [float(value) for value in range(1, 13)]
        smoothing_window = math.floor(math.sqrt(4) + 0.5)
        warmup = 4 + smoothing_window - 2
        result = apply_expr(values, hma(pl.col(COLUMN_X), 4))
        assert result[:warmup] == [None] * warmup
        assert result[warmup] is not None

    def test_empty(self) -> None:
        """
        Verifies that an empty input yields an empty output.
        """
        assert_matches(apply_expr([], hma(pl.col(COLUMN_X), 4)), [])

    def test_all_null(self) -> None:
        """
        Verifies that an all-null series yields an all-null output.
        """
        assert_matches(
            apply_expr([None, None, None, None, None], hma(pl.col(COLUMN_X), 4)), [None, None, None, None, None]
        )

    def test_minimum_window(self) -> None:
        """
        Verifies the smallest meaningful window (``window == 2``) computes against the reference.
        """
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert_matches(apply_expr(values, hma(pl.col(COLUMN_X), 2)), hma_reference(values, 2))

    def test_window_equals_length_all_null(self) -> None:
        """
        Verifies that when ``window`` equals the series length the whole result is warm-up ``null`` (the inner
        ``WMA(x, window)`` never sees a full window, so no value is ever emitted).
        """
        assert_matches(apply_expr([1.0, 2.0, 3.0, 4.0], hma(pl.col(COLUMN_X), 4)), [None, None, None, None])

    def test_single_row(self) -> None:
        """
        Verifies behavior on a one-element series (entirely warm-up null).
        """
        assert_matches(apply_expr([42.0], hma(pl.col(COLUMN_X), 2)), [None])

    def test_null_propagates(self) -> None:
        """
        Verifies that an interior ``null`` propagates through every window that reaches it.
        """
        values = [2.0, 4.0, None, 8.0, 10.0, 12.0]
        assert_matches(apply_expr(values, hma(pl.col(COLUMN_X), 4)), hma_reference(values, 4))

    def test_nan_propagates(self) -> None:
        """
        Verifies that an interior ``NaN`` propagates as ``NaN`` through every window that reaches it.
        """
        values = [2.0, 4.0, math.nan, 8.0, 10.0, 12.0]
        assert_matches(apply_expr(values, hma(pl.col(COLUMN_X), 4)), hma_reference(values, 4))


class TestHmaCorrectness:
    """
    Against the naive reference oracle and frozen golden-master values.
    """

    def test_matches_reference(self) -> None:
        """
        Verifies agreement with the naive closed-form reference across several windows.
        """
        values = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0, 3.0, 5.0, 8.0]
        for window in (2, 3, 4, 5, 6, 7):
            assert_matches(apply_expr(values, hma(pl.col(COLUMN_X), window)), hma_reference(values, window))

    def test_golden_master_ramp(self) -> None:
        """
        Verifies the frozen reference: HMA(window=4) over a 1..10 ramp tracks the ramp once warmed up.
        """
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        assert_matches(
            apply_expr(values, hma(pl.col(COLUMN_X), 4)),
            [None, None, None, None, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        )

    def test_golden_master_overshoot(self) -> None:
        """
        Verifies the frozen reference on a step series, where the inner lag correction ``2 * WMA(x, half) - WMA(x,
        window)`` over- and under-shoots beyond the range of the input window before the final smoothing settles.
        """
        values = [1.0, 1.0, 1.0, 10.0, 10.0, 10.0, 10.0, 10.0]
        assert_matches(
            apply_expr(values, hma(pl.col(COLUMN_X), 4)),
            [None, None, None, None, 11.6, 11.5, 10.3, 10.0],
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    def test_golden_master_round_half_up(self) -> None:
        """
        Verifies the round-half-**up** period reduction at ``window == 5``, where ``window / 2`` lands on a ``.5``
        boundary: the half-period is ``floor(5 / 2 + 0.5) == 3``, not the banker-rounded (round-half-to-even) ``2``.
        """
        values = [float(value) for value in range(1, 13)]
        assert_matches(
            apply_expr(values, hma(pl.col(COLUMN_X), 5)),
            [
                None,
                None,
                None,
                None,
                None,
                5.666667,
                6.666667,
                7.666667,
                8.666667,
                9.666667,
                10.666667,
                11.666667,
            ],
            rel_tol=RELATIVE_TOLERANCE_SCALE,
            abs_tol=ABSOLUTE_TOLERANCE_SCALE,
        )


class TestHmaProperties:
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
            apply_expr(values, hma(pl.col(COLUMN_X), window)),
            hma_reference(values, window),
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=input_scale(values) * EXACT_TOLERANCE_FACTOR,
        )

    @given(
        case=_cases(st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False)),
        exponent=st.sampled_from([-4, -3, -2, -1, 1, 2, 3, 4]),
    )
    def test_scale_homogeneity(
        self,
        case: tuple[list[float], int],
        exponent: int,
    ) -> None:
        """
        Verifies that HMA is homogeneous of degree 1: ``hma(k * x) == k * hma(x)``. ``k`` is a power of two so the
        rescaling is lossless and cannot introduce a floating-point artifact.
        """
        k = 2.0**exponent
        values, window = case
        scaled_values = [value * k for value in values]
        result_base = apply_expr(values, hma(pl.col(COLUMN_X), window))
        result_scaled = apply_expr(scaled_values, hma(pl.col(COLUMN_X), window))
        assert_scale_homogeneous(result_scaled, result_base, k=k, degree=1)

    @given(case=_cases(st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False)))
    def test_warmup_null_count_property(
        self,
        case: tuple[list[float], int],
    ) -> None:
        """
        Verifies the warm-up null run is exactly ``window + round_half_up(sqrt(window)) - 2`` long on clean input.
        """
        values, window = case
        warmup = _warmup(window)
        result = apply_expr(values, hma(pl.col(COLUMN_X), window))
        assert all(value is None for value in result[:warmup])
        if len(values) > warmup:
            assert result[warmup] is not None

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
            apply_expr(values, hma(pl.col(COLUMN_X), window)),
            hma_reference(values, window),
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=input_scale(values) * EXACT_TOLERANCE_FACTOR,
        )

    @given(
        case=_cases(st.floats(min_value=1e-3, max_value=1.0, allow_nan=False, allow_infinity=False)),
        scale=st.sampled_from([1e-6, 1e6, 1e9]),
    )
    def test_matches_reference_at_large_magnitude(
        self,
        case: tuple[list[float], int],
        scale: float,
    ) -> None:
        """
        Verifies that at extreme positive magnitudes the implementation stays finite where the reference is and agrees.
        """
        values, window = case
        scaled_values = [value * scale for value in values]
        assert_matches(
            apply_expr(scaled_values, hma(pl.col(COLUMN_X), window)),
            hma_reference(scaled_values, window),
            rel_tol=RELATIVE_TOLERANCE_SCALE,
            abs_tol=input_scale(scaled_values) * EXACT_TOLERANCE_FACTOR,
        )
