"""
Tests for ``pomata.indicators.roc`` — the Rate of Change momentum oscillator.

Categories are split into classes; cross-cutting categories elsewhere use markers (see ``tests/README.md``).
"""

import math

import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st
from polars.testing import assert_frame_equal
from tests.indicators.oracles import roc_reference
from tests.support import (
    ABSOLUTE_TOLERANCE_REFERENCE,
    COLUMN_X,
    GROUP_KEY,
    RELATIVE_TOLERANCE_PROPERTY,
    apply_expr,
    assert_matches,
    assert_scale_homogeneous,
    count_leading_nulls,
    missing_data_floats,
)

from pomata.indicators import roc

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- derived, not chosen; the rationale is the shared method in CORRECTNESS.md, the numbers are this
# indicator's. To add an indicator, set its three facts here; the property tier below is then the same shape as every
# other indicator's.
#   1. warmup  W(window) = window   (the lagged term ``expr.shift(window)`` is undefined for the first ``window`` rows,
#              so no change can be measured there)
#   2. memory  the oracle shares pomata's seeding, so the property holds from the first defined row (M = 0); each
#              example carries D in [window, 2 * window] defined values -- one window of output, never all warm-up
#   3. domain  finite floats; ROC is scale-INVARIANT, so the scale tier draws strictly positive values (so a power-of-
#              two rescaling cannot flip a sign) and the magnitude is widened per test below
# ROC is a scale-INVARIANT ratio (the common scale cancels in ``(x_t - x_{t-n}) / x_{t-n}``): its value is O(1) whatever
# the input magnitude, so its tolerance is ABSOLUTE (never input_scale-sized), and it carries a scale-INVARIANCE
# property in place of the homogeneity / large-magnitude tests of a scale-dependent indicator -- a large-magnitude test
# would be vacuous because the common scale cancels in the ratio. Repetitions N are the shared CI profile
# (tests/conftest.py); override per-test only if its parameter space is larger.
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


class TestRocContract:
    """
    Type, shape, and lazy/eager guarantees.
    """

    def test_returns_expr(self) -> None:
        """
        Verifies that the factory returns a ``pl.Expr`` without touching a frame.
        """
        assert isinstance(roc(pl.col(COLUMN_X), 3), pl.Expr)

    def test_preserves_length_and_dtype(self) -> None:
        """
        Verifies that the output has one value per input row and is ``Float64``.
        """
        frame = pl.DataFrame({COLUMN_X: pl.Series(COLUMN_X, [1.0, 2.0, 3.0, 4.0, 5.0])})
        result = frame.select(roc(pl.col(COLUMN_X), 2).alias("y"))
        assert result.height == frame.height
        assert result.schema["y"] == pl.Float64

    def test_lazy_eager_parity(self) -> None:
        """
        Verifies that eager and lazy application produce identical materialized output.
        """
        frame = pl.DataFrame({COLUMN_X: pl.Series(COLUMN_X, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])})
        result_eager = frame.select(roc(pl.col(COLUMN_X), 2).alias("y"))
        result_lazy = frame.lazy().select(roc(pl.col(COLUMN_X), 2).alias("y")).collect()
        assert_frame_equal(result_eager, result_lazy)

    def test_over_partitions_independently(self) -> None:
        """
        Verifies that under ``.over`` the lagged shift resets per group and never spans group boundaries.
        """
        frame = pl.DataFrame({GROUP_KEY: ["a", "a", "a", "b", "b", "b"], COLUMN_X: [1.0, 2.0, 4.0, 10.0, 20.0, 40.0]})
        result = frame.select(roc(pl.col(COLUMN_X), 1).over(GROUP_KEY).alias("y"))["y"].to_list()
        assert_matches(result, [None, 100.0, 100.0, None, 100.0, 100.0])


class TestRocEdge:
    """
    Boundaries, warm-up, and null / NaN / division-by-zero handling.
    """

    def test_window_below_one_raises(self) -> None:
        """
        Verifies that ``window < 1`` raises ``ValueError``.
        """
        with pytest.raises(ValueError, match="window must be >= 1"):
            roc(pl.col(COLUMN_X), 0)

    def test_warmup_null_count(self) -> None:
        """
        Verifies that the first ``window`` rows are null (the lagged term ``expr.shift(window)`` is undefined there) and
        the row immediately after warm-up is defined.
        """
        result = apply_expr([1.0, 2.0, 3.0, 4.0, 5.0], roc(pl.col(COLUMN_X), 2))
        assert result[:2] == [None, None]
        assert result[2] is not None

    def test_window_one(self) -> None:
        """
        Verifies the single-step percentage change at ``window == 1`` (the one-period simple return in percent).
        """
        assert_matches(apply_expr([2.0, 4.0, 6.0], roc(pl.col(COLUMN_X), 1)), [None, 100.0, 50.0])

    def test_window_equals_length(self) -> None:
        """
        Verifies that when ``window`` equals the series length the whole output is null (no lag is reachable).
        """
        assert_matches(apply_expr([2.0, 4.0, 6.0], roc(pl.col(COLUMN_X), 3)), [None, None, None])

    def test_window_exceeds_length(self) -> None:
        """
        Verifies that when ``window`` exceeds the series length the whole output is null.
        """
        assert_matches(apply_expr([2.0, 4.0, 6.0], roc(pl.col(COLUMN_X), 5)), [None, None, None])

    def test_single_row(self) -> None:
        """
        Verifies behavior on a one-element series.
        """
        assert_matches(apply_expr([42.0], roc(pl.col(COLUMN_X), 1)), [None])

    def test_empty(self) -> None:
        """
        Verifies that an empty series yields an empty result.
        """
        assert_matches(apply_expr([], roc(pl.col(COLUMN_X), 1)), [])

    def test_all_null(self) -> None:
        """
        Verifies that an all-null series yields all null.
        """
        assert_matches(apply_expr([None, None, None], roc(pl.col(COLUMN_X), 1)), [None, None, None])

    def test_all_nan(self) -> None:
        """
        Verifies that an all-NaN series yields null at warm-up then NaN (a NaN at either row propagates).
        """
        assert_matches(apply_expr([math.nan, math.nan, math.nan], roc(pl.col(COLUMN_X), 1)), [None, math.nan, math.nan])

    def test_constant_series_is_zero(self) -> None:
        """
        Verifies that the ROC of a constant non-zero series is ``0`` once warmed up: the change is zero at every defined
        row, so the percentage return is exactly ``0``.
        """
        assert_matches(apply_expr([5.0, 5.0, 5.0, 5.0], roc(pl.col(COLUMN_X), 1)), [None, 0.0, 0.0, 0.0])

    def test_null_at_current_or_lagged_propagates(self) -> None:
        """
        Verifies that a ``null`` at the current, the lagged, or a trailing row yields ``null`` at that position.
        """
        assert_matches(
            apply_expr([1.0, None, 3.0, 4.0], roc(pl.col(COLUMN_X), 1)),
            [None, None, None, 33.333333333333336],
        )
        assert_matches(
            apply_expr([2.0, 4.0, None, 8.0, 10.0], roc(pl.col(COLUMN_X), 2)), [None, None, None, 100.0, None]
        )
        assert_matches(apply_expr([1.0, 2.0, 3.0, None], roc(pl.col(COLUMN_X), 1)), [None, 100.0, 50.0, None])

    def test_nan_at_current_or_lagged_propagates(self) -> None:
        """
        Verifies that a ``NaN`` at the current or the lagged row yields ``NaN`` at exactly that position.
        """
        assert_matches(
            apply_expr([1.0, math.nan, 3.0, 4.0], roc(pl.col(COLUMN_X), 1)),
            [None, math.nan, math.nan, 33.333333333333336],
        )

    def test_zero_lagged_nonzero_change_is_signed_inf(self) -> None:
        """
        Verifies that a non-zero change over a zero lagged value follows IEEE-754 and yields ``+/-inf``, with the sign
        tracking the direction of the change.
        """
        result = apply_expr([0.0, 5.0, 0.0, -5.0], roc(pl.col(COLUMN_X), 1))
        assert result[0] is None
        assert result[1] == math.inf
        assert result[2] == -100.0
        assert result[3] == -math.inf

    def test_zero_lagged_mixed_change(self) -> None:
        """
        Verifies the two zero-denominator IEEE-754 branches together: a zero change over zero is ``NaN`` (``0 / 0``)
        and a non-zero change over zero is ``+inf``.
        """
        assert_matches(apply_expr([0.0, 0.0, 5.0], roc(pl.col(COLUMN_X), 1)), [None, math.nan, math.inf])


class TestRocCorrectness:
    """
    Against the naive reference oracle and frozen golden-master values.
    """

    def test_matches_reference(self) -> None:
        """
        Verifies agreement with the naive closed-form reference across several windows.
        """
        values = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]
        for window in (1, 2, 3, 4, 5):
            assert_matches(apply_expr(values, roc(pl.col(COLUMN_X), window)), roc_reference(values, window))

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference: ROC(window=2) over [2, 4, 6, 8, 10] == [None, None, 200, 100, 66.6667].
        """
        assert_matches(
            apply_expr([2.0, 4.0, 6.0, 8.0, 10.0], roc(pl.col(COLUMN_X), 2)),
            [None, None, 200.0, 100.0, 66.66666666666667],
        )


class TestRocProperties:
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
            apply_expr(values, roc(pl.col(COLUMN_X), window)),
            roc_reference(values, window),
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

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
        Verifies that ROC is scale-invariant: ``roc(k * x) == roc(x)`` for any non-zero ``k`` (the scale cancels in the
        ratio). ``k`` is a power of two so the rescaling is lossless and cannot introduce a floating-point artifact.
        """
        k = 2.0**exponent
        values, window = case
        result_base = apply_expr(values, roc(pl.col(COLUMN_X), window))
        result_scaled = apply_expr([value * k for value in values], roc(pl.col(COLUMN_X), window))
        assert_scale_homogeneous(result_scaled, result_base, k=k, degree=0)

    @given(case=_cases(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)))
    def test_warmup_null_count_property(
        self,
        case: tuple[list[float], int],
    ) -> None:
        """
        Verifies that the leading-null run is exactly ``min(window, len(values))``.
        """
        values, window = case
        result = apply_expr(values, roc(pl.col(COLUMN_X), window))
        leading_nulls = count_leading_nulls(result)
        assert leading_nulls == min(window, len(values))

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
            apply_expr(values, roc(pl.col(COLUMN_X), window)),
            roc_reference(values, window),
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )
