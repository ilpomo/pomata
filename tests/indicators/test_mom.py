"""
Tests for ``pomata.indicators.mom`` — the Momentum (MOM) oscillator (the fixed-lag price difference).

Categories are split into classes; cross-cutting categories elsewhere use markers (see ``tests/README.md``).
"""

import math

import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st
from tests.indicators.oracles import mom_reference
from tests.support import (
    ABSOLUTE_TOLERANCE_REFERENCE,
    COLUMN_X,
    EXACT_TOLERANCE_FACTOR,
    GROUP_KEY,
    RELATIVE_TOLERANCE_PROPERTY,
    RELATIVE_TOLERANCE_SCALE,
    apply_expr,
    assert_matches,
    assert_scale_homogeneous,
    count_leading_nulls,
    input_scale,
    missing_data_floats,
)

from pomata.indicators import mom

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- derived, not chosen; the rationale is the shared method in CORRECTNESS.md, the numbers are this
# indicator's. To add an indicator, set its three facts here; the property tier below is then the same shape as every
# other indicator's.
#   1. warmup  W(window) = window   (no value reaches back ``window`` rows until row ``window``; clamped to the series
#              length)
#   2. memory  the oracle shares pomata's seeding, so the property holds from the first defined row (M = 0); each
#              example carries D in [window, 2 * window] defined values -- one window of output, never all warm-up
#   3. domain  finite floats; the magnitude is widened per test below
# MOM is the fixed-lag difference ``x_t - x_{t-n}`` -- homogeneous of degree 1, so it carries a degree-1
# scale-homogeneity property and a large-magnitude test. The difference is exact (no recurrence, no sqrt), so the
# streaming implementation and the two-pass oracle agree bit-for-bit even at extreme magnitude; its magnitude-relative
# band is therefore input_scale * EXACT_TOLERANCE_FACTOR, the residual-is-exactly-zero factor. Repetitions N are the
# shared CI profile (tests/conftest.py); override per-test only if its parameter space is larger.
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


class TestMomContract:
    """
    Type, shape, and lazy/eager guarantees.
    """

    def test_over_partitions_independently(self) -> None:
        """
        Verifies that under ``.over`` the shift resets per group and never reaches across group boundaries.
        """
        frame = pl.DataFrame({GROUP_KEY: ["a", "a", "a", "b", "b", "b"], COLUMN_X: [1.0, 2.0, 3.0, 10.0, 20.0, 30.0]})
        result = frame.select(mom(pl.col(COLUMN_X), 2).over(GROUP_KEY).alias("y"))["y"].to_list()
        assert_matches(result, [None, None, 2.0, None, None, 20.0])


class TestMomEdge:
    """
    Boundaries, warm-up, and null / NaN handling.
    """

    def test_window_below_one_raises(self) -> None:
        """
        Verifies that ``window < 1`` raises ``ValueError``.
        """
        with pytest.raises(ValueError, match="window must be >= 1"):
            mom(pl.col(COLUMN_X), 0)

    def test_warmup_null_count(self) -> None:
        """
        Verifies that the first ``window`` rows are null and the next row is defined.
        """
        result = apply_expr([1.0, 2.0, 3.0, 4.0, 5.0], mom(pl.col(COLUMN_X), 3))
        assert result[:3] == [None, None, None]
        assert result[3] is not None

    def test_window_one(self) -> None:
        """
        Verifies that ``window == 1`` is the first difference with a single leading null.
        """
        assert_matches(apply_expr([2.0, 4.0, 6.0, 8.0], mom(pl.col(COLUMN_X), 1)), [None, 2.0, 2.0, 2.0])

    def test_window_equals_length(self) -> None:
        """
        Verifies that when ``window`` equals the series length the whole output is null (no value reaches back).
        """
        assert_matches(apply_expr([1.0, 2.0, 3.0], mom(pl.col(COLUMN_X), 3)), [None, None, None])

    def test_window_exceeds_length(self) -> None:
        """
        Verifies that when ``window`` exceeds the series length the whole output is null (warm-up clamps to length).
        """
        assert_matches(apply_expr([1.0, 2.0, 3.0], mom(pl.col(COLUMN_X), 5)), [None, None, None])

    def test_single_row(self) -> None:
        """
        Verifies behavior on a one-element series: the lone value is always warm-up.
        """
        assert_matches(apply_expr([42.0], mom(pl.col(COLUMN_X), 1)), [None])
        assert_matches(apply_expr([42.0], mom(pl.col(COLUMN_X), 2)), [None])

    def test_all_null(self) -> None:
        """
        Verifies that an all-null series yields all null.
        """
        assert_matches(apply_expr([None, None, None, None], mom(pl.col(COLUMN_X), 1)), [None, None, None, None])

    def test_all_nan(self) -> None:
        """
        Verifies that an all-NaN series yields null during warm-up and ``NaN`` thereafter.
        """
        assert_matches(
            apply_expr([math.nan, math.nan, math.nan, math.nan], mom(pl.col(COLUMN_X), 1)),
            [None, math.nan, math.nan, math.nan],
        )

    def test_null_propagates(self) -> None:
        """
        Verifies that a ``null`` at either difference endpoint yields ``null`` at that position only.
        """
        assert_matches(
            apply_expr([1.0, None, 3.0, 4.0, 5.0, 6.0], mom(pl.col(COLUMN_X), 2)),
            [None, None, 2.0, None, 2.0, 2.0],
        )

    def test_nan_propagates(self) -> None:
        """
        Verifies that a ``NaN`` at either difference endpoint yields ``NaN`` at that position only (no latching).
        """
        assert_matches(
            apply_expr([1.0, math.nan, 3.0, 4.0, 5.0, 6.0], mom(pl.col(COLUMN_X), 2)),
            [None, None, 2.0, math.nan, 2.0, 2.0],
        )

    def test_constant_series_is_zero(self) -> None:
        """
        Verifies that the momentum of a constant series is zero once warmed up.
        """
        assert_matches(
            apply_expr([5.0, 5.0, 5.0, 5.0, 5.0, 5.0], mom(pl.col(COLUMN_X), 3)),
            [None, None, None, 0.0, 0.0, 0.0],
        )


class TestMomCorrectness:
    """
    Against the naive reference oracle and frozen golden-master values.
    """

    def test_matches_reference(self) -> None:
        """
        Verifies agreement with the naive closed-form reference across several windows.
        """
        values = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]
        for window in (1, 2, 3, 4, 5):
            assert_matches(apply_expr(values, mom(pl.col(COLUMN_X), window)), mom_reference(values, window))

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference: mom(window=3) over [3, 1, 4, 1, 5, 9, 2, 6].
        """
        assert_matches(
            apply_expr([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0], mom(pl.col(COLUMN_X), 3)),
            [None, None, None, -2.0, 4.0, 5.0, 1.0, 1.0],
        )


class TestMomProperties:
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
            apply_expr(values, mom(pl.col(COLUMN_X), window)),
            mom_reference(values, window),
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

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
            apply_expr(values, mom(pl.col(COLUMN_X), window)),
            mom_reference(values, window),
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
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
        Verifies that ``mom`` is homogeneous of degree 1: scaling every input value by a constant ``k`` scales the
        output by the same ``k`` -- ``mom(k * x) == k * mom(x)``. ``k`` is a power of two, so the rescale is exact
        and adds no floating-point error.
        """
        k = 2.0**exponent
        values, window = case
        result_base = apply_expr(values, mom(pl.col(COLUMN_X), window))
        result_scaled = apply_expr([value * k for value in values], mom(pl.col(COLUMN_X), window))
        assert_scale_homogeneous(result_scaled, result_base, k=k, degree=1)

    @given(case=_cases(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)))
    def test_warmup_null_count_property(
        self,
        case: tuple[list[float], int],
    ) -> None:
        """
        Verifies that the leading-null run is exactly ``min(window, len(values))``.
        """
        values, window = case
        result = apply_expr(values, mom(pl.col(COLUMN_X), window))
        leading_nulls = count_leading_nulls(result)
        assert leading_nulls == min(window, len(values))

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
            apply_expr(scaled_values, mom(pl.col(COLUMN_X), window)),
            mom_reference(scaled_values, window),
            rel_tol=RELATIVE_TOLERANCE_SCALE,
            abs_tol=input_scale(scaled_values) * EXACT_TOLERANCE_FACTOR,
        )
