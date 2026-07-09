"""
Tests for ``pomata.indicators.linear_regression_angle`` — the rolling least-squares slope as an angle in degrees.

``linear_regression_angle`` is single-input, so tests use the shared ``apply_expr`` helper to materialize the factory
over a one-column ``Float64`` frame; ``assert_matches`` and the naive ``linear_regression_angle_reference`` oracle are
shared across the suite. The angle is a non-linear (arctangent) transform of the slope, so — unlike the other
regression outputs — it is neither scale-invariant nor degree-1 homogeneous; it is instead bounded in ``(-90, 90)``, so
it carries a boundedness property in place of the homogeneity / large-magnitude tests.

The ladder is the canonical one: contract, edge (window floor / warm-up / null / NaN), correctness (vs the closed-form
reference and a frozen golden master), and properties (reference agreement incl. missing data, boundedness). Categories
are split into classes; cross-cutting categories use markers (see ``tests/README.md``).
"""

import math

import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st
from tests.indicators.oracles import linear_regression_angle_reference
from tests.support import (
    ABSOLUTE_TOLERANCE_EXACT,
    COLUMN_X,
    EXACT_TOLERANCE_FACTOR,
    GROUP_KEY,
    RELATIVE_TOLERANCE_SCALE,
    WINDOW_MAX,
    apply_expr,
    assert_matches,
    input_scale,
    missing_data_floats,
)

from pomata.indicators import linear_regression_angle

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- derived, not chosen; the rationale is the shared method in CORRECTNESS.md, the numbers are this
# indicator's. To add an indicator, set its three facts here; the property tier below is then the same shape as every
# other indicator's.
#   1. warmup  W(window) = window - 1   (the window must hold ``window`` non-null values before a result is emitted)
#   2. memory  the oracle is windowed like pomata, so the property holds from the first defined row (M = 0); each
#              example carries D in [window, 2 * window] defined values -- one window of output, never all warm-up
#   3. domain  finite floats over the test's regime (any-input / bounded / missing-data), widened per test below
# Windows span ``window_min`` .. WINDOW_MAX; a line needs ``window >= 2`` (the floor in ``_cases``). The angle is an
# arctangent (scale-invariant, O(1) in ``(-90, 90)``), so there is no scale-homogeneity or large-magnitude VALUE test
# (a non-linear transform does not scale). Its absolute floor is still sized ``input_scale * EXACT_TOLERANCE_FACTOR``,
# like the slope, because near a flat line the angle inherits the slope's magnitude-scaled rounding through ``arctan``
# -- a fixed floor is silently too tight at large magnitude. Repetitions N are the shared CI profile
# (tests/conftest.py); override per-test only if its parameter space is larger.
# ----------------------------------------------------------------------------------------------------------------------


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
    length = (window - 1) + defined
    return draw(st.lists(values, min_size=length, max_size=length)), window


class TestLinearRegressionAngleContract:
    """
    Type, shape, and lazy/eager guarantees.
    """

    def test_over_partitions_independently(self) -> None:
        """
        Verifies that under ``.over`` the window resets per group and never spans group boundaries.
        """
        frame = pl.DataFrame(
            {GROUP_KEY: ["a"] * 4 + ["b"] * 4, COLUMN_X: [10.0, 11.0, 13.0, 12.0, 20.0, 22.0, 21.0, 24.0]}
        )
        expr = linear_regression_angle(pl.col(COLUMN_X), 3).over(GROUP_KEY)
        grouped = frame.select(expr.alias("y"))["y"].to_list()
        group_a = apply_expr([10.0, 11.0, 13.0, 12.0], linear_regression_angle(pl.col(COLUMN_X), 3))
        group_b = apply_expr([20.0, 22.0, 21.0, 24.0], linear_regression_angle(pl.col(COLUMN_X), 3))
        assert_matches(grouped, group_a + group_b)


class TestLinearRegressionAngleEdge:
    """
    Boundaries, warm-up, and null / NaN handling.
    """

    def test_window_below_two_raises(self) -> None:
        """
        Verifies that ``window < 2`` raises ``ValueError`` (a line needs at least two points).
        """
        with pytest.raises(ValueError, match="window must be >= 2"):
            linear_regression_angle(pl.col(COLUMN_X), 1)

    def test_single_row(self) -> None:
        """
        Verifies that a one-element series is all warm-up: a window of more than one observation yields null.
        """
        assert_matches(apply_expr([42.0], linear_regression_angle(pl.col(COLUMN_X), 2)), [None])

    def test_all_null(self) -> None:
        """
        Verifies that an all-null series stays null (no window ever holds the required non-null values).
        """
        assert_matches(apply_expr([None, None, None], linear_regression_angle(pl.col(COLUMN_X), 2)), [None, None, None])

    def test_null_in_window_is_null(self) -> None:
        """
        Verifies that an interior ``null`` nulls every window that overlaps it, then the output recovers.
        """
        values = [10.0, 11.0, 13.0, None, 14.0, 15.0, 16.0]
        assert_matches(
            apply_expr(values, linear_regression_angle(pl.col(COLUMN_X), 3)),
            linear_regression_angle_reference(values, 3),
        )

    def test_nan_propagates(self) -> None:
        """
        Verifies that a NaN propagates (matching the naive reference).
        """
        values = [10.0, 11.0, 13.0, 13.5, 14.0, math.nan, 16.0]
        assert_matches(
            apply_expr(values, linear_regression_angle(pl.col(COLUMN_X), 3)),
            linear_regression_angle_reference(values, 3),
        )

    def test_warmup_null_count(self) -> None:
        """
        Verifies the warm-up is ``window - 1`` rows.
        """
        result = apply_expr([10.0, 11.0, 13.0, 12.0, 14.0], linear_regression_angle(pl.col(COLUMN_X), 3))
        assert result[:2] == [None, None]
        assert result[2] is not None

    def test_window_exceeds_length(self) -> None:
        """
        Verifies the whole output is null when ``window`` exceeds the series length (no full window ever forms).
        """
        assert_matches(apply_expr([1.0, 2.0, 3.0], linear_regression_angle(pl.col(COLUMN_X), 5)), [None, None, None])

    def test_flat_series_is_zero_angle(self) -> None:
        """
        Verifies a constant series has zero slope, so every defined angle collapses to ``0.0`` (no spurious tilt).
        """
        angles = apply_expr([5.0] * 8, linear_regression_angle(pl.col(COLUMN_X), 4))
        defined = [value for value in angles if value is not None]
        assert defined
        for value in defined:
            assert math.isclose(value, 0.0, rel_tol=0.0, abs_tol=ABSOLUTE_TOLERANCE_EXACT)


class TestLinearRegressionAngleCorrectness:
    """
    Against the naive reference oracle and frozen golden-master values.
    """

    def test_matches_reference(self) -> None:
        """
        Verifies agreement with the naive closed-form reference across several windows.
        """
        values = [10.0, 11.0, 13.0, 12.0, 14.0, 13.0, 15.0, 14.0, 16.0, 15.0]
        for window in (2, 3, 4, 5):
            assert_matches(
                apply_expr(values, linear_regression_angle(pl.col(COLUMN_X), window)),
                linear_regression_angle_reference(values, window),
            )

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference: angle(window=3) over the sample series.
        """
        result = apply_expr(
            [10.0, 11.0, 13.0, 12.0, 14.0, 13.0, 15.0], linear_regression_angle(pl.col(COLUMN_X), 3).round(4)
        )
        assert_matches(result, [None, None, 56.3099, 26.5651, 26.5651, 26.5651, 26.5651])


class TestLinearRegressionAngleProperties:
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
            apply_expr(values, linear_regression_angle(pl.col(COLUMN_X), window)),
            linear_regression_angle_reference(values, window),
            # Scale band, not the 1e-10 headline: the least-squares fit cancels a difference of large index-weighted
            # terms as the line flattens, so the relative residual is genuinely looser there; the input_scale abs floor
            # carries the near-flat regime (see CORRECTNESS.md, "Where it stops").
            rel_tol=RELATIVE_TOLERANCE_SCALE,
            abs_tol=input_scale(values) * EXACT_TOLERANCE_FACTOR,
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
            apply_expr(values, linear_regression_angle(pl.col(COLUMN_X), window)),
            linear_regression_angle_reference(values, window),
            # Scale band, not the 1e-10 headline: the least-squares fit cancels a difference of large index-weighted
            # terms as the line flattens, so the relative residual is genuinely looser there; the input_scale abs floor
            # carries the near-flat regime (see CORRECTNESS.md, "Where it stops").
            rel_tol=RELATIVE_TOLERANCE_SCALE,
            abs_tol=input_scale(values) * EXACT_TOLERANCE_FACTOR,
        )

    @given(case=_cases(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)))
    def test_bounded(
        self,
        case: tuple[list[float], int],
    ) -> None:
        """
        Verifies that every defined angle lies strictly within ``(-90, 90)`` degrees.
        """
        values, window = case
        for value in apply_expr(values, linear_regression_angle(pl.col(COLUMN_X), window)):
            if value is not None and not math.isnan(value):
                assert -90.0 < value < 90.0
