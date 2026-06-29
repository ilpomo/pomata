"""
Tests for ``pomata.metrics.downside_deviation_rolling`` — the rolling (windowed) twin of
:func:`pomata.metrics.downside_deviation`.

``downside_deviation_rolling`` is single-input and WINDOWED-SERIES-VALUED (a return series → a series the same length,
one value per trailing window), so tests use the shared ``apply_expr`` helper; ``assert_matches`` and the naive
``downside_deviation_rolling_reference`` oracle (the reducing :func:`downside_deviation` recomputed over each window)
are shared across the suite. The rolling null/NaN policy differs from the reducing one: a window holding any ``null`` is
``null`` (it must hold ``window`` non-null values), and a ``NaN`` inside a window propagates.

The ladder is the canonical one: contract (type / length-preserving / lazy-eager / ``.over`` per-group warm-up), edge
(validation / empty / warm-up / null-in-window / NaN / no-downside), correctness (vs the closed-form reference and a
frozen golden master), and properties (reference agreement for any input and under missing data). Categories are split
into classes; cross-cutting categories use markers.
"""

import math
from collections.abc import Sequence

import polars as pl
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from polars.testing import assert_frame_equal
from tests.metrics.oracles import downside_deviation_rolling_reference
from tests.support import (
    ABSOLUTE_TOLERANCE_REFERENCE,
    COLUMN_X,
    GROUP_KEY,
    RELATIVE_TOLERANCE_PROPERTY,
    RELATIVE_TOLERANCE_REFERENCE,
    WINDOW_MAX,
    apply_expr,
    assert_matches,
    missing_data_floats,
    streaming_abs_tol,
    subnormal_safe_floats,
    windows_well_spread,
)

from pomata.metrics import downside_deviation_rolling


def _abs_tol(values: Sequence[float | None]) -> float:
    """The magnitude-relative absolute tolerance for the annualized rolling downside deviation (sqrt-of-time scale)."""
    return streaming_abs_tol(values, periods=PERIODS)


# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- downside_deviation_rolling is WINDOWED and series-valued. Facts (mirroring the windowed indicators):
#   1. shape   length-preserving: one output row per input row; the first ``window - 1`` rows are warm-up ``null``
#   2. domain  subnormal_safe_floats: finite returns floored off subnormal so the squared shortfall is well-conditioned;
#              the missing variant mixes null / NaN
#   3. window  window_min = 1 (the population RMS is defined for one observation) .. WINDOW_MAX
# Each case carries (window - 1) warm-up rows + a window of defined output, so no example is all warm-up. The rolling
# one-pass primitive agrees with the two-pass oracle to a 1e-6 band (looser than the reducing 1e-10), pinned at SCALE.
# ----------------------------------------------------------------------------------------------------------------------
PERIODS = 252


@st.composite
def _cases[T](draw: st.DrawFn, values: st.SearchStrategy[T], window_min: int = 1) -> tuple[list[T], int]:
    """A (series, window) pair sized so every example has a window of defined output past the warm-up."""
    window = draw(st.integers(min_value=window_min, max_value=WINDOW_MAX))
    defined = draw(st.integers(min_value=window, max_value=2 * window))
    length = (window - 1) + defined
    return draw(st.lists(values, min_size=length, max_size=length)), window


class TestDownsideDeviationRollingContract:
    """
    Type, shape, and lazy/eager guarantees.
    """

    def test_returns_expr(self) -> None:
        """
        Verifies that the factory returns a ``pl.Expr`` without touching a frame.
        """
        assert isinstance(downside_deviation_rolling(pl.col(COLUMN_X), 3, periods_per_year=PERIODS), pl.Expr)

    def test_preserves_length_and_dtype(self) -> None:
        """
        Verifies that the metric maps a series to a ``Float64`` series of the same length.
        """
        frame = pl.DataFrame({COLUMN_X: pl.Series(COLUMN_X, [0.01, -0.02, 0.03, -0.01, 0.02], dtype=pl.Float64)})
        result = frame.select(downside_deviation_rolling(pl.col(COLUMN_X), 3, periods_per_year=PERIODS).alias("d"))
        assert result.height == frame.height
        assert result.schema["d"] == pl.Float64

    def test_lazy_eager_parity(self) -> None:
        """
        Verifies that eager and lazy application produce identical materialized output.
        """
        frame = pl.DataFrame({COLUMN_X: pl.Series(COLUMN_X, [0.01, -0.02, 0.03, -0.01, 0.02], dtype=pl.Float64)})
        expr = downside_deviation_rolling(pl.col(COLUMN_X), 3, periods_per_year=PERIODS).alias("d")
        assert_frame_equal(frame.select(expr), frame.lazy().select(expr).collect())

    def test_over_partitions_independently(self) -> None:
        """
        Verifies that under ``.over`` each group warms up independently and the window never spans a boundary.
        """
        group_a = [0.01, -0.02, 0.03, -0.01]
        group_b = [0.02, -0.05, 0.01, -0.01]
        frame = pl.DataFrame({GROUP_KEY: ["a"] * len(group_a) + ["b"] * len(group_b), COLUMN_X: group_a + group_b})
        grouped = frame.select(
            downside_deviation_rolling(pl.col(COLUMN_X), 2, periods_per_year=PERIODS).over(GROUP_KEY).alias("d")
        )["d"].to_list()
        expected = downside_deviation_rolling_reference(group_a, 2, PERIODS) + downside_deviation_rolling_reference(
            group_b, 2, PERIODS
        )
        assert_matches(grouped, expected, rel_tol=RELATIVE_TOLERANCE_REFERENCE)


class TestDownsideDeviationRollingEdge:
    """
    Validation, boundaries, and null / NaN handling.
    """

    def test_window_below_one_raises(self) -> None:
        """
        Verifies that ``window < 1`` raises ``ValueError``.
        """
        with pytest.raises(ValueError, match="window must be >= 1"):
            downside_deviation_rolling(pl.col(COLUMN_X), 0, periods_per_year=PERIODS)

    def test_periods_per_year_below_one_raises(self) -> None:
        """
        Verifies that ``periods_per_year < 1`` raises ``ValueError``.
        """
        with pytest.raises(ValueError, match="periods_per_year must be >= 1"):
            downside_deviation_rolling(pl.col(COLUMN_X), 3, periods_per_year=0)

    def test_non_finite_threshold_raises(self) -> None:
        """
        Verifies that a non-finite ``threshold`` raises ``ValueError``.
        """
        for invalid in (math.nan, math.inf, -math.inf):
            with pytest.raises(ValueError, match="threshold must be a finite number"):
                downside_deviation_rolling(pl.col(COLUMN_X), 3, periods_per_year=PERIODS, threshold=invalid)

    def test_empty(self) -> None:
        """
        Verifies that an empty series yields an empty result.
        """
        assert apply_expr([], downside_deviation_rolling(pl.col(COLUMN_X), 3, periods_per_year=PERIODS)) == []

    def test_warmup_null_count(self) -> None:
        """
        Verifies that the first ``window - 1`` rows are ``null`` and the rest match the reference.
        """
        values = [0.01, -0.02, 0.03, -0.01, 0.02]
        assert_matches(
            apply_expr(values, downside_deviation_rolling(pl.col(COLUMN_X), 3, periods_per_year=PERIODS)),
            downside_deviation_rolling_reference(values, 3, PERIODS),
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
        )

    def test_null_in_window_is_null(self) -> None:
        """
        Verifies that a window containing a ``null`` yields ``null`` (the window must hold ``window`` non-null values).
        """
        values = [0.01, None, 0.03, -0.01, 0.02]
        assert_matches(
            apply_expr(values, downside_deviation_rolling(pl.col(COLUMN_X), 3, periods_per_year=PERIODS)),
            downside_deviation_rolling_reference(values, 3, PERIODS),
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
        )

    def test_nan_propagates(self) -> None:
        """
        Verifies that a ``NaN`` inside a window propagates to ``NaN`` for the windows that touch it.
        """
        values = [0.01, math.nan, 0.03, -0.01, 0.02]
        assert_matches(
            apply_expr(values, downside_deviation_rolling(pl.col(COLUMN_X), 3, periods_per_year=PERIODS)),
            downside_deviation_rolling_reference(values, 3, PERIODS),
        )

    def test_no_downside_window_is_zero(self) -> None:
        """
        Verifies that a window with every return at or above the threshold has zero shortfall, so the result is ``0``.
        """
        assert_matches(
            apply_expr(
                [0.01, 0.02, 0.03, 0.04], downside_deviation_rolling(pl.col(COLUMN_X), 3, periods_per_year=PERIODS)
            ),
            [None, None, 0.0, 0.0],
        )


class TestDownsideDeviationRollingCorrectness:
    """
    Against the naive reference oracle and frozen golden-master values.
    """

    def test_matches_reference(self) -> None:
        """
        Verifies agreement with the naive closed-form reference over a representative series.
        """
        values = [0.012, -0.008, 0.02, -0.015, 0.005, 0.0, -0.02, 0.018]
        assert_matches(
            apply_expr(values, downside_deviation_rolling(pl.col(COLUMN_X), 4, periods_per_year=PERIODS)),
            downside_deviation_rolling_reference(values, 4, PERIODS),
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference: the daily-annualized rolling downside deviation over a window of three.
        """
        values = [0.01, -0.02, 0.03, -0.01, 0.02, 0.0, -0.015]
        assert_matches(
            apply_expr(values, downside_deviation_rolling(pl.col(COLUMN_X), 3, periods_per_year=252).round(4)),
            [None, None, 0.1833, 0.2049, 0.0917, 0.0917, 0.1375],
        )


class TestDownsideDeviationRollingProperties:
    """
    Invariants that must hold for all inputs (property-based).
    """

    @given(case=_cases(subnormal_safe_floats(bound=1e3)))
    def test_matches_reference_for_any_input(self, case: tuple[list[float], int]) -> None:
        """
        Verifies that, for any well-conditioned series and window, the implementation matches the naive reference.
        """
        values, window = case
        assume(windows_well_spread(values, window))
        assert_matches(
            apply_expr(values, downside_deviation_rolling(pl.col(COLUMN_X), window, periods_per_year=PERIODS)),
            downside_deviation_rolling_reference(values, window, PERIODS),
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=_abs_tol(values),
        )

    @given(case=_cases(missing_data_floats(min_magnitude=1e-3)))
    def test_matches_reference_under_missing_data(self, case: tuple[list[float | None], int]) -> None:
        """
        Verifies that, for inputs freely mixing null / NaN / finite, the implementation matches the naive reference.
        """
        values, window = case
        assume(windows_well_spread(values, window))
        assert_matches(
            apply_expr(values, downside_deviation_rolling(pl.col(COLUMN_X), window, periods_per_year=PERIODS)),
            downside_deviation_rolling_reference(values, window, PERIODS),
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=_abs_tol(values),
        )
