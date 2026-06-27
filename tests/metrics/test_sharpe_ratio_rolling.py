"""
Tests for ``pomata.metrics.sharpe_ratio_rolling`` — the rolling (windowed) twin of :func:`pomata.metrics.sharpe_ratio`.

``sharpe_ratio_rolling`` is single-input and WINDOWED-SERIES-VALUED (a return series → a series the same length, one
value per trailing window), so tests use the shared ``apply_expr`` helper; ``assert_matches`` and the naive
``sharpe_ratio_rolling_reference`` oracle (the reducing :func:`sharpe_ratio` recomputed over each window) are shared
across the suite. A window holding any ``null`` is ``null`` (it must hold ``window`` non-null values); a ``NaN`` inside
a window propagates.

The ladder is the canonical one: contract (type / length-preserving / lazy-eager / ``.over`` per-group warm-up), edge
(validation / empty / warm-up / null-in-window / NaN / zero-volatility), correctness (vs the closed-form reference and a
frozen golden master), and properties (reference agreement for any input and under missing data). Categories are split
into classes; cross-cutting categories use markers.
"""

import math

import polars as pl
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from polars.testing import assert_frame_equal
from tests.metrics.oracles import sharpe_ratio_rolling_reference
from tests.support import (
    ABSOLUTE_TOLERANCE_REFERENCE,
    COLUMN_X,
    GROUP_KEY,
    RELATIVE_TOLERANCE_REFERENCE,
    RELATIVE_TOLERANCE_SCALE,
    WINDOW_MAX,
    apply_expr,
    assert_matches,
    windows_well_spread,
)

from pomata.metrics import sharpe_ratio_rolling

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- sharpe_ratio_rolling is WINDOWED and series-valued (a mean over a standard deviation per window).
# Facts:
#   1. shape   length-preserving: one output row per input row; the first ``window - 1`` rows are warm-up ``null``
#   2. domain  magnitude-bounded returns (``|r|`` in [0.01, 1], sign-varied): the Sharpe ratio is scale-invariant, and
#              same-magnitude values keep the one-pass sliding standard deviation free of the cross-window cancellation
#              a far larger exiting value would leave; the missing variant mixes null / NaN
#   3. window  window_min = 2 (a sample standard deviation needs two observations) .. WINDOW_MAX
# The one-pass rolling standard deviation diverges from the two-pass oracle on near-constant windows, so the property
# tiers require every window to be well-spread; the dimensionless ratio then agrees to a 1e-6 band.
# ----------------------------------------------------------------------------------------------------------------------
PERIODS = 252
_VALUE = st.one_of(
    st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False),
    st.floats(min_value=-1.0, max_value=-0.01, allow_nan=False, allow_infinity=False),
)
_VALUE_MISSING = st.one_of(st.none(), st.just(math.nan), _VALUE)


@st.composite
def _cases[T](draw: st.DrawFn, values: st.SearchStrategy[T], window_min: int = 2) -> tuple[list[T], int]:
    """A (series, window) pair sized so every example has a window of defined output past the warm-up."""
    window = draw(st.integers(min_value=window_min, max_value=WINDOW_MAX))
    defined = draw(st.integers(min_value=window, max_value=2 * window))
    length = (window - 1) + defined
    return draw(st.lists(values, min_size=length, max_size=length)), window


class TestSharpeRollingContract:
    """
    Type, shape, and lazy/eager guarantees.
    """

    def test_returns_expr(self) -> None:
        """
        Verifies that the factory returns a ``pl.Expr`` without touching a frame.
        """
        assert isinstance(sharpe_ratio_rolling(pl.col(COLUMN_X), 3, periods_per_year=PERIODS), pl.Expr)

    def test_preserves_length_and_dtype(self) -> None:
        """
        Verifies that the metric maps a series to a ``Float64`` series of the same length.
        """
        frame = pl.DataFrame({COLUMN_X: pl.Series(COLUMN_X, [0.01, -0.02, 0.03, -0.01, 0.02], dtype=pl.Float64)})
        result = frame.select(sharpe_ratio_rolling(pl.col(COLUMN_X), 3, periods_per_year=PERIODS).alias("s"))
        assert result.height == frame.height
        assert result.schema["s"] == pl.Float64

    def test_lazy_eager_parity(self) -> None:
        """
        Verifies that eager and lazy application produce identical materialized output.
        """
        frame = pl.DataFrame({COLUMN_X: pl.Series(COLUMN_X, [0.01, -0.02, 0.03, -0.01, 0.02], dtype=pl.Float64)})
        expr = sharpe_ratio_rolling(pl.col(COLUMN_X), 3, periods_per_year=PERIODS).alias("s")
        assert_frame_equal(frame.select(expr), frame.lazy().select(expr).collect())

    def test_over_partitions_independently(self) -> None:
        """
        Verifies that under ``.over`` each group warms up independently and the window never spans a boundary.
        """
        group_a = [0.01, -0.02, 0.03, -0.01]
        group_b = [0.02, -0.05, 0.01, -0.01]
        frame = pl.DataFrame({GROUP_KEY: ["a"] * len(group_a) + ["b"] * len(group_b), COLUMN_X: group_a + group_b})
        grouped = frame.select(
            sharpe_ratio_rolling(pl.col(COLUMN_X), 2, periods_per_year=PERIODS).over(GROUP_KEY).alias("s")
        )["s"].to_list()
        expected = sharpe_ratio_rolling_reference(group_a, 2, PERIODS) + sharpe_ratio_rolling_reference(
            group_b, 2, PERIODS
        )
        assert_matches(grouped, expected, rel_tol=RELATIVE_TOLERANCE_REFERENCE)


class TestSharpeRollingEdge:
    """
    Validation, boundaries, and null / NaN handling.
    """

    def test_window_below_two_raises(self) -> None:
        """
        Verifies that ``window < 2`` raises ``ValueError`` (a sample standard deviation needs two observations).
        """
        with pytest.raises(ValueError, match="window must be >= 2"):
            sharpe_ratio_rolling(pl.col(COLUMN_X), 1, periods_per_year=PERIODS)

    def test_periods_per_year_below_one_raises(self) -> None:
        """
        Verifies that ``periods_per_year < 1`` raises ``ValueError``.
        """
        with pytest.raises(ValueError, match="periods_per_year must be >= 1"):
            sharpe_ratio_rolling(pl.col(COLUMN_X), 3, periods_per_year=0)

    def test_non_finite_risk_free_rate_raises(self) -> None:
        """
        Verifies that a non-finite ``risk_free_rate`` raises ``ValueError``.
        """
        for invalid in (math.nan, math.inf, -math.inf):
            with pytest.raises(ValueError, match="risk_free_rate must be a finite number"):
                sharpe_ratio_rolling(pl.col(COLUMN_X), 3, periods_per_year=PERIODS, risk_free_rate=invalid)

    def test_empty(self) -> None:
        """
        Verifies that an empty series yields an empty result.
        """
        assert apply_expr([], sharpe_ratio_rolling(pl.col(COLUMN_X), 3, periods_per_year=PERIODS)) == []

    def test_warm_up_is_null(self) -> None:
        """
        Verifies that the first ``window - 1`` rows are ``null`` and the rest match the reference.
        """
        values = [0.01, -0.02, 0.03, -0.01, 0.02]
        assert_matches(
            apply_expr(values, sharpe_ratio_rolling(pl.col(COLUMN_X), 3, periods_per_year=PERIODS)),
            sharpe_ratio_rolling_reference(values, 3, PERIODS),
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
        )

    def test_null_in_window_is_null(self) -> None:
        """
        Verifies that a window containing a ``null`` yields ``null`` (the window must hold ``window`` non-null values).
        """
        values = [0.01, None, 0.03, -0.01, 0.02]
        assert_matches(
            apply_expr(values, sharpe_ratio_rolling(pl.col(COLUMN_X), 3, periods_per_year=PERIODS)),
            sharpe_ratio_rolling_reference(values, 3, PERIODS),
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
        )

    def test_nan_propagates(self) -> None:
        """
        Verifies that a ``NaN`` inside a window propagates to ``NaN`` for the windows that touch it.
        """
        values = [0.01, math.nan, 0.03, -0.01, 0.02]
        assert_matches(
            apply_expr(values, sharpe_ratio_rolling(pl.col(COLUMN_X), 3, periods_per_year=PERIODS)),
            sharpe_ratio_rolling_reference(values, 3, PERIODS),
        )

    def test_zero_volatility_is_inf(self) -> None:
        """
        Verifies that a constant window has zero dispersion with a positive mean, so the ratio is ``+inf``.
        """
        assert_matches(
            apply_expr([0.5, 0.5, 0.5, 0.5], sharpe_ratio_rolling(pl.col(COLUMN_X), 3, periods_per_year=PERIODS)),
            [None, None, math.inf, math.inf],
        )


class TestSharpeRollingCorrectness:
    """
    Against the naive reference oracle and frozen golden-master values.
    """

    def test_matches_reference(self) -> None:
        """
        Verifies agreement with the naive closed-form reference over a representative series.
        """
        values = [0.012, -0.008, 0.02, -0.015, 0.005, 0.01, -0.02, 0.018]
        assert_matches(
            apply_expr(
                values, sharpe_ratio_rolling(pl.col(COLUMN_X), 4, periods_per_year=PERIODS, risk_free_rate=0.02)
            ),
            sharpe_ratio_rolling_reference(values, 4, PERIODS, 0.02),
            rel_tol=RELATIVE_TOLERANCE_SCALE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference: the daily-annualized rolling Sharpe over a window of three.
        """
        values = [0.03, -0.01, 0.02, -0.015, 0.025, -0.005, 0.02]
        assert_matches(
            apply_expr(values, sharpe_ratio_rolling(pl.col(COLUMN_X), 3, periods_per_year=252).round(4)),
            [None, None, 10.1678, -1.3977, 7.2837, 1.271, 13.1689],
        )


class TestSharpeRollingProperties:
    """
    Invariants that must hold for all inputs (property-based).
    """

    @given(case=_cases(_VALUE))
    def test_matches_reference_for_any_input(self, case: tuple[list[float], int]) -> None:
        """
        Verifies that, for any well-conditioned series and window, the implementation matches the naive reference.
        """
        values, window = case
        assume(windows_well_spread(values, window))
        assert_matches(
            apply_expr(values, sharpe_ratio_rolling(pl.col(COLUMN_X), window, periods_per_year=PERIODS)),
            sharpe_ratio_rolling_reference(values, window, PERIODS),
            rel_tol=RELATIVE_TOLERANCE_SCALE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    @given(case=_cases(_VALUE_MISSING))
    def test_matches_reference_under_missing_data(self, case: tuple[list[float | None], int]) -> None:
        """
        Verifies that, for inputs freely mixing null / NaN / finite, the implementation matches the naive reference.
        """
        values, window = case
        assume(windows_well_spread(values, window))
        assert_matches(
            apply_expr(values, sharpe_ratio_rolling(pl.col(COLUMN_X), window, periods_per_year=PERIODS)),
            sharpe_ratio_rolling_reference(values, window, PERIODS),
            rel_tol=RELATIVE_TOLERANCE_SCALE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )
