"""
Tests for ``pomata.metrics.alpha_rolling`` — the rolling (windowed) twin of :func:`pomata.metrics.alpha`.

``alpha_rolling`` is two-input and WINDOWED-SERIES-VALUED (a return series and a benchmark series → a series the same
length, one value per trailing window), so tests read the materialized output of ``materialize`` over the ``returns`` /
``benchmark`` columns; ``assert_matches`` and the naive ``alpha_rolling_reference`` oracle (the reducing :func:`alpha`
recomputed over each window) are shared across the suite. A window holding a ``null`` in either leg is ``null`` (it must
hold ``window`` complete pairs); a ``NaN`` in either leg propagates.

The ladder is the canonical one: contract (type / length-preserving / lazy-eager / ``.over`` per-group warm-up), edge
(validation / empty / warm-up / null-in-window / NaN / constant-benchmark), correctness (vs the closed-form reference
and a frozen golden master), and properties (reference agreement for any input and under missing data). Categories are
split into classes; cross-cutting categories use markers.
"""

import math

import polars as pl
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from polars.testing import assert_frame_equal
from tests.metrics.oracles import alpha_rolling_reference
from tests.support import (
    ABSOLUTE_TOLERANCE_REFERENCE,
    BENCHMARK,
    GROUP_KEY,
    RELATIVE_TOLERANCE_REFERENCE,
    RELATIVE_TOLERANCE_SCALE,
    RETURNS,
    WINDOW_MAX,
    assert_matches,
    materialize,
    split_pairs,
    windows_well_conditioned,
)

from pomata.metrics import alpha_rolling

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- alpha_rolling is WINDOWED, series-valued, two-input. Facts:
#   1. shape   length-preserving: one value per row; the first ``window - 1`` rows are warm-up ``null``
#   2. domain  magnitude-bounded returns (``|r|`` in [0.01, 0.1], sign-varied): same-magnitude values keep the one-pass
#              sliding covariance/variance free of cross-window cancellation; the missing variant mixes null / NaN
#   3. window  window_min = 2 (the embedded slope needs two observations) .. WINDOW_MAX
# The property tiers require every window's benchmark to be well-spread so the embedded slope is well-conditioned; the
# one-pass rolling alpha then agrees with the two-pass oracle to a 1e-6 band.
# ----------------------------------------------------------------------------------------------------------------------
PERIODS = 252
_VALUE = st.one_of(
    st.floats(min_value=0.01, max_value=0.1, allow_nan=False, allow_infinity=False),
    st.floats(min_value=-0.1, max_value=-0.01, allow_nan=False, allow_infinity=False),
)
_VALUE_MISSING = st.one_of(st.none(), st.just(math.nan), _VALUE)
_PAIR = st.tuples(_VALUE, _VALUE)
_PAIR_MISSING = st.tuples(_VALUE_MISSING, _VALUE_MISSING)


@st.composite
def _cases[T](draw: st.DrawFn, pairs: st.SearchStrategy[T], window_min: int = 2) -> tuple[list[T], int]:
    """A (list of (return, benchmark) pairs, window) sized so every example has a window of defined output."""
    window = draw(st.integers(min_value=window_min, max_value=WINDOW_MAX))
    defined = draw(st.integers(min_value=window, max_value=2 * window))
    length = (window - 1) + defined
    return draw(st.lists(pairs, min_size=length, max_size=length)), window


class TestAlphaRollingContract:
    """
    Type, shape, and lazy/eager guarantees.
    """

    def test_returns_expr(self) -> None:
        """
        Verifies that the factory returns a ``pl.Expr`` without touching a frame.
        """
        assert isinstance(alpha_rolling(pl.col(RETURNS), pl.col(BENCHMARK), 3, periods_per_year=PERIODS), pl.Expr)

    def test_preserves_length_and_dtype(self) -> None:
        """
        Verifies that the metric maps the two series to a ``Float64`` series of the same length.
        """
        frame = pl.DataFrame(
            {
                RETURNS: pl.Series(RETURNS, [0.01, -0.02, 0.03, -0.01, 0.02], dtype=pl.Float64),
                BENCHMARK: pl.Series(BENCHMARK, [0.008, -0.015, 0.025, -0.008, 0.018], dtype=pl.Float64),
            }
        )
        result = frame.select(alpha_rolling(pl.col(RETURNS), pl.col(BENCHMARK), 3, periods_per_year=PERIODS).alias("a"))
        assert result.height == frame.height
        assert result.schema["a"] == pl.Float64

    def test_lazy_eager_parity(self) -> None:
        """
        Verifies that eager and lazy application produce identical materialized output.
        """
        frame = pl.DataFrame(
            {
                RETURNS: pl.Series(RETURNS, [0.01, -0.02, 0.03, -0.01, 0.02], dtype=pl.Float64),
                BENCHMARK: pl.Series(BENCHMARK, [0.008, -0.015, 0.025, -0.008, 0.018], dtype=pl.Float64),
            }
        )
        expr = alpha_rolling(pl.col(RETURNS), pl.col(BENCHMARK), 3, periods_per_year=PERIODS).alias("a")
        assert_frame_equal(frame.select(expr), frame.lazy().select(expr).collect())

    def test_over_partitions_independently(self) -> None:
        """
        Verifies that under ``.over`` each group warms up independently and the window never spans a boundary.
        """
        returns_a = [0.01, -0.02, 0.03, -0.01]
        benchmark_a = [0.008, -0.015, 0.025, -0.008]
        returns_b = [0.02, -0.05, 0.01, -0.01]
        benchmark_b = [0.018, -0.04, 0.012, -0.008]
        frame = pl.DataFrame(
            {
                GROUP_KEY: ["a"] * len(returns_a) + ["b"] * len(returns_b),
                RETURNS: returns_a + returns_b,
                BENCHMARK: benchmark_a + benchmark_b,
            }
        )
        grouped = frame.select(
            alpha_rolling(pl.col(RETURNS), pl.col(BENCHMARK), 3, periods_per_year=4).over(GROUP_KEY).alias("a")
        )["a"].to_list()
        expected = alpha_rolling_reference(returns_a, benchmark_a, 3, 4) + alpha_rolling_reference(
            returns_b, benchmark_b, 3, 4
        )
        assert_matches(grouped, expected, rel_tol=RELATIVE_TOLERANCE_REFERENCE)


class TestAlphaRollingEdge:
    """
    Validation, boundaries, and null / NaN handling.
    """

    def test_window_below_two_raises(self) -> None:
        """
        Verifies that ``window < 2`` raises ``ValueError``.
        """
        with pytest.raises(ValueError, match="window must be >= 2"):
            alpha_rolling(pl.col(RETURNS), pl.col(BENCHMARK), 1, periods_per_year=PERIODS)

    def test_periods_per_year_below_one_raises(self) -> None:
        """
        Verifies that ``periods_per_year < 1`` raises ``ValueError``.
        """
        with pytest.raises(ValueError, match="periods_per_year must be >= 1"):
            alpha_rolling(pl.col(RETURNS), pl.col(BENCHMARK), 3, periods_per_year=0)

    def test_non_finite_risk_free_rate_raises(self) -> None:
        """
        Verifies that a non-finite ``risk_free_rate`` raises ``ValueError``.
        """
        for invalid in (math.nan, math.inf, -math.inf):
            with pytest.raises(ValueError, match="risk_free_rate must be a finite number"):
                alpha_rolling(pl.col(RETURNS), pl.col(BENCHMARK), 3, periods_per_year=PERIODS, risk_free_rate=invalid)

    def test_empty(self) -> None:
        """
        Verifies that empty series yield an empty result.
        """
        assert (
            materialize(
                {RETURNS: [], BENCHMARK: []},
                alpha_rolling(pl.col(RETURNS), pl.col(BENCHMARK), 3, periods_per_year=PERIODS),
            )
            == []
        )

    def test_warmup_null_count(self) -> None:
        """
        Verifies that the first ``window - 1`` rows are ``null`` and the rest match the reference.
        """
        returns = [0.01, -0.02, 0.03, -0.01, 0.02]
        benchmark = [0.008, -0.015, 0.025, -0.008, 0.018]
        assert_matches(
            materialize(
                {RETURNS: returns, BENCHMARK: benchmark},
                alpha_rolling(pl.col(RETURNS), pl.col(BENCHMARK), 3, periods_per_year=PERIODS),
            ),
            alpha_rolling_reference(returns, benchmark, 3, PERIODS),
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
        )

    def test_null_in_window_is_null(self) -> None:
        """
        Verifies that a window with a ``null`` in either leg yields ``null`` (the window must hold ``window`` pairs).
        """
        returns = [0.01, None, 0.03, -0.01, 0.02]
        benchmark = [0.008, -0.015, 0.025, None, 0.018]
        assert_matches(
            materialize(
                {RETURNS: returns, BENCHMARK: benchmark},
                alpha_rolling(pl.col(RETURNS), pl.col(BENCHMARK), 3, periods_per_year=PERIODS),
            ),
            alpha_rolling_reference(returns, benchmark, 3, PERIODS),
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
        )

    def test_nan_propagates(self) -> None:
        """
        Verifies that a ``NaN`` in either leg of a window propagates to ``NaN`` for the windows that touch it.
        """
        returns = [0.01, math.nan, 0.03, -0.01, 0.02]
        benchmark = [0.008, -0.015, 0.025, -0.008, 0.018]
        assert_matches(
            materialize(
                {RETURNS: returns, BENCHMARK: benchmark},
                alpha_rolling(pl.col(RETURNS), pl.col(BENCHMARK), 3, periods_per_year=PERIODS),
            ),
            alpha_rolling_reference(returns, benchmark, 3, PERIODS),
        )

    def test_constant_benchmark_window_is_nan(self) -> None:
        """
        Verifies that a window whose benchmark is constant makes the embedded slope ``NaN`` (via the
        ``rolling_max == rolling_min`` guard, regardless of the constant's magnitude, here ``0.1``), which propagates.
        """
        returns = [0.01, -0.02, 0.03, -0.01]
        benchmark = [0.1, 0.1, 0.1, 0.1]
        assert_matches(
            materialize(
                {RETURNS: returns, BENCHMARK: benchmark},
                alpha_rolling(pl.col(RETURNS), pl.col(BENCHMARK), 3, periods_per_year=PERIODS),
            ),
            alpha_rolling_reference(returns, benchmark, 3, PERIODS),
        )


class TestAlphaRollingCorrectness:
    """
    Against the naive reference oracle and frozen golden-master values.
    """

    def test_matches_reference(self) -> None:
        """
        Verifies agreement with the naive closed-form reference over a representative pair of series.
        """
        returns = [0.02, -0.01, 0.03, -0.02, 0.015, 0.005, -0.01, 0.02]
        benchmark = [0.015, -0.008, 0.025, -0.015, 0.01, 0.004, -0.012, 0.018]
        assert_matches(
            materialize(
                {RETURNS: returns, BENCHMARK: benchmark},
                alpha_rolling(pl.col(RETURNS), pl.col(BENCHMARK), 4, periods_per_year=PERIODS),
            ),
            alpha_rolling_reference(returns, benchmark, 4, PERIODS),
            rel_tol=RELATIVE_TOLERANCE_SCALE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference: the daily-annualized rolling Jensen's alpha over a window of four.
        """
        returns = [0.02, -0.01, 0.03, -0.02, 0.015, 0.005, -0.01, 0.02]
        benchmark = [0.015, -0.008, 0.025, -0.015, 0.01, 0.004, -0.012, 0.018]
        assert_matches(
            materialize(
                {RETURNS: returns, BENCHMARK: benchmark},
                alpha_rolling(pl.col(RETURNS), pl.col(BENCHMARK), 4, periods_per_year=252).round(4),
            ),
            [None, None, None, -0.0864, -0.0096, -0.0227, 0.4932, 0.7998],
        )


class TestAlphaRollingProperties:
    """
    Invariants that must hold for all inputs (property-based).
    """

    @given(case=_cases(_PAIR))
    def test_matches_reference_for_any_input(self, case: tuple[list[tuple[float, float]], int]) -> None:
        """
        Verifies that, for any well-conditioned pair of series and window, the implementation matches the reference.
        """
        pairs, window = case
        returns, benchmark = split_pairs(pairs)
        assume(windows_well_conditioned(benchmark, window))
        assert_matches(
            materialize(
                {RETURNS: returns, BENCHMARK: benchmark},
                alpha_rolling(pl.col(RETURNS), pl.col(BENCHMARK), window, periods_per_year=PERIODS),
            ),
            alpha_rolling_reference(returns, benchmark, window, PERIODS),
            rel_tol=RELATIVE_TOLERANCE_SCALE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    @given(case=_cases(_PAIR_MISSING))
    def test_matches_reference_under_missing_data(
        self, case: tuple[list[tuple[float | None, float | None]], int]
    ) -> None:
        """
        Verifies that, for inputs freely mixing null / NaN / finite per leg, the implementation matches the reference.
        """
        pairs, window = case
        returns, benchmark = split_pairs(pairs)
        assume(windows_well_conditioned(benchmark, window))
        assert_matches(
            materialize(
                {RETURNS: returns, BENCHMARK: benchmark},
                alpha_rolling(pl.col(RETURNS), pl.col(BENCHMARK), window, periods_per_year=PERIODS),
            ),
            alpha_rolling_reference(returns, benchmark, window, PERIODS),
            rel_tol=RELATIVE_TOLERANCE_SCALE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )
