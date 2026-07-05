"""
Tests for ``pomata.metrics.beta_rolling`` — the rolling (windowed) twin of :func:`pomata.metrics.beta`.

``beta_rolling`` is two-input and WINDOWED-SERIES-VALUED (a return series and a benchmark series → a series the same
length, one slope per trailing window), so tests read the materialized output of ``materialize`` over the ``returns`` /
``benchmark`` columns; ``assert_matches`` and the naive ``beta_rolling_reference`` oracle (the reducing :func:`beta`
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
from tests.metrics.oracles import beta_rolling_reference
from tests.support import (
    ABSOLUTE_TOLERANCE_REFERENCE,
    BENCHMARK,
    RELATIVE_TOLERANCE_REFERENCE,
    RELATIVE_TOLERANCE_SCALE,
    RETURNS,
    WINDOW_MAX,
    assert_matches,
    materialize,
    split_pairs,
    windows_well_conditioned,
)

from pomata.metrics import beta_rolling

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- beta_rolling is WINDOWED, series-valued, two-input. Facts:
#   1. shape   length-preserving: one slope per row; the first ``window - 1`` rows are warm-up ``null``
#   2. domain  magnitude-bounded returns (``|r|`` in [0.01, 0.1], sign-varied): same-magnitude values keep the one-pass
#              sliding covariance/variance free of cross-window cancellation; the missing variant mixes null / NaN
#   3. window  window_min = 2 (a covariance/variance needs two observations) .. WINDOW_MAX
# The property tiers require every window's benchmark to be well-spread so the cov-over-var slope is well-conditioned;
# the one-pass rolling slope then agrees with the two-pass oracle to a 1e-6 band.
# ----------------------------------------------------------------------------------------------------------------------
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


class TestBetaRollingContract:
    """
    Type, shape, and lazy/eager guarantees.
    """


class TestBetaRollingEdge:
    """
    Validation, boundaries, and null / NaN handling.
    """

    def test_window_below_two_raises(self) -> None:
        """
        Verifies that ``window < 2`` raises ``ValueError``.
        """
        with pytest.raises(ValueError, match="window must be >= 2"):
            beta_rolling(pl.col(RETURNS), pl.col(BENCHMARK), 1)

    def test_warmup_null_count(self) -> None:
        """
        Verifies that the first ``window - 1`` rows are ``null`` and the rest match the reference.
        """
        returns = [0.01, -0.02, 0.03, -0.01, 0.02]
        benchmark = [0.008, -0.015, 0.025, -0.008, 0.018]
        assert_matches(
            materialize({RETURNS: returns, BENCHMARK: benchmark}, beta_rolling(pl.col(RETURNS), pl.col(BENCHMARK), 3)),
            beta_rolling_reference(returns, benchmark, 3),
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
        )

    def test_null_in_window_is_null(self) -> None:
        """
        Verifies that a window with a ``null`` in either leg yields ``null`` (the window must hold ``window`` pairs).
        """
        returns = [0.01, None, 0.03, -0.01, 0.02]
        benchmark = [0.008, -0.015, 0.025, None, 0.018]
        assert_matches(
            materialize({RETURNS: returns, BENCHMARK: benchmark}, beta_rolling(pl.col(RETURNS), pl.col(BENCHMARK), 3)),
            beta_rolling_reference(returns, benchmark, 3),
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
        )

    def test_nan_propagates(self) -> None:
        """
        Verifies that a ``NaN`` in either leg of a window propagates to ``NaN`` for the windows that touch it.
        """
        returns = [0.01, math.nan, 0.03, -0.01, 0.02]
        benchmark = [0.008, -0.015, 0.025, -0.008, 0.018]
        assert_matches(
            materialize({RETURNS: returns, BENCHMARK: benchmark}, beta_rolling(pl.col(RETURNS), pl.col(BENCHMARK), 3)),
            beta_rolling_reference(returns, benchmark, 3),
        )

    def test_constant_benchmark_window_is_nan(self) -> None:
        """
        Verifies that a window whose benchmark is constant has zero variance, so the slope is ``NaN`` -- detected via
        the ``rolling_max == rolling_min`` guard regardless of the constant's magnitude (here ``0.1``, whose two-pass
        variance residual is not exactly zero).
        """
        returns = [0.01, -0.02, 0.03, -0.01]
        benchmark = [0.1, 0.1, 0.1, 0.1]
        assert_matches(
            materialize({RETURNS: returns, BENCHMARK: benchmark}, beta_rolling(pl.col(RETURNS), pl.col(BENCHMARK), 3)),
            beta_rolling_reference(returns, benchmark, 3),
        )

    def test_null_in_constant_benchmark_window_is_null(self) -> None:
        """
        Verifies that a window with a ``null`` in the returns leg but a constant benchmark yields ``null`` (the
        pairwise-complete contract), not the ``NaN`` the flat-benchmark branch alone would emit -- the branch is gated
        on the window actually holding complete pairs.
        """
        returns = [0.02, None, 0.03, 0.01, 0.02]
        benchmark = [0.1, 0.1, 0.1, 0.1, 0.1]
        assert_matches(
            materialize({RETURNS: returns, BENCHMARK: benchmark}, beta_rolling(pl.col(RETURNS), pl.col(BENCHMARK), 3)),
            beta_rolling_reference(returns, benchmark, 3),
        )


class TestBetaRollingCorrectness:
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
            materialize({RETURNS: returns, BENCHMARK: benchmark}, beta_rolling(pl.col(RETURNS), pl.col(BENCHMARK), 4)),
            beta_rolling_reference(returns, benchmark, 4),
            rel_tol=RELATIVE_TOLERANCE_SCALE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference: the rolling regression slope over a window of four.
        """
        returns = [0.02, -0.01, 0.03, -0.02, 0.015, 0.005, -0.01, 0.02]
        benchmark = [0.015, -0.008, 0.025, -0.015, 0.01, 0.004, -0.012, 0.018]
        assert_matches(
            materialize(
                {RETURNS: returns, BENCHMARK: benchmark}, beta_rolling(pl.col(RETURNS), pl.col(BENCHMARK), 4).round(4)
            ),
            [None, None, None, 1.2608, 1.2628, 1.2652, 1.2592, 1.0331],
        )


class TestBetaRollingProperties:
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
                {RETURNS: returns, BENCHMARK: benchmark}, beta_rolling(pl.col(RETURNS), pl.col(BENCHMARK), window)
            ),
            beta_rolling_reference(returns, benchmark, window),
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
                {RETURNS: returns, BENCHMARK: benchmark}, beta_rolling(pl.col(RETURNS), pl.col(BENCHMARK), window)
            ),
            beta_rolling_reference(returns, benchmark, window),
            rel_tol=RELATIVE_TOLERANCE_SCALE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )
