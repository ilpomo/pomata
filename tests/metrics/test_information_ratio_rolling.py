"""
Tests for ``pomata.metrics.information_ratio_rolling`` — the rolling twin of :func:`pomata.metrics.information_ratio`.

``information_ratio_rolling`` is two-input and WINDOWED-SERIES-VALUED (a return series and a benchmark series → a series
the same length, one value per trailing window), so tests read the materialized output of ``materialize`` over the
``returns`` / ``benchmark`` columns; ``assert_matches`` and the ``information_ratio_rolling_reference`` oracle (the
reducing :func:`information_ratio` recomputed over each window) are shared across the suite. A window holding a ``null``
in either leg is ``null`` (it must hold ``window`` complete pairs); a ``NaN`` in either leg propagates.

The ladder is the canonical one: contract (type / length-preserving / lazy-eager / ``.over`` per-group warm-up), edge
(validation / empty / warm-up / null-in-window / NaN / zero-tracking-error), correctness (vs the closed-form reference
and a frozen golden master), and properties (reference agreement for any input and under missing data). Categories are
split into classes; cross-cutting categories use markers.
"""

import math
from collections.abc import Sequence

import polars as pl
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from tests.metrics.oracles import information_ratio_rolling_reference
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

from pomata.metrics import information_ratio_rolling

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- information_ratio_rolling is WINDOWED, series-valued, two-input. Facts:
#   1. shape   length-preserving: one value per row; the first ``window - 1`` rows are warm-up ``null``
#   2. domain  magnitude-bounded returns (``|r|`` in [0.01, 0.1], sign-varied): same-magnitude values keep the one-pass
#              sliding tracking error free of cross-window cancellation; the missing variant mixes null / NaN per leg
#   3. window  window_min = 2 (a sample tracking error needs two observations) .. WINDOW_MAX
# The property tiers require every window's active return (portfolio minus benchmark) to be well-spread so the tracking-
# error denominator is well-conditioned; the one-pass rolling ratio then agrees with the two-pass oracle to a 1e-6 band.
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


def _active_windows_conditioned(
    returns: Sequence[float | None], benchmark: Sequence[float | None], window: int
) -> bool:
    """Whether every window's active-return variance is a real fraction of its magnitude (sound tracking error)."""
    active = [
        None if value_returns is None or value_benchmark is None else value_returns - value_benchmark
        for value_returns, value_benchmark in zip(returns, benchmark, strict=True)
    ]
    return windows_well_conditioned(active, window)


class TestInformationRatioRollingContract:
    """
    Type, shape, and lazy/eager guarantees.
    """


class TestInformationRatioRollingEdge:
    """
    Validation, boundaries, and null / NaN handling.
    """

    def test_window_below_two_raises(self) -> None:
        """
        Verifies that ``window < 2`` raises ``ValueError`` (a sample tracking error needs two observations).
        """
        with pytest.raises(ValueError, match="window must be >= 2"):
            information_ratio_rolling(pl.col(RETURNS), pl.col(BENCHMARK), 1, periods_per_year=PERIODS)

    def test_periods_per_year_below_one_raises(self) -> None:
        """
        Verifies that ``periods_per_year < 1`` raises ``ValueError``.
        """
        with pytest.raises(ValueError, match="periods_per_year must be >= 1"):
            information_ratio_rolling(pl.col(RETURNS), pl.col(BENCHMARK), 3, periods_per_year=0)

    def test_null_in_window_is_null(self) -> None:
        """
        Verifies that a window with a ``null`` in either leg yields ``null`` (the window must hold ``window`` pairs).
        """
        returns = [0.01, None, 0.03, -0.01, 0.02]
        benchmark = [0.008, -0.015, 0.025, None, 0.018]
        assert_matches(
            materialize(
                {RETURNS: returns, BENCHMARK: benchmark},
                information_ratio_rolling(pl.col(RETURNS), pl.col(BENCHMARK), 3, periods_per_year=PERIODS),
            ),
            information_ratio_rolling_reference(returns, benchmark, 3, PERIODS),
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
                information_ratio_rolling(pl.col(RETURNS), pl.col(BENCHMARK), 3, periods_per_year=PERIODS),
            ),
            information_ratio_rolling_reference(returns, benchmark, 3, PERIODS),
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
                information_ratio_rolling(pl.col(RETURNS), pl.col(BENCHMARK), 3, periods_per_year=PERIODS),
            ),
            information_ratio_rolling_reference(returns, benchmark, 3, PERIODS),
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
        )

    def test_window_exceeds_length(self) -> None:
        """
        Verifies that when ``window`` exceeds the series length the whole output is null (no window ever fills).
        """
        returns = [0.01, -0.02, 0.03]
        benchmark = [0.008, -0.015, 0.025]
        assert_matches(
            materialize(
                {RETURNS: returns, BENCHMARK: benchmark},
                information_ratio_rolling(pl.col(RETURNS), pl.col(BENCHMARK), 5, periods_per_year=PERIODS),
            ),
            [None, None, None],
        )

    def test_window_equals_length(self) -> None:
        """
        Verifies that when ``window`` equals the series length only the last row is defined, matching the reference.
        """
        returns = [0.01, -0.02, 0.03, -0.01]
        benchmark = [0.008, -0.015, 0.025, -0.008]
        result = materialize(
            {RETURNS: returns, BENCHMARK: benchmark},
            information_ratio_rolling(pl.col(RETURNS), pl.col(BENCHMARK), 4, periods_per_year=PERIODS),
        )
        assert result[:-1] == [None, None, None]
        assert result[-1] is not None
        assert_matches(
            result,
            information_ratio_rolling_reference(returns, benchmark, 4, PERIODS),
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
        )

    def test_constant_window_by_slide_is_inf(self) -> None:
        """
        Verifies that a window left bit-constant once a much larger active value slides out has an exactly-zero
        tracking error, so the ratio degenerates to the documented ``inf`` -- not the plausible finite the raw
        incremental ``rolling_std`` residue would fake.
        """
        result = materialize(
            {RETURNS: [1_000_000.0, 0.1, 0.1, 0.1, 0.1], BENCHMARK: [0.0, 0.0, 0.0, 0.0, 0.0]},
            information_ratio_rolling(pl.col(RETURNS), pl.col(BENCHMARK), 3, periods_per_year=PERIODS),
        )
        assert result[:2] == [None, None]
        assert result[3] == math.inf
        assert result[4] == math.inf

    def test_zero_tracking_error_is_inf(self) -> None:
        """
        Verifies that a window with a constant active return has zero tracking error with a positive mean, so the ratio
        is ``+inf``.
        """
        returns = [0.01, 0.01, 0.01, 0.01]
        benchmark = [0.0, 0.0, 0.0, 0.0]
        assert_matches(
            materialize(
                {RETURNS: returns, BENCHMARK: benchmark},
                information_ratio_rolling(pl.col(RETURNS), pl.col(BENCHMARK), 3, periods_per_year=PERIODS),
            ),
            information_ratio_rolling_reference(returns, benchmark, 3, PERIODS),
        )

    def test_flat_zero_active_window_by_slide_is_nan(self) -> None:
        """
        Verifies that a window whose active returns are all exactly zero degenerates to ``0/0 -> NaN`` even after
        larger active values have slid out of the window: the exact rolling mean pins the numerator to zero, so the
        incremental running-sum residue cannot ride above the exactly-zero tracking error as a spurious ``inf``.
        """
        returns = [0.0132, -0.3625, 0.0, 0.4404, 0.0, 0.0, 0.0, 0.0]
        benchmark = [0.3365, 0.2832, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        result = materialize(
            {RETURNS: returns, BENCHMARK: benchmark},
            information_ratio_rolling(pl.col(RETURNS), pl.col(BENCHMARK), 4, periods_per_year=PERIODS),
        )
        assert result[-1] is not None
        assert math.isnan(result[-1])


class TestInformationRatioRollingCorrectness:
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
                information_ratio_rolling(pl.col(RETURNS), pl.col(BENCHMARK), 4, periods_per_year=PERIODS),
            ),
            information_ratio_rolling_reference(returns, benchmark, 4, PERIODS),
            rel_tol=RELATIVE_TOLERANCE_SCALE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference: the daily-annualized rolling information ratio over a window of four.
        """
        returns = [0.02, -0.01, 0.03, -0.02, 0.015, 0.005, -0.01, 0.02]
        benchmark = [0.015, -0.008, 0.025, -0.015, 0.01, 0.004, -0.012, 0.018]
        assert_matches(
            materialize(
                {RETURNS: returns, BENCHMARK: benchmark},
                information_ratio_rolling(pl.col(RETURNS), pl.col(BENCHMARK), 4, periods_per_year=252).round(4),
            ),
            [None, None, None, 2.3539, 2.3539, 5.0387, 2.8393, 22.9129],
        )


class TestInformationRatioRollingProperties:
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
        assume(_active_windows_conditioned(returns, benchmark, window))
        assert_matches(
            materialize(
                {RETURNS: returns, BENCHMARK: benchmark},
                information_ratio_rolling(pl.col(RETURNS), pl.col(BENCHMARK), window, periods_per_year=PERIODS),
            ),
            information_ratio_rolling_reference(returns, benchmark, window, PERIODS),
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
        assume(_active_windows_conditioned(returns, benchmark, window))
        assert_matches(
            materialize(
                {RETURNS: returns, BENCHMARK: benchmark},
                information_ratio_rolling(pl.col(RETURNS), pl.col(BENCHMARK), window, periods_per_year=PERIODS),
            ),
            information_ratio_rolling_reference(returns, benchmark, window, PERIODS),
            rel_tol=RELATIVE_TOLERANCE_SCALE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )
