"""
Tests for ``pomata.metrics.capture_ratio`` — the ratio of upside capture to downside capture.

``capture_ratio`` is two-input and REDUCING (a return series and a benchmark series → one scalar), so tests read the
single output row of ``materialize``; ``assert_matches`` and the naive ``capture_ratio_reference`` oracle (upside
capture over downside capture) are shared across the suite. It is a ratio of two capture ratios, neither
scale-homogeneous nor scale-invariant -- its correctness is pinned by the reference, a golden master, and the
metamorphic identity linking it to :func:`capture_upside_ratio` and :func:`capture_downside_ratio`.

The ladder is the canonical one: contract (type / reduces-to-scalar / lazy-eager / ``.over`` per-group independence),
edge (validation / empty / missing-regime / null misalignment / NaN), correctness (vs the closed-form reference and a
frozen golden master), and properties (reference agreement incl. missing data, the component-definition identity).
Categories are split into classes; cross-cutting categories use markers.
"""

import math

import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st
from tests.metrics.oracles import capture_ratio_reference
from tests.support import (
    ABSOLUTE_TOLERANCE_REFERENCE,
    BENCHMARK,
    RELATIVE_TOLERANCE_PROPERTY,
    RELATIVE_TOLERANCE_REFERENCE,
    RETURNS,
    assert_matches,
    materialize,
    split_pairs,
)

from pomata.metrics import capture_downside_ratio, capture_ratio, capture_upside_ratio

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- capture_ratio is windowless and REDUCING (M = 0); a case is a list of (return, benchmark) pairs.
#   1. shape   reducing: the output is one scalar (one row in ``select``; one per group under ``.over``)
#   2. domain  magnitude-bounded returns (``|r|`` in [0.01, 0.5], sign-varied), so the geometric power avoids
#              the near-one catastrophic cancellation; the missing variant mixes null/NaN per leg
#   3. scale   neither (a ratio of two capture ratios) -> reference + component identity
# PERIODS vary over a realistic set in the fuzz. Repetitions N are the shared CI profile.
# ----------------------------------------------------------------------------------------------------------------------
SERIES_MAX = 50
PERIODS = 252
_PERIODS = st.sampled_from([1, 4, 12, 52, 252])
_VALUE = st.one_of(
    st.floats(min_value=0.01, max_value=0.5, allow_nan=False, allow_infinity=False),
    st.floats(min_value=-0.5, max_value=-0.01, allow_nan=False, allow_infinity=False),
)
_VALUE_MISSING = st.one_of(st.none(), st.just(math.nan), _VALUE)
_PAIR = st.tuples(_VALUE, _VALUE)
_PAIR_MISSING = st.tuples(_VALUE_MISSING, _VALUE_MISSING)


@st.composite
def _cases[T](draw: st.DrawFn, pairs: st.SearchStrategy[T], min_size: int = 1) -> list[T]:
    """A list of (return, benchmark) pairs sized from the facts above."""
    return draw(st.lists(pairs, min_size=min_size, max_size=SERIES_MAX))


class TestCaptureRatioContract:
    """
    Type, shape, and lazy/eager guarantees.
    """


class TestCaptureRatioEdge:
    """
    Validation, boundaries, and null / NaN handling.
    """

    def test_periods_per_year_below_one_raises(self) -> None:
        """
        Verifies that ``periods_per_year < 1`` raises ``ValueError``.
        """
        with pytest.raises(ValueError, match="periods_per_year must be >= 1"):
            capture_ratio(pl.col(RETURNS), pl.col(BENCHMARK), periods_per_year=0)

    def test_missing_regime_is_null(self) -> None:
        """
        Verifies that with no down-market period the downside capture is undefined, so the ratio is ``null``.
        """
        assert_matches(
            materialize(
                {RETURNS: [0.01, 0.02, 0.03], BENCHMARK: [0.01, 0.02, 0.03]},
                capture_ratio(pl.col(RETURNS), pl.col(BENCHMARK), periods_per_year=PERIODS),
            ),
            [None],
        )

    def test_null_misalignment_drops_pair(self) -> None:
        """
        Verifies that an observation with a ``null`` in either leg is dropped, matching the reference over the retained
        pairs.
        """
        returns = [0.01, None, 0.03, -0.01, 0.02]
        benchmark = [0.008, -0.01, None, -0.005, 0.018]
        assert_matches(
            materialize(
                {RETURNS: returns, BENCHMARK: benchmark},
                capture_ratio(pl.col(RETURNS), pl.col(BENCHMARK), periods_per_year=PERIODS),
            ),
            [capture_ratio_reference(returns, benchmark, PERIODS)],
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
        )

    def test_nan_poisons(self) -> None:
        """
        Verifies that a NaN in either leg of a retained pair poisons the result to NaN.
        """
        returns = [0.02, math.nan, 0.03, -0.01]
        benchmark = [0.015, -0.01, 0.025, -0.008]
        assert_matches(
            materialize(
                {RETURNS: returns, BENCHMARK: benchmark},
                capture_ratio(pl.col(RETURNS), pl.col(BENCHMARK), periods_per_year=PERIODS),
            ),
            [math.nan],
        )

    def test_return_below_negative_one_is_nan(self) -> None:
        """
        Verifies that a selected gross return at or below ``-1`` (a wiped-out leg) is out of the geometric-growth
        domain, so the result is a loud ``NaN`` in whichever leg it lands.
        """
        assert_matches(
            materialize(
                {RETURNS: [0.02, -1.5, 0.01], BENCHMARK: [0.01, 0.02, -0.03]},
                capture_ratio(pl.col(RETURNS), pl.col(BENCHMARK), periods_per_year=PERIODS),
            ),
            [math.nan],
        )
        assert_matches(
            materialize(
                {RETURNS: [0.02, -0.03, 0.01], BENCHMARK: [0.01, -1.2, 0.03]},
                capture_ratio(pl.col(RETURNS), pl.col(BENCHMARK), periods_per_year=PERIODS),
            ),
            [math.nan],
        )


class TestCaptureRatioCorrectness:
    """
    Against the naive reference oracle and frozen golden-master values.
    """

    def test_matches_reference(self) -> None:
        """
        Verifies agreement with the naive closed-form reference over a representative pair of series.
        """
        returns = [0.012, -0.008, 0.02, -0.015, 0.005, 0.0, -0.02, 0.018]
        benchmark = [0.01, -0.006, 0.018, -0.012, 0.004, 0.002, -0.018, 0.015]
        assert_matches(
            materialize(
                {RETURNS: returns, BENCHMARK: benchmark},
                capture_ratio(pl.col(RETURNS), pl.col(BENCHMARK), periods_per_year=PERIODS),
            ),
            [capture_ratio_reference(returns, benchmark, PERIODS)],
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference: the daily-annualized capture ratio (upside over downside) is 1.3479.
        """
        returns = [0.012, -0.008, 0.02, -0.015, 0.005, 0.0, -0.02, 0.018]
        benchmark = [0.01, -0.006, 0.018, -0.012, 0.004, 0.002, -0.018, 0.015]
        assert_matches(
            materialize(
                {RETURNS: returns, BENCHMARK: benchmark},
                capture_ratio(pl.col(RETURNS), pl.col(BENCHMARK), periods_per_year=252).round(4),
            ),
            [1.3479],
        )


class TestCaptureRatioProperties:
    """
    Invariants that must hold for all inputs (property-based).
    """

    @given(case=_cases(_PAIR, min_size=1), periods=_PERIODS)
    def test_matches_reference_for_any_input(self, case: list[tuple[float, float]], periods: int) -> None:
        """
        Verifies that, for any pair of series, the implementation matches the naive reference.
        """
        returns, benchmark = split_pairs(case)
        assert_matches(
            materialize(
                {RETURNS: returns, BENCHMARK: benchmark},
                capture_ratio(pl.col(RETURNS), pl.col(BENCHMARK), periods_per_year=periods),
            ),
            [capture_ratio_reference(returns, benchmark, periods)],
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    @given(case=_cases(_PAIR_MISSING, min_size=0), periods=_PERIODS)
    def test_matches_reference_under_missing_data(
        self, case: list[tuple[float | None, float | None]], periods: int
    ) -> None:
        """
        Verifies that, for inputs freely mixing null / NaN / finite per leg, the implementation matches the naive
        reference.
        """
        returns, benchmark = split_pairs(case)
        assert_matches(
            materialize(
                {RETURNS: returns, BENCHMARK: benchmark},
                capture_ratio(pl.col(RETURNS), pl.col(BENCHMARK), periods_per_year=periods),
            ),
            [capture_ratio_reference(returns, benchmark, periods)],
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    @given(case=_cases(_PAIR, min_size=1), periods=_PERIODS)
    def test_matches_component_definition(self, case: list[tuple[float, float]], periods: int) -> None:
        """
        Verifies the metamorphic identity: ``capture_ratio`` equals :func:`capture_upside_ratio` divided by
        :func:`capture_downside_ratio`, computed as separate metrics.
        """
        returns, benchmark = split_pairs(case)
        composed = capture_upside_ratio(
            pl.col(RETURNS), pl.col(BENCHMARK), periods_per_year=periods
        ) / capture_downside_ratio(pl.col(RETURNS), pl.col(BENCHMARK), periods_per_year=periods)
        direct = materialize(
            {RETURNS: returns, BENCHMARK: benchmark},
            capture_ratio(pl.col(RETURNS), pl.col(BENCHMARK), periods_per_year=periods),
        )
        assert_matches(
            direct,
            materialize({RETURNS: returns, BENCHMARK: benchmark}, composed),
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )
