"""
Tests for ``pomata.metrics.information_ratio`` — the annualized active return per unit of tracking error.

``information_ratio`` is two-input and REDUCING (a return series and a benchmark series → one scalar), so tests read the
single output row of ``materialize``; ``assert_matches`` and the naive ``information_ratio_reference`` oracle (mean
active return over its sample standard deviation, annualized, on pairwise-complete observations) are shared across the
suite. It is invariant under a joint rescale of both legs (a mean over a standard deviation), so it carries a
scale-invariance tier.

The ladder is the canonical one: contract (type / reduces-to-scalar / lazy-eager / ``.over`` per-group independence),
edge (validation / empty / single-pair / zero-tracking-error / null misalignment / NaN), correctness (vs the closed-form
reference and a frozen golden master), and properties (reference agreement incl. missing data, joint scale invariance).
Categories are split into classes; cross-cutting categories use markers.
"""

import math
from collections.abc import Sequence

import polars as pl
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from tests.metrics.oracles import information_ratio_reference
from tests.support import (
    ABSOLUTE_TOLERANCE_REFERENCE,
    BENCHMARK,
    RELATIVE_TOLERANCE_PROPERTY,
    RELATIVE_TOLERANCE_REFERENCE,
    RELATIVE_TOLERANCE_SCALE,
    RETURNS,
    assert_matches,
    materialize,
    split_pairs,
    subnormal_safe_floats,
    well_spread,
)

from pomata.metrics import information_ratio

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- information_ratio is windowless and REDUCING (M = 0); a case is a list of (return, benchmark) pairs.
#   1. shape   reducing: the output is one scalar (one row in ``select``; one per group under ``.over``)
#   2. domain  subnormal_safe_floats(bound=0.1); the missing variant mixes null / NaN per leg; the property tiers
#              require a well-spread active series so the tracking-error denominator is well-conditioned
#   3. scale   invariant under a joint rescale of both legs -> scale-invariance tier
# PERIODS vary over a realistic set in the fuzz. Repetitions N are the shared CI profile.
# ----------------------------------------------------------------------------------------------------------------------
SERIES_MAX = 50
PERIODS = 252
_PERIODS = st.sampled_from([1, 4, 12, 52, 252])
_VALUE = subnormal_safe_floats(bound=0.1)
_VALUE_MISSING = st.one_of(st.none(), st.just(math.nan), _VALUE)
_PAIR = st.tuples(_VALUE, _VALUE)
_PAIR_MISSING = st.tuples(_VALUE_MISSING, _VALUE_MISSING)


@st.composite
def _cases[T](draw: st.DrawFn, pairs: st.SearchStrategy[T], min_size: int = 1) -> list[T]:
    """A list of (return, benchmark) pairs sized from the facts above."""
    return draw(st.lists(pairs, min_size=min_size, max_size=SERIES_MAX))


def _complete_active(returns: Sequence[float | None], benchmark: Sequence[float | None]) -> list[float | None]:
    """The active returns (portfolio minus benchmark) of the pairwise-complete observations."""
    return [x - y for x, y in zip(returns, benchmark, strict=True) if x is not None and y is not None]


class TestInformationRatioContract:
    """
    Type, shape, and lazy/eager guarantees.
    """


class TestInformationRatioEdge:
    """
    Validation, boundaries, and null / NaN handling.
    """

    def test_periods_per_year_below_one_raises(self) -> None:
        """
        Verifies that ``periods_per_year < 1`` raises ``ValueError``.
        """
        with pytest.raises(ValueError, match="periods_per_year must be >= 1"):
            information_ratio(pl.col(RETURNS), pl.col(BENCHMARK), periods_per_year=0)

    def test_null_misalignment_drops_pair(self) -> None:
        """
        Verifies that a ``null`` in either leg drops that pair (excluded from the reduction), matching the reference.
        """
        returns = [0.012, -0.008, 0.02, None, 0.005, 0.0, -0.02, 0.018]
        benchmark = [0.01, -0.006, 0.018, -0.012, 0.004, 0.002, -0.018, 0.015]
        assert_matches(
            materialize(
                {RETURNS: returns, BENCHMARK: benchmark},
                information_ratio(pl.col(RETURNS), pl.col(BENCHMARK), periods_per_year=PERIODS),
            ),
            [information_ratio_reference(returns, benchmark, PERIODS)],
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    def test_nan_poisons(self) -> None:
        """
        Verifies that a NaN in either leg of a retained pair poisons the result to NaN.
        """
        returns = [0.01, math.nan, 0.03, -0.01]
        benchmark = [0.008, -0.01, 0.025, -0.005]
        assert_matches(
            materialize(
                {RETURNS: returns, BENCHMARK: benchmark},
                information_ratio(pl.col(RETURNS), pl.col(BENCHMARK), periods_per_year=PERIODS),
            ),
            [math.nan],
        )

    def test_single_pair(self) -> None:
        """
        Verifies that a single complete pair yields ``null`` (the tracking error needs two observations).
        """
        assert_matches(
            materialize(
                {RETURNS: [0.05], BENCHMARK: [0.04]},
                information_ratio(pl.col(RETURNS), pl.col(BENCHMARK), periods_per_year=PERIODS),
            ),
            [None],
        )

    def test_zero_tracking_error_is_inf(self) -> None:
        """
        Verifies that a constant active series has zero tracking error with a positive mean, so the ratio is ``+inf``.
        """
        result = materialize(
            {RETURNS: [0.01, 0.01, 0.01], BENCHMARK: [0.0, 0.0, 0.0]},
            information_ratio(pl.col(RETURNS), pl.col(BENCHMARK), periods_per_year=PERIODS),
        )
        assert_matches(result, [math.inf])


class TestInformationRatioCorrectness:
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
                information_ratio(pl.col(RETURNS), pl.col(BENCHMARK), periods_per_year=PERIODS),
            ),
            [information_ratio_reference(returns, benchmark, PERIODS)],
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference: the daily-annualized information ratio of the series is -0.842.
        """
        returns = [0.012, -0.008, 0.02, -0.015, 0.005, 0.0, -0.02, 0.018]
        benchmark = [0.01, -0.006, 0.018, -0.012, 0.004, 0.002, -0.018, 0.015]
        assert_matches(
            materialize(
                {RETURNS: returns, BENCHMARK: benchmark},
                information_ratio(pl.col(RETURNS), pl.col(BENCHMARK), periods_per_year=252).round(4),
            ),
            [-0.842],
        )


class TestInformationRatioProperties:
    """
    Invariants that must hold for all inputs (property-based).
    """

    @given(case=_cases(_PAIR, min_size=2), periods=_PERIODS)
    def test_matches_reference_for_any_input(self, case: list[tuple[float, float]], periods: int) -> None:
        """
        Verifies that, for any well-conditioned pair of series, the implementation matches the naive reference.
        """
        returns, benchmark = split_pairs(case)
        assume(well_spread(_complete_active(returns, benchmark)))
        assert_matches(
            materialize(
                {RETURNS: returns, BENCHMARK: benchmark},
                information_ratio(pl.col(RETURNS), pl.col(BENCHMARK), periods_per_year=periods),
            ),
            [information_ratio_reference(returns, benchmark, periods)],
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
        assume(well_spread(_complete_active(returns, benchmark)))
        assert_matches(
            materialize(
                {RETURNS: returns, BENCHMARK: benchmark},
                information_ratio(pl.col(RETURNS), pl.col(BENCHMARK), periods_per_year=periods),
            ),
            [information_ratio_reference(returns, benchmark, periods)],
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    @given(case=_cases(_PAIR, min_size=2), exponent=st.sampled_from([-4, -3, -2, -1, 1, 2, 3, 4]))
    def test_scale_invariance(self, case: list[tuple[float, float]], exponent: int) -> None:
        """
        Verifies that ``information_ratio`` is scale-invariant: scaling every input value by a constant ``k`` leaves
        the output unchanged -- ``information_ratio(k * x) == information_ratio(x)``. ``k`` is a power of two, so
        the rescale is exact and adds no floating-point error.
        """
        returns, benchmark = split_pairs(case)
        assume(well_spread(_complete_active(returns, benchmark)))
        k = 2.0**exponent
        base = materialize(
            {RETURNS: returns, BENCHMARK: benchmark},
            information_ratio(pl.col(RETURNS), pl.col(BENCHMARK), periods_per_year=PERIODS),
        )
        scaled = materialize(
            {RETURNS: [x * k for x in returns], BENCHMARK: [y * k for y in benchmark]},
            information_ratio(pl.col(RETURNS), pl.col(BENCHMARK), periods_per_year=PERIODS),
        )
        assert_matches(scaled, base, rel_tol=RELATIVE_TOLERANCE_SCALE, abs_tol=ABSOLUTE_TOLERANCE_REFERENCE)
