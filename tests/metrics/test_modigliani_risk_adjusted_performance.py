"""
Tests for ``pomata.metrics.modigliani_risk_adjusted_performance`` — M-squared, the return rescaled to benchmark risk.

``modigliani_risk_adjusted_performance`` is two-input and REDUCING (a return series and a benchmark series → one
scalar), so tests read the single output row of ``materialize``; ``assert_matches`` and the naive
``modigliani_risk_adjusted_performance_reference`` oracle (the risk-free rate plus the portfolio Sharpe ratio scaled by
the benchmark volatility, on pairwise-complete observations) are shared across the suite. It is an annualized return, so
it is neither scale-homogeneous nor scale-invariant -- its correctness is pinned by the reference, a golden master, and
the metamorphic identity linking it to :func:`sharpe_ratio` and :func:`volatility`.

The ladder is the canonical one: contract (type / reduces-to-scalar / lazy-eager / ``.over`` per-group independence),
edge (validation / empty / single-pair / constant-portfolio / null misalignment / NaN), correctness (vs the closed-form
reference and a frozen golden master), and properties (reference agreement incl. missing data, the component-definition
identity). Categories are split into classes; cross-cutting categories use markers.
"""

import math
from collections.abc import Sequence

import polars as pl
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from polars.testing import assert_frame_equal
from tests.metrics.oracles import modigliani_risk_adjusted_performance_reference
from tests.support import (
    ABSOLUTE_TOLERANCE_REFERENCE,
    BENCHMARK,
    GROUP_KEY,
    RELATIVE_TOLERANCE_PROPERTY,
    RELATIVE_TOLERANCE_REFERENCE,
    RETURNS,
    assert_matches,
    materialize,
    split_pairs,
    subnormal_safe_floats,
    well_spread,
)

from pomata.metrics import modigliani_risk_adjusted_performance, sharpe_ratio, volatility

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- M-squared is windowless and REDUCING (M = 0); a case is a list of (return, benchmark) pairs. Facts:
#   1. shape   reducing: the output is one scalar (one row in ``select``; one per group under ``.over``)
#   2. domain  subnormal_safe_floats(bound=0.1); the missing variant mixes null / NaN per leg; the property tiers
#              require a well-spread portfolio so the embedded Sharpe denominator is well-conditioned
#   3. scale   neither (an annualized return rescaled to benchmark risk) -> reference + component identity
# PERIODS / risk-free vary over realistic sets in the fuzz. Repetitions N are the shared CI profile.
# ----------------------------------------------------------------------------------------------------------------------
SERIES_MAX = 50
PERIODS = 252
_PERIODS = st.sampled_from([1, 4, 12, 52, 252])
_RISK_FREE = st.sampled_from([0.0, 0.02, 0.05])
_VALUE = subnormal_safe_floats(bound=0.1)
_VALUE_MISSING = st.one_of(st.none(), st.just(math.nan), _VALUE)
_PAIR = st.tuples(_VALUE, _VALUE)
_PAIR_MISSING = st.tuples(_VALUE_MISSING, _VALUE_MISSING)


@st.composite
def _cases[T](draw: st.DrawFn, pairs: st.SearchStrategy[T], min_size: int = 1) -> list[T]:
    """A list of (return, benchmark) pairs sized from the facts above."""
    return draw(st.lists(pairs, min_size=min_size, max_size=SERIES_MAX))


def _complete_returns(returns: Sequence[float | None], benchmark: Sequence[float | None]) -> list[float | None]:
    """The portfolio returns of the pairwise-complete observations (the Sharpe component's input)."""
    return [x for x, y in zip(returns, benchmark, strict=True) if x is not None and y is not None]


class TestModiglianiContract:
    """
    Type, shape, and lazy/eager guarantees.
    """

    def test_returns_expr(self) -> None:
        """
        Verifies that the factory returns a ``pl.Expr`` without touching a frame.
        """
        assert isinstance(
            modigliani_risk_adjusted_performance(pl.col(RETURNS), pl.col(BENCHMARK), periods_per_year=PERIODS), pl.Expr
        )

    def test_reduces_to_scalar(self) -> None:
        """
        Verifies that the metric reduces the two series to one ``Float64`` row.
        """
        frame = pl.DataFrame(
            {
                RETURNS: pl.Series(RETURNS, [0.01, -0.02, 0.015, -0.03], dtype=pl.Float64),
                BENCHMARK: pl.Series(BENCHMARK, [0.008, -0.015, 0.012, -0.025], dtype=pl.Float64),
            }
        )
        result = frame.select(
            modigliani_risk_adjusted_performance(pl.col(RETURNS), pl.col(BENCHMARK), periods_per_year=PERIODS).alias(
                "m"
            )
        )
        assert result.height == 1
        assert result.schema["m"] == pl.Float64

    def test_lazy_eager_parity(self) -> None:
        """
        Verifies that eager and lazy application produce identical materialized output.
        """
        frame = pl.DataFrame(
            {
                RETURNS: pl.Series(RETURNS, [0.01, -0.02, 0.015, -0.03], dtype=pl.Float64),
                BENCHMARK: pl.Series(BENCHMARK, [0.008, -0.015, 0.012, -0.025], dtype=pl.Float64),
            }
        )
        expr = modigliani_risk_adjusted_performance(pl.col(RETURNS), pl.col(BENCHMARK), periods_per_year=PERIODS).alias(
            "m"
        )
        assert_frame_equal(frame.select(expr), frame.lazy().select(expr).collect())

    def test_over_partitions_independently(self) -> None:
        """
        Verifies that under ``.over`` the measure is computed per group (broadcast) and never spans boundaries.
        """
        returns_a = [0.01, -0.02, 0.015, -0.03, 0.005]
        benchmark_a = [0.008, -0.015, 0.012, -0.025, 0.004]
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
            modigliani_risk_adjusted_performance(pl.col(RETURNS), pl.col(BENCHMARK), periods_per_year=4)
            .over(GROUP_KEY)
            .alias("m")
        )["m"].to_list()
        expected_a = modigliani_risk_adjusted_performance_reference(returns_a, benchmark_a, 4, 0.0)
        expected_b = modigliani_risk_adjusted_performance_reference(returns_b, benchmark_b, 4, 0.0)
        assert_matches(
            grouped, [expected_a] * len(returns_a) + [expected_b] * len(returns_b), rel_tol=RELATIVE_TOLERANCE_REFERENCE
        )


class TestModiglianiEdge:
    """
    Validation, boundaries, and null / NaN handling.
    """

    def test_periods_per_year_below_one_raises(self) -> None:
        """
        Verifies that ``periods_per_year < 1`` raises ``ValueError``.
        """
        with pytest.raises(ValueError, match="periods_per_year must be >= 1"):
            modigliani_risk_adjusted_performance(pl.col(RETURNS), pl.col(BENCHMARK), periods_per_year=0)

    def test_non_finite_risk_free_rate_raises(self) -> None:
        """
        Verifies that a non-finite ``risk_free_rate`` raises ``ValueError``.
        """
        for invalid in (math.nan, math.inf, -math.inf):
            with pytest.raises(ValueError, match="risk_free_rate must be a finite number"):
                modigliani_risk_adjusted_performance(
                    pl.col(RETURNS), pl.col(BENCHMARK), periods_per_year=PERIODS, risk_free_rate=invalid
                )

    def test_empty(self) -> None:
        """
        Verifies that empty series yield ``null``.
        """
        assert_matches(
            materialize(
                {RETURNS: [], BENCHMARK: []},
                modigliani_risk_adjusted_performance(pl.col(RETURNS), pl.col(BENCHMARK), periods_per_year=PERIODS),
            ),
            [None],
        )

    def test_single_pair(self) -> None:
        """
        Verifies that a single complete pair yields ``null`` (the Sharpe ratio and volatility need two observations).
        """
        assert_matches(
            materialize(
                {RETURNS: [0.05], BENCHMARK: [0.04]},
                modigliani_risk_adjusted_performance(pl.col(RETURNS), pl.col(BENCHMARK), periods_per_year=PERIODS),
            ),
            [None],
        )

    def test_constant_portfolio_is_inf(self) -> None:
        """
        Verifies that a constant portfolio has an infinite Sharpe ratio, which propagates to ``+inf``.
        """
        result = materialize(
            {RETURNS: [0.01, 0.01, 0.01], BENCHMARK: [0.02, -0.01, 0.03]},
            modigliani_risk_adjusted_performance(pl.col(RETURNS), pl.col(BENCHMARK), periods_per_year=PERIODS),
        )
        assert_matches(result, [math.inf])

    def test_all_null(self) -> None:
        """
        Verifies that all-null series yield ``null``.
        """
        assert_matches(
            materialize(
                {RETURNS: [None, None], BENCHMARK: [None, None]},
                modigliani_risk_adjusted_performance(pl.col(RETURNS), pl.col(BENCHMARK), periods_per_year=PERIODS),
            ),
            [None],
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
                modigliani_risk_adjusted_performance(pl.col(RETURNS), pl.col(BENCHMARK), periods_per_year=PERIODS),
            ),
            [math.nan],
        )


class TestModiglianiCorrectness:
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
                modigliani_risk_adjusted_performance(
                    pl.col(RETURNS), pl.col(BENCHMARK), periods_per_year=PERIODS, risk_free_rate=0.02
                ),
            ),
            [modigliani_risk_adjusted_performance_reference(returns, benchmark, PERIODS, 0.02)],
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference: the daily-annualized M-squared at a 2% risk-free rate is 0.3274.
        """
        returns = [0.012, -0.008, 0.02, -0.015, 0.005, 0.0, -0.02, 0.018]
        benchmark = [0.01, -0.006, 0.018, -0.012, 0.004, 0.002, -0.018, 0.015]
        assert_matches(
            materialize(
                {RETURNS: returns, BENCHMARK: benchmark},
                modigliani_risk_adjusted_performance(
                    pl.col(RETURNS), pl.col(BENCHMARK), periods_per_year=252, risk_free_rate=0.02
                ).round(4),
            ),
            [0.3274],
        )


class TestModiglianiProperties:
    """
    Invariants that must hold for all inputs (property-based).
    """

    @given(case=_cases(_PAIR, min_size=2), periods=_PERIODS, risk_free=_RISK_FREE)
    def test_matches_reference_for_any_input(
        self, case: list[tuple[float, float]], periods: int, risk_free: float
    ) -> None:
        """
        Verifies that, for any well-conditioned pair of series, the implementation matches the naive reference.
        """
        returns, benchmark = split_pairs(case)
        assume(well_spread(_complete_returns(returns, benchmark)))
        assert_matches(
            materialize(
                {RETURNS: returns, BENCHMARK: benchmark},
                modigliani_risk_adjusted_performance(
                    pl.col(RETURNS), pl.col(BENCHMARK), periods_per_year=periods, risk_free_rate=risk_free
                ),
            ),
            [modigliani_risk_adjusted_performance_reference(returns, benchmark, periods, risk_free)],
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    @given(case=_cases(_PAIR_MISSING, min_size=0), periods=_PERIODS, risk_free=_RISK_FREE)
    def test_matches_reference_under_missing_data(
        self, case: list[tuple[float | None, float | None]], periods: int, risk_free: float
    ) -> None:
        """
        Verifies that, for inputs freely mixing null / NaN / finite per leg, the implementation matches the naive
        reference.
        """
        returns, benchmark = split_pairs(case)
        assume(well_spread(_complete_returns(returns, benchmark)))
        assert_matches(
            materialize(
                {RETURNS: returns, BENCHMARK: benchmark},
                modigliani_risk_adjusted_performance(
                    pl.col(RETURNS), pl.col(BENCHMARK), periods_per_year=periods, risk_free_rate=risk_free
                ),
            ),
            [modigliani_risk_adjusted_performance_reference(returns, benchmark, periods, risk_free)],
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    @given(case=_cases(_PAIR, min_size=2), periods=_PERIODS, risk_free=_RISK_FREE)
    def test_matches_component_definition(
        self, case: list[tuple[float, float]], periods: int, risk_free: float
    ) -> None:
        """
        Verifies the metamorphic identity: M-squared equals the risk-free rate plus the portfolio :func:`sharpe_ratio`
        ratio scaled by the benchmark :func:`volatility`, computed as separate metrics.
        """
        returns, benchmark = split_pairs(case)
        assume(well_spread(_complete_returns(returns, benchmark)))
        composed = risk_free + sharpe_ratio(
            pl.col(RETURNS), periods_per_year=periods, risk_free_rate=risk_free
        ) * volatility(pl.col(BENCHMARK), periods_per_year=periods)
        direct = materialize(
            {RETURNS: returns, BENCHMARK: benchmark},
            modigliani_risk_adjusted_performance(
                pl.col(RETURNS), pl.col(BENCHMARK), periods_per_year=periods, risk_free_rate=risk_free
            ),
        )
        assert_matches(
            direct,
            materialize({RETURNS: returns, BENCHMARK: benchmark}, composed),
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )
