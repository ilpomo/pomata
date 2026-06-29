"""
Tests for ``pomata.metrics.beta`` — the regression slope of a portfolio's returns on its benchmark's returns.

``beta`` is two-input and REDUCING (a return series and a benchmark series → one scalar), so tests read the single
output row of ``materialize``; ``assert_matches`` and the naive ``beta_reference`` oracle (population covariance over
benchmark variance, on pairwise-complete observations) are shared across the suite. It is invariant under a joint
rescale of both legs (a ratio of homogeneous-degree-two moments), so it carries a scale-invariance tier.

The ladder is the canonical one: contract (type / reduces-to-scalar / lazy-eager / ``.over`` per-group independence),
edge (empty / single-pair / constant-benchmark / null misalignment / NaN), correctness (vs the closed-form reference and
a frozen golden master), and properties (reference agreement incl. missing data, joint scale invariance). Categories
are split into classes; cross-cutting categories use markers.
"""

import math

import polars as pl
from hypothesis import assume, given
from hypothesis import strategies as st
from polars.testing import assert_frame_equal
from tests.metrics.oracles import beta_reference
from tests.support import (
    ABSOLUTE_TOLERANCE_REFERENCE,
    BENCHMARK,
    GROUP_KEY,
    RELATIVE_TOLERANCE_PROPERTY,
    RELATIVE_TOLERANCE_REFERENCE,
    RELATIVE_TOLERANCE_SCALE,
    RETURNS,
    assert_matches,
    complete_benchmark,
    materialize,
    split_pairs,
    subnormal_safe_floats,
    well_spread,
)

from pomata.metrics import beta

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- beta is windowless and REDUCING (M = 0); a case is a list of (return, benchmark) pairs. Facts:
#   1. shape   reducing: the output is one scalar (one row in ``select``; one per group under ``.over``)
#   2. domain  subnormal_safe_floats(bound=0.1); the missing variant mixes null / NaN independently per leg; the
#              property tiers require a well-spread benchmark so the covariance-over-variance slope is well-conditioned
#   3. scale   invariant under a joint rescale of both legs -> scale-invariance tier
# Repetitions N are the shared CI profile (tests/conftest.py).
# ----------------------------------------------------------------------------------------------------------------------
SERIES_MAX = 50
_VALUE = subnormal_safe_floats(bound=0.1)
_VALUE_MISSING = st.one_of(st.none(), st.just(math.nan), _VALUE)
_PAIR = st.tuples(_VALUE, _VALUE)
_PAIR_MISSING = st.tuples(_VALUE_MISSING, _VALUE_MISSING)


@st.composite
def _cases[T](draw: st.DrawFn, pairs: st.SearchStrategy[T], min_size: int = 1) -> list[T]:
    """A list of (return, benchmark) pairs sized from the facts above."""
    return draw(st.lists(pairs, min_size=min_size, max_size=SERIES_MAX))


class TestBetaContract:
    """
    Type, shape, and lazy/eager guarantees.
    """

    def test_returns_expr(self) -> None:
        """
        Verifies that the factory returns a ``pl.Expr`` without touching a frame.
        """
        assert isinstance(beta(pl.col(RETURNS), pl.col(BENCHMARK)), pl.Expr)

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
        result = frame.select(beta(pl.col(RETURNS), pl.col(BENCHMARK)).alias("b"))
        assert result.height == 1
        assert result.schema["b"] == pl.Float64

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
        expr = beta(pl.col(RETURNS), pl.col(BENCHMARK)).alias("b")
        assert_frame_equal(frame.select(expr), frame.lazy().select(expr).collect())

    def test_over_partitions_independently(self) -> None:
        """
        Verifies that under ``.over`` the slope is computed per group (broadcast) and never spans boundaries.
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
        grouped = frame.select(beta(pl.col(RETURNS), pl.col(BENCHMARK)).over(GROUP_KEY).alias("b"))["b"].to_list()
        expected_a = beta_reference(returns_a, benchmark_a)
        expected_b = beta_reference(returns_b, benchmark_b)
        assert_matches(
            grouped, [expected_a] * len(returns_a) + [expected_b] * len(returns_b), rel_tol=RELATIVE_TOLERANCE_REFERENCE
        )


class TestBetaEdge:
    """
    Boundaries and null / NaN handling.
    """

    def test_empty(self) -> None:
        """
        Verifies that empty series yield ``null``.
        """
        assert_matches(materialize({RETURNS: [], BENCHMARK: []}, beta(pl.col(RETURNS), pl.col(BENCHMARK))), [None])

    def test_single_pair(self) -> None:
        """
        Verifies that a single complete pair yields ``null`` (the slope needs two observations).
        """
        assert_matches(
            materialize({RETURNS: [0.05], BENCHMARK: [0.04]}, beta(pl.col(RETURNS), pl.col(BENCHMARK))), [None]
        )

    def test_constant_benchmark_is_nan(self) -> None:
        """
        Verifies that a constant (zero-variance) benchmark is reported as ``NaN`` regardless of the constant's
        magnitude: the guard detects ``max == min`` rather than relying on the ``cov / var`` cancellation, which is
        exact only for some constants (``0.5``) and leaves a finite residual for others (``0.1``).
        """
        for constant in (0.1, 1.0 / 3.0, 0.123456789):
            result = materialize(
                {RETURNS: [0.01, -0.02, 0.03], BENCHMARK: [constant, constant, constant]},
                beta(pl.col(RETURNS), pl.col(BENCHMARK)),
            )
            assert_matches(result, [math.nan])

    def test_null_misalignment_drops_pair(self) -> None:
        """
        Verifies that an observation with a ``null`` in either leg is dropped, matching the reference over the retained
        pairs.
        """
        returns = [0.01, None, 0.03, -0.01, 0.02]
        benchmark = [0.008, -0.01, None, -0.005, 0.018]
        assert_matches(
            materialize({RETURNS: returns, BENCHMARK: benchmark}, beta(pl.col(RETURNS), pl.col(BENCHMARK))),
            [beta_reference(returns, benchmark)],
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
        )

    def test_all_null(self) -> None:
        """
        Verifies that all-null series yield ``null``.
        """
        assert_matches(
            materialize({RETURNS: [None, None], BENCHMARK: [None, None]}, beta(pl.col(RETURNS), pl.col(BENCHMARK))),
            [None],
        )

    def test_nan_poisons(self) -> None:
        """
        Verifies that a NaN in either leg of a retained pair poisons the result to NaN.
        """
        returns = [0.01, math.nan, 0.03, -0.01]
        benchmark = [0.008, -0.01, 0.025, -0.005]
        assert_matches(
            materialize({RETURNS: returns, BENCHMARK: benchmark}, beta(pl.col(RETURNS), pl.col(BENCHMARK))), [math.nan]
        )


class TestBetaCorrectness:
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
            materialize({RETURNS: returns, BENCHMARK: benchmark}, beta(pl.col(RETURNS), pl.col(BENCHMARK))),
            [beta_reference(returns, benchmark)],
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference: the regression slope of the portfolio on the benchmark is 1.162.
        """
        returns = [0.012, -0.008, 0.02, -0.015, 0.005, 0.0, -0.02, 0.018]
        benchmark = [0.01, -0.006, 0.018, -0.012, 0.004, 0.002, -0.018, 0.015]
        assert_matches(
            materialize({RETURNS: returns, BENCHMARK: benchmark}, beta(pl.col(RETURNS), pl.col(BENCHMARK)).round(4)),
            [1.162],
        )


class TestBetaProperties:
    """
    Invariants that must hold for all inputs (property-based).
    """

    @given(case=_cases(_PAIR, min_size=2))
    def test_matches_reference_for_any_input(self, case: list[tuple[float, float]]) -> None:
        """
        Verifies that, for any well-conditioned pair of series, the implementation matches the naive reference.
        """
        returns, benchmark = split_pairs(case)
        assume(well_spread(complete_benchmark(returns, benchmark)))
        assert_matches(
            materialize({RETURNS: returns, BENCHMARK: benchmark}, beta(pl.col(RETURNS), pl.col(BENCHMARK))),
            [beta_reference(returns, benchmark)],
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    @given(case=_cases(_PAIR_MISSING, min_size=0))
    def test_matches_reference_under_missing_data(self, case: list[tuple[float | None, float | None]]) -> None:
        """
        Verifies that, for inputs freely mixing null / NaN / finite per leg, the implementation matches the naive
        reference.
        """
        returns, benchmark = split_pairs(case)
        assume(well_spread(complete_benchmark(returns, benchmark)))
        assert_matches(
            materialize({RETURNS: returns, BENCHMARK: benchmark}, beta(pl.col(RETURNS), pl.col(BENCHMARK))),
            [beta_reference(returns, benchmark)],
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    @given(case=_cases(_PAIR, min_size=2), exponent=st.sampled_from([-4, -2, -1, 1, 2, 4]))
    def test_scale_invariance(self, case: list[tuple[float, float]], exponent: int) -> None:
        """
        Verifies that a joint positive rescale of both legs leaves the slope unchanged (a ratio of
        homogeneous-degree-two moments), using powers of two so the rescaling is lossless.
        """
        returns, benchmark = split_pairs(case)
        assume(well_spread(complete_benchmark(returns, benchmark)))
        k = 2.0**exponent
        base = materialize({RETURNS: returns, BENCHMARK: benchmark}, beta(pl.col(RETURNS), pl.col(BENCHMARK)))
        scaled = materialize(
            {RETURNS: [x * k for x in returns], BENCHMARK: [y * k for y in benchmark]},
            beta(pl.col(RETURNS), pl.col(BENCHMARK)),
        )
        assert_matches(scaled, base, rel_tol=RELATIVE_TOLERANCE_SCALE, abs_tol=ABSOLUTE_TOLERANCE_REFERENCE)
