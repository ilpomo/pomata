"""
Tests for ``pomata.metrics.capture_downside_ratio`` — how much of the benchmark's loss a portfolio captured in down
markets.

``capture_downside_ratio`` is two-input and REDUCING (a return series and a benchmark series → one scalar), so tests
read the single output row of ``materialize``; ``assert_matches`` and the naive ``capture_downside_ratio_reference``
oracle (the geometric annualized portfolio return over the geometric annualized benchmark return on negative-benchmark
periods) are shared across the suite. It is a ratio of two annualized geometric returns, neither scale-homogeneous nor
scale-invariant -- its correctness is pinned by the reference, a golden master, and the metamorphic identity that a
portfolio identical to its benchmark captures exactly one.

The ladder is the canonical one: contract (type / reduces-to-scalar / lazy-eager / ``.over`` per-group independence),
edge (validation / empty / no-down-market / null misalignment / NaN), correctness (vs the closed-form reference and a
frozen golden master), and properties (reference agreement incl. missing data, the self-capture identity). Categories
are split into classes; cross-cutting categories use markers.
"""

import math
from collections.abc import Sequence

import polars as pl
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from polars.testing import assert_frame_equal
from tests.metrics.oracles import capture_downside_ratio_reference
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
)

from pomata.metrics import capture_downside_ratio

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- capture_downside_ratio is windowless and REDUCING (M = 0); a case is a list of (return, benchmark)
# pairs.
#   1. shape   reducing: the output is one scalar (one row in ``select``; one per group under ``.over``)
#   2. domain  magnitude-bounded returns (``|r|`` in [0.01, 0.5], sign-varied), so the geometric power avoids
#              the near-one catastrophic cancellation; the missing variant mixes null/NaN per leg
#   3. scale   neither (a ratio of two annualized geometric returns) -> reference + self-capture identity
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


def _has_substantial_loss(returns: Sequence[float | None]) -> bool:
    """Whether any return is clearly negative (so ``1 + r`` is below one and the geometric growth is non-degenerate)."""
    return any(value is not None and not math.isnan(value) and value < -1e-3 for value in returns)


class TestCaptureDownsideContract:
    """
    Type, shape, and lazy/eager guarantees.
    """

    def test_returns_expr(self) -> None:
        """
        Verifies that the factory returns a ``pl.Expr`` without touching a frame.
        """
        assert isinstance(capture_downside_ratio(pl.col(RETURNS), pl.col(BENCHMARK), periods_per_year=PERIODS), pl.Expr)

    def test_reduces_to_scalar(self) -> None:
        """
        Verifies that the metric reduces the two series to one ``Float64`` row.
        """
        frame = pl.DataFrame(
            {
                RETURNS: pl.Series(RETURNS, [0.02, -0.02, 0.03, -0.01], dtype=pl.Float64),
                BENCHMARK: pl.Series(BENCHMARK, [0.015, -0.015, 0.025, -0.008], dtype=pl.Float64),
            }
        )
        result = frame.select(
            capture_downside_ratio(pl.col(RETURNS), pl.col(BENCHMARK), periods_per_year=PERIODS).alias("c")
        )
        assert result.height == 1
        assert result.schema["c"] == pl.Float64

    def test_lazy_eager_parity(self) -> None:
        """
        Verifies that eager and lazy application produce identical materialized output.
        """
        frame = pl.DataFrame(
            {
                RETURNS: pl.Series(RETURNS, [0.02, -0.02, 0.03, -0.01], dtype=pl.Float64),
                BENCHMARK: pl.Series(BENCHMARK, [0.015, -0.015, 0.025, -0.008], dtype=pl.Float64),
            }
        )
        expr = capture_downside_ratio(pl.col(RETURNS), pl.col(BENCHMARK), periods_per_year=PERIODS).alias("c")
        assert_frame_equal(frame.select(expr), frame.lazy().select(expr).collect())

    def test_over_partitions_independently(self) -> None:
        """
        Verifies that under ``.over`` the ratio is computed per group (broadcast) and never spans boundaries.
        """
        returns_a = [0.02, -0.02, 0.03, -0.01, 0.04]
        benchmark_a = [0.015, -0.015, 0.025, -0.008, 0.03]
        returns_b = [0.05, -0.03, 0.02, -0.01]
        benchmark_b = [0.04, -0.02, 0.015, -0.005]
        frame = pl.DataFrame(
            {
                GROUP_KEY: ["a"] * len(returns_a) + ["b"] * len(returns_b),
                RETURNS: returns_a + returns_b,
                BENCHMARK: benchmark_a + benchmark_b,
            }
        )
        grouped = frame.select(
            capture_downside_ratio(pl.col(RETURNS), pl.col(BENCHMARK), periods_per_year=4).over(GROUP_KEY).alias("c")
        )["c"].to_list()
        expected_a = capture_downside_ratio_reference(returns_a, benchmark_a, 4)
        expected_b = capture_downside_ratio_reference(returns_b, benchmark_b, 4)
        assert_matches(
            grouped, [expected_a] * len(returns_a) + [expected_b] * len(returns_b), rel_tol=RELATIVE_TOLERANCE_REFERENCE
        )


class TestCaptureDownsideEdge:
    """
    Validation, boundaries, and null / NaN handling.
    """

    def test_periods_per_year_below_one_raises(self) -> None:
        """
        Verifies that ``periods_per_year < 1`` raises ``ValueError``.
        """
        with pytest.raises(ValueError, match="periods_per_year must be >= 1"):
            capture_downside_ratio(pl.col(RETURNS), pl.col(BENCHMARK), periods_per_year=0)

    def test_empty(self) -> None:
        """
        Verifies that empty series yield ``null``.
        """
        assert_matches(
            materialize(
                {RETURNS: [], BENCHMARK: []},
                capture_downside_ratio(pl.col(RETURNS), pl.col(BENCHMARK), periods_per_year=PERIODS),
            ),
            [None],
        )

    def test_no_down_market_is_null(self) -> None:
        """
        Verifies that with no negative-benchmark period the ratio is undefined, so the result is ``null``.
        """
        assert_matches(
            materialize(
                {RETURNS: [0.01, 0.02, 0.03], BENCHMARK: [0.01, 0.02, 0.03]},
                capture_downside_ratio(pl.col(RETURNS), pl.col(BENCHMARK), periods_per_year=PERIODS),
            ),
            [None],
        )

    def test_all_null(self) -> None:
        """
        Verifies that all-null series yield ``null``.
        """
        assert_matches(
            materialize(
                {RETURNS: [None, None], BENCHMARK: [None, None]},
                capture_downside_ratio(pl.col(RETURNS), pl.col(BENCHMARK), periods_per_year=PERIODS),
            ),
            [None],
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
                capture_downside_ratio(pl.col(RETURNS), pl.col(BENCHMARK), periods_per_year=PERIODS),
            ),
            [math.nan],
        )


class TestCaptureDownsideCorrectness:
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
                capture_downside_ratio(pl.col(RETURNS), pl.col(BENCHMARK), periods_per_year=PERIODS),
            ),
            [capture_downside_ratio_reference(returns, benchmark, PERIODS)],
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference: the daily-annualized downside capture ratio is 1.0224.
        """
        returns = [0.012, -0.008, 0.02, -0.015, 0.005, 0.0, -0.02, 0.018]
        benchmark = [0.01, -0.006, 0.018, -0.012, 0.004, 0.002, -0.018, 0.015]
        assert_matches(
            materialize(
                {RETURNS: returns, BENCHMARK: benchmark},
                capture_downside_ratio(pl.col(RETURNS), pl.col(BENCHMARK), periods_per_year=252).round(4),
            ),
            [1.0224],
        )


class TestCaptureDownsideProperties:
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
                capture_downside_ratio(pl.col(RETURNS), pl.col(BENCHMARK), periods_per_year=periods),
            ),
            [capture_downside_ratio_reference(returns, benchmark, periods)],
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
                capture_downside_ratio(pl.col(RETURNS), pl.col(BENCHMARK), periods_per_year=periods),
            ),
            [capture_downside_ratio_reference(returns, benchmark, periods)],
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    @given(case=_cases(_PAIR, min_size=1), periods=_PERIODS)
    def test_self_capture_is_one(self, case: list[tuple[float, float]], periods: int) -> None:
        """
        Verifies the metamorphic identity: a portfolio identical to its benchmark captures exactly one of its downside.
        """
        returns, _ = split_pairs(case)
        assume(_has_substantial_loss(returns))
        assert_matches(
            materialize(
                {RETURNS: returns, BENCHMARK: returns},
                capture_downside_ratio(pl.col(RETURNS), pl.col(BENCHMARK), periods_per_year=periods),
            ),
            [1.0],
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
        )
