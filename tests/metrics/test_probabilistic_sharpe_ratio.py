"""
Tests for ``pomata.metrics.probabilistic_sharpe_ratio`` — the confidence that the true Sharpe ratio beats a benchmark.

``probabilistic_sharpe_ratio`` is single-input and REDUCING (a return series → one scalar), so tests read the single
output row of ``apply_expr``; ``assert_matches`` and the naive ``probabilistic_sharpe_ratio_reference`` oracle (the
Bailey & López de Prado statistic) are shared across the suite. It is scale-invariant at a zero risk-free rate (a
standardized statistic), so it carries a scale-invariance tier; it is also a probability, so it lies in ``[0, 1]``. As a
function of the Sharpe ratio and the higher moments it is conditioning-sensitive, so the reference comparisons filter
near-constant samples and use the scale band.

The ladder is the canonical one: contract (type / reduces-to-scalar / lazy-eager / ``.over`` per-group independence),
edge (validation / empty / single-row / zero-volatility / null / NaN), correctness (vs the closed-form reference and a
frozen golden master), and properties (reference agreement incl. missing data, scale invariance, the unit-interval
bound). Categories are split into classes; cross-cutting categories use markers.
"""

import math

import polars as pl
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from polars.testing import assert_frame_equal
from tests.metrics.oracles import probabilistic_sharpe_ratio_reference
from tests.support import (
    ABSOLUTE_TOLERANCE_REFERENCE,
    COLUMN_X,
    GROUP_KEY,
    RELATIVE_TOLERANCE_REFERENCE,
    RELATIVE_TOLERANCE_SCALE,
    apply_expr,
    assert_matches,
    missing_data_floats,
    standardized_moment_floats,
    well_spread,
)

from pomata.metrics import probabilistic_sharpe_ratio

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- probabilistic_sharpe_ratio is windowless and REDUCING (M = 0); a case is just a return series. Facts:
#   1. shape   reducing: the output is one scalar (one row in ``select``; one per group under ``.over``)
#   2. domain  standardized_moment_floats(bound): floored off moment underflow; the missing variant mixes null / NaN
#   3. scale   invariant at risk_free_rate 0 (a standardized statistic) -> scale-invariance tier; a probability in [0,1]
# PERIODS / risk-free / benchmark vary over realistic sets in the fuzz. Repetitions N are the shared CI profile.
# ----------------------------------------------------------------------------------------------------------------------
SERIES_MAX = 50
PERIODS = 252
_PERIODS = st.sampled_from([1, 4, 12, 52, 252])
_RISK_FREE = st.sampled_from([0.0, 0.02, 0.05])
_BENCHMARK = st.sampled_from([0.0, 0.05, 0.1])


@st.composite
def _cases[T](draw: st.DrawFn, returns: st.SearchStrategy[T], min_size: int = 1) -> list[T]:
    """A return series sized from the facts above."""
    return draw(st.lists(returns, min_size=min_size, max_size=SERIES_MAX))


class TestProbabilisticSharpeRatioContract:
    """
    Type, shape, and lazy/eager guarantees.
    """

    def test_returns_expr(self) -> None:
        """
        Verifies that the factory returns a ``pl.Expr`` without touching a frame.
        """
        assert isinstance(probabilistic_sharpe_ratio(pl.col(COLUMN_X), periods_per_year=PERIODS), pl.Expr)

    def test_reduces_to_scalar(self) -> None:
        """
        Verifies that the metric reduces a series to one ``Float64`` row.
        """
        frame = pl.DataFrame({COLUMN_X: pl.Series(COLUMN_X, [0.01, -0.02, 0.015, -0.03, 0.02], dtype=pl.Float64)})
        result = frame.select(probabilistic_sharpe_ratio(pl.col(COLUMN_X), periods_per_year=PERIODS).alias("p"))
        assert result.height == 1
        assert result.schema["p"] == pl.Float64

    def test_lazy_eager_parity(self) -> None:
        """
        Verifies that eager and lazy application produce identical materialized output.
        """
        frame = pl.DataFrame({COLUMN_X: pl.Series(COLUMN_X, [0.01, -0.02, 0.015, -0.03, 0.02], dtype=pl.Float64)})
        expr = probabilistic_sharpe_ratio(pl.col(COLUMN_X), periods_per_year=PERIODS).alias("p")
        assert_frame_equal(frame.select(expr), frame.lazy().select(expr).collect())

    def test_over_partitions_independently(self) -> None:
        """
        Verifies that under ``.over`` the statistic is computed per group (broadcast) and never spans boundaries.
        """
        group_a = [0.01, -0.02, 0.015, -0.03, 0.005, 0.04]
        group_b = [0.02, -0.05, 0.01, -0.01, 0.03]
        frame = pl.DataFrame({GROUP_KEY: ["a"] * len(group_a) + ["b"] * len(group_b), COLUMN_X: group_a + group_b})
        grouped = frame.select(
            probabilistic_sharpe_ratio(pl.col(COLUMN_X), periods_per_year=PERIODS).over(GROUP_KEY).alias("p")
        )["p"].to_list()
        expected_a = probabilistic_sharpe_ratio_reference(group_a, PERIODS, 0.0, 0.0)
        expected_b = probabilistic_sharpe_ratio_reference(group_b, PERIODS, 0.0, 0.0)
        assert_matches(
            grouped, [expected_a] * len(group_a) + [expected_b] * len(group_b), rel_tol=RELATIVE_TOLERANCE_REFERENCE
        )


class TestProbabilisticSharpeRatioEdge:
    """
    Validation, boundaries, and null / NaN handling.
    """

    def test_periods_per_year_below_one_raises(self) -> None:
        """
        Verifies that ``periods_per_year < 1`` raises ``ValueError``.
        """
        with pytest.raises(ValueError, match="periods_per_year must be >= 1"):
            probabilistic_sharpe_ratio(pl.col(COLUMN_X), periods_per_year=0)

    def test_non_finite_parameters_raise(self) -> None:
        """
        Verifies that a non-finite ``benchmark_sharpe`` or ``risk_free_rate`` raises ``ValueError``.
        """
        for invalid in (math.nan, math.inf, -math.inf):
            with pytest.raises(ValueError, match="benchmark_sharpe must be a finite number"):
                probabilistic_sharpe_ratio(pl.col(COLUMN_X), periods_per_year=PERIODS, benchmark_sharpe=invalid)
            with pytest.raises(ValueError, match="risk_free_rate must be a finite number"):
                probabilistic_sharpe_ratio(pl.col(COLUMN_X), periods_per_year=PERIODS, risk_free_rate=invalid)

    def test_empty(self) -> None:
        """
        Verifies that an empty series yields ``null``.
        """
        assert_matches(apply_expr([], probabilistic_sharpe_ratio(pl.col(COLUMN_X), periods_per_year=PERIODS)), [None])

    def test_single_row(self) -> None:
        """
        Verifies that a one-element series yields ``null`` (the sample Sharpe ratio needs two observations).
        """
        assert_matches(
            apply_expr([0.05], probabilistic_sharpe_ratio(pl.col(COLUMN_X), periods_per_year=PERIODS)), [None]
        )

    def test_zero_volatility_is_nan(self) -> None:
        """
        Verifies that a constant series has zero dispersion (and undefined higher moments), so the result is ``NaN``.
        """
        assert_matches(
            apply_expr(
                [0.01, 0.01, 0.01, 0.01], probabilistic_sharpe_ratio(pl.col(COLUMN_X), periods_per_year=PERIODS)
            ),
            [math.nan],
        )

    def test_all_null(self) -> None:
        """
        Verifies that an all-null series yields ``null``.
        """
        assert_matches(
            apply_expr([None, None], probabilistic_sharpe_ratio(pl.col(COLUMN_X), periods_per_year=PERIODS)), [None]
        )

    def test_nan_poisons(self) -> None:
        """
        Verifies that a NaN return poisons the result to NaN.
        """
        values = [0.01, math.nan, -0.02, 0.03]
        assert_matches(
            apply_expr(values, probabilistic_sharpe_ratio(pl.col(COLUMN_X), periods_per_year=PERIODS)), [math.nan]
        )


class TestProbabilisticSharpeRatioCorrectness:
    """
    Against the naive reference oracle and frozen golden-master values.
    """

    def test_matches_reference(self) -> None:
        """
        Verifies agreement with the naive closed-form reference over a representative return series.
        """
        values = [0.012, -0.008, 0.02, -0.015, 0.005, 0.0, -0.02, 0.018, 0.01, -0.004]
        assert_matches(
            apply_expr(
                values,
                probabilistic_sharpe_ratio(pl.col(COLUMN_X), periods_per_year=PERIODS, benchmark_sharpe=0.05),
            ),
            [probabilistic_sharpe_ratio_reference(values, PERIODS, 0.05, 0.0)],
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference: a steady positive series has high confidence of a positive Sharpe ratio, 0.9922.
        """
        values = [0.012, 0.008, 0.015, -0.004, 0.02, 0.006, 0.011, -0.003, 0.014, 0.009]
        assert_matches(
            apply_expr(values, probabilistic_sharpe_ratio(pl.col(COLUMN_X), periods_per_year=252).round(4)), [0.9922]
        )


class TestProbabilisticSharpeRatioProperties:
    """
    Invariants that must hold for all inputs (property-based).
    """

    @given(
        case=_cases(standardized_moment_floats(bound=1e3), min_size=2),
        periods=_PERIODS,
        risk_free=_RISK_FREE,
        benchmark=_BENCHMARK,
    )
    def test_matches_reference_for_any_input(
        self, case: list[float], periods: int, risk_free: float, benchmark: float
    ) -> None:
        """
        Verifies that, for any well-conditioned return series, the implementation matches the naive reference.
        """
        assume(well_spread(case))
        assert_matches(
            apply_expr(
                case,
                probabilistic_sharpe_ratio(
                    pl.col(COLUMN_X), periods_per_year=periods, benchmark_sharpe=benchmark, risk_free_rate=risk_free
                ),
            ),
            [probabilistic_sharpe_ratio_reference(case, periods, benchmark, risk_free)],
            rel_tol=RELATIVE_TOLERANCE_SCALE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    @given(
        case=_cases(missing_data_floats(min_magnitude=1e-3), min_size=0),
        periods=_PERIODS,
        risk_free=_RISK_FREE,
        benchmark=_BENCHMARK,
    )
    def test_matches_reference_under_missing_data(
        self, case: list[float | None], periods: int, risk_free: float, benchmark: float
    ) -> None:
        """
        Verifies that, for well-conditioned inputs freely mixing null / NaN / finite, the implementation matches the
        naive reference.
        """
        assume(well_spread(case))
        assert_matches(
            apply_expr(
                case,
                probabilistic_sharpe_ratio(
                    pl.col(COLUMN_X), periods_per_year=periods, benchmark_sharpe=benchmark, risk_free_rate=risk_free
                ),
            ),
            [probabilistic_sharpe_ratio_reference(case, periods, benchmark, risk_free)],
            rel_tol=RELATIVE_TOLERANCE_SCALE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    @given(
        case=_cases(standardized_moment_floats(bound=1e3), min_size=2), exponent=st.sampled_from([-4, -2, -1, 1, 2, 4])
    )
    def test_scale_invariant(self, case: list[float], exponent: int) -> None:
        """
        Verifies that a positive rescale of the returns leaves the statistic unchanged at a zero risk-free rate (a
        standardized statistic), using powers of two so the rescaling is lossless.
        """
        k = 2.0**exponent
        base = apply_expr(case, probabilistic_sharpe_ratio(pl.col(COLUMN_X), periods_per_year=PERIODS))
        scaled = apply_expr(
            [value * k for value in case], probabilistic_sharpe_ratio(pl.col(COLUMN_X), periods_per_year=PERIODS)
        )
        assert_matches(scaled, base, rel_tol=RELATIVE_TOLERANCE_SCALE, abs_tol=ABSOLUTE_TOLERANCE_REFERENCE)

    @given(case=_cases(standardized_moment_floats(bound=1e3), min_size=2))
    def test_within_unit_interval(self, case: list[float]) -> None:
        """
        Verifies that a defined result is a probability in ``[0, 1]``.
        """
        assume(well_spread(case))
        result = apply_expr(case, probabilistic_sharpe_ratio(pl.col(COLUMN_X), periods_per_year=PERIODS))[0]
        if result is not None and not math.isnan(result):
            assert -ABSOLUTE_TOLERANCE_REFERENCE <= result <= 1.0 + ABSOLUTE_TOLERANCE_REFERENCE
