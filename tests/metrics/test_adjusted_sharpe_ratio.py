"""
Tests for ``pomata.metrics.adjusted_sharpe_ratio`` — the Sharpe ratio penalized for skewness and kurtosis.

``adjusted_sharpe_ratio`` is single-input and REDUCING (a return series → one scalar), so tests read the single output
row of ``apply_expr``; ``assert_matches`` and the naive ``adjusted_sharpe_ratio_reference`` oracle are shared across the
suite. At ``risk_free_rate = 0`` it is scale-invariant (the Sharpe ratio and the standardized moments are), so it
carries a scale-invariance tier; as a function of moments it is conditioning-sensitive, so the reference comparisons
filter near-constant samples and use the scale band.

The ladder is the canonical one: contract (type / reduces-to-scalar / lazy-eager / ``.over`` per-group independence),
edge (validation / empty / single-row / zero-volatility / null / NaN), correctness (vs the closed-form reference and a
frozen golden master), and properties (reference agreement incl. missing data, scale invariance). Categories are split
into classes; cross-cutting categories use markers.
"""

import math

import polars as pl
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from polars.testing import assert_frame_equal
from tests.metrics.oracles import adjusted_sharpe_ratio_reference
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

from pomata.metrics import adjusted_sharpe_ratio

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- adjusted_sharpe_ratio is windowless and REDUCING (M = 0); a case is just a return series. Facts:
#   1. shape   reducing: the output is one scalar (one row in ``select``; one per group under ``.over``)
#   2. domain  standardized_moment_floats(bound): floored off moment underflow; the missing variant mixes null / NaN
#   3. scale   invariant at risk_free_rate 0 (Sharpe and standardized moments are) -> scale-invariance tier
# PERIODS / risk-free vary over realistic sets in the fuzz. Repetitions N are the shared CI profile.
# ----------------------------------------------------------------------------------------------------------------------
SERIES_MAX = 50
PERIODS = 252
_PERIODS = st.sampled_from([1, 4, 12, 52, 252])
_RISK_FREE = st.sampled_from([0.0, 0.02, 0.05])


@st.composite
def _cases[T](draw: st.DrawFn, returns: st.SearchStrategy[T], min_size: int = 1) -> list[T]:
    """A return series sized from the facts above."""
    return draw(st.lists(returns, min_size=min_size, max_size=SERIES_MAX))


class TestAdjustedSharpeRatioContract:
    """
    Type, shape, and lazy/eager guarantees.
    """

    def test_returns_expr(self) -> None:
        """
        Verifies that the factory returns a ``pl.Expr`` without touching a frame.
        """
        assert isinstance(adjusted_sharpe_ratio(pl.col(COLUMN_X), periods_per_year=PERIODS), pl.Expr)

    def test_reduces_to_scalar(self) -> None:
        """
        Verifies that the metric reduces a series to one ``Float64`` row.
        """
        frame = pl.DataFrame({COLUMN_X: pl.Series(COLUMN_X, [0.01, -0.02, 0.015, -0.03, 0.02], dtype=pl.Float64)})
        result = frame.select(adjusted_sharpe_ratio(pl.col(COLUMN_X), periods_per_year=PERIODS).alias("a"))
        assert result.height == 1
        assert result.schema["a"] == pl.Float64

    def test_lazy_eager_parity(self) -> None:
        """
        Verifies that eager and lazy application produce identical materialized output.
        """
        frame = pl.DataFrame({COLUMN_X: pl.Series(COLUMN_X, [0.01, -0.02, 0.015, -0.03, 0.02], dtype=pl.Float64)})
        expr = adjusted_sharpe_ratio(pl.col(COLUMN_X), periods_per_year=PERIODS).alias("a")
        assert_frame_equal(frame.select(expr), frame.lazy().select(expr).collect())

    def test_over_partitions_independently(self) -> None:
        """
        Verifies that under ``.over`` the ratio is computed per group (broadcast) and never spans boundaries.
        """
        group_a = [0.01, -0.02, 0.015, -0.03, 0.005, 0.04]
        group_b = [0.02, -0.05, 0.01, -0.01, 0.03]
        frame = pl.DataFrame({GROUP_KEY: ["a"] * len(group_a) + ["b"] * len(group_b), COLUMN_X: group_a + group_b})
        grouped = frame.select(
            adjusted_sharpe_ratio(pl.col(COLUMN_X), periods_per_year=PERIODS).over(GROUP_KEY).alias("a")
        )["a"].to_list()
        expected_a = adjusted_sharpe_ratio_reference(group_a, PERIODS, 0.0)
        expected_b = adjusted_sharpe_ratio_reference(group_b, PERIODS, 0.0)
        assert_matches(
            grouped, [expected_a] * len(group_a) + [expected_b] * len(group_b), rel_tol=RELATIVE_TOLERANCE_REFERENCE
        )


class TestAdjustedSharpeRatioEdge:
    """
    Validation, boundaries, and null / NaN handling.
    """

    def test_periods_per_year_below_one_raises(self) -> None:
        """
        Verifies that ``periods_per_year < 1`` raises ``ValueError``.
        """
        with pytest.raises(ValueError, match="periods_per_year must be >= 1"):
            adjusted_sharpe_ratio(pl.col(COLUMN_X), periods_per_year=0)

    def test_non_finite_risk_free_rate_raises(self) -> None:
        """
        Verifies that a non-finite ``risk_free_rate`` raises ``ValueError``.
        """
        for invalid in (math.nan, math.inf, -math.inf):
            with pytest.raises(ValueError, match="risk_free_rate must be a finite number"):
                adjusted_sharpe_ratio(pl.col(COLUMN_X), periods_per_year=PERIODS, risk_free_rate=invalid)

    def test_empty(self) -> None:
        """
        Verifies that an empty series yields ``null``.
        """
        assert_matches(apply_expr([], adjusted_sharpe_ratio(pl.col(COLUMN_X), periods_per_year=PERIODS)), [None])

    def test_single_row(self) -> None:
        """
        Verifies that a one-element series yields ``null`` (the sample Sharpe ratio needs two observations).
        """
        assert_matches(apply_expr([0.05], adjusted_sharpe_ratio(pl.col(COLUMN_X), periods_per_year=PERIODS)), [None])

    def test_zero_volatility_is_nan(self) -> None:
        """
        Verifies that a constant series has an undefined Sharpe ratio and moments, so the result is ``NaN``.
        """
        assert_matches(
            apply_expr([0.01, 0.01, 0.01, 0.01], adjusted_sharpe_ratio(pl.col(COLUMN_X), periods_per_year=PERIODS)),
            [math.nan],
        )

    def test_all_null(self) -> None:
        """
        Verifies that an all-null series yields ``null``.
        """
        assert_matches(
            apply_expr([None, None], adjusted_sharpe_ratio(pl.col(COLUMN_X), periods_per_year=PERIODS)), [None]
        )

    def test_nan_poisons(self) -> None:
        """
        Verifies that a NaN return poisons the result to NaN.
        """
        values = [0.01, math.nan, -0.02, 0.03]
        assert_matches(
            apply_expr(values, adjusted_sharpe_ratio(pl.col(COLUMN_X), periods_per_year=PERIODS)), [math.nan]
        )


class TestAdjustedSharpeRatioCorrectness:
    """
    Against the naive reference oracle and frozen golden-master values.
    """

    def test_matches_reference(self) -> None:
        """
        Verifies agreement with the naive closed-form reference over a representative return series.
        """
        values = [0.012, -0.008, 0.02, -0.015, 0.005, 0.0, -0.02, 0.018]
        assert_matches(
            apply_expr(values, adjusted_sharpe_ratio(pl.col(COLUMN_X), periods_per_year=PERIODS, risk_free_rate=0.02)),
            [adjusted_sharpe_ratio_reference(values, PERIODS, 0.02)],
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference: the skew/kurtosis-adjusted daily-annualized Sharpe of the series is 2.992.
        """
        values = [0.03, -0.02, 0.04, -0.03, 0.02, -0.01, 0.025, -0.015]
        assert_matches(
            apply_expr(values, adjusted_sharpe_ratio(pl.col(COLUMN_X), periods_per_year=252).round(4)), [2.992]
        )


class TestAdjustedSharpeRatioProperties:
    """
    Invariants that must hold for all inputs (property-based).
    """

    @given(case=_cases(standardized_moment_floats(bound=1e3), min_size=2), periods=_PERIODS, risk_free=_RISK_FREE)
    def test_matches_reference_for_any_input(self, case: list[float], periods: int, risk_free: float) -> None:
        """
        Verifies that, for any well-conditioned return series, the implementation matches the naive reference.
        """
        assume(well_spread(case))
        assert_matches(
            apply_expr(
                case, adjusted_sharpe_ratio(pl.col(COLUMN_X), periods_per_year=periods, risk_free_rate=risk_free)
            ),
            [adjusted_sharpe_ratio_reference(case, periods, risk_free)],
            rel_tol=RELATIVE_TOLERANCE_SCALE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    @given(case=_cases(missing_data_floats(min_magnitude=1e-3), min_size=0), periods=_PERIODS, risk_free=_RISK_FREE)
    def test_matches_reference_under_missing_data(
        self, case: list[float | None], periods: int, risk_free: float
    ) -> None:
        """
        Verifies that, for well-conditioned inputs freely mixing null / NaN / finite, the implementation matches the
        naive reference.
        """
        assume(well_spread(case))
        assert_matches(
            apply_expr(
                case, adjusted_sharpe_ratio(pl.col(COLUMN_X), periods_per_year=periods, risk_free_rate=risk_free)
            ),
            [adjusted_sharpe_ratio_reference(case, periods, risk_free)],
            rel_tol=RELATIVE_TOLERANCE_SCALE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    @given(
        case=_cases(standardized_moment_floats(bound=1e3), min_size=2), exponent=st.sampled_from([-4, -2, -1, 1, 2, 4])
    )
    def test_scale_invariance(self, case: list[float], exponent: int) -> None:
        """
        Verifies that a positive rescale of the returns leaves the adjusted Sharpe ratio unchanged at a zero risk-free
        rate, using powers of two so the rescaling is lossless.
        """
        k = 2.0**exponent
        base = apply_expr(case, adjusted_sharpe_ratio(pl.col(COLUMN_X), periods_per_year=PERIODS))
        scaled = apply_expr(
            [value * k for value in case], adjusted_sharpe_ratio(pl.col(COLUMN_X), periods_per_year=PERIODS)
        )
        assert_matches(scaled, base, rel_tol=RELATIVE_TOLERANCE_SCALE, abs_tol=ABSOLUTE_TOLERANCE_REFERENCE)
