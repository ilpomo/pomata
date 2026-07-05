"""
Tests for ``pomata.metrics.sortino_ratio`` — the annualized Sortino ratio of a return series.

``sortino_ratio`` is single-input and REDUCING (a return series → one scalar), so tests read the single output row of
``apply_expr``; ``assert_matches`` and the naive ``sortino_ratio_reference`` oracle (the excess mean over the downside
deviation, annualized) are shared across the suite. At ``risk_free_rate = 0`` it is scale-invariant (a ratio of a mean
to a downside deviation), so it carries a scale-invariance tier (run at the default risk-free rate).

The ladder is the canonical one: contract (type / reduces-to-scalar / lazy-eager / ``.over`` per-group independence),
edge (validation / empty / single-row / no-downside / null / NaN), correctness (vs the closed-form reference and a
frozen golden master), and properties (reference agreement incl. missing data, scale invariance). Categories are split
into classes; cross-cutting categories use markers.
"""

import math

import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st
from tests.metrics.oracles import sortino_ratio_reference
from tests.support import (
    ABSOLUTE_TOLERANCE_REFERENCE,
    COLUMN_X,
    RELATIVE_TOLERANCE_PROPERTY,
    RELATIVE_TOLERANCE_REFERENCE,
    RELATIVE_TOLERANCE_SCALE,
    apply_expr,
    assert_matches,
    missing_data_floats,
    subnormal_safe_floats,
)

from pomata.metrics import sortino_ratio

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- sortino is windowless and REDUCING (M = 0); a case is just a return series. Facts:
#   1. shape   reducing: the output is one scalar (one row in ``select``; one per group under ``.over``)
#   2. domain  subnormal_safe_floats(bound): finite returns floored off subnormal so the downside deviation stays
#              well-conditioned; the missing variant mixes null / NaN
#   3. scale   invariant at risk_free_rate 0 (mean over downside deviation) -> scale-invariance tier
# PERIODS / risk-free vary over realistic sets in the reference fuzz. Repetitions N are the shared CI profile.
# ----------------------------------------------------------------------------------------------------------------------
SERIES_MAX = 50
PERIODS = 252
_PERIODS = st.sampled_from([1, 4, 12, 52, 252])
_RISK_FREE = st.sampled_from([0.0, 0.02, 0.05])


@st.composite
def _cases[T](draw: st.DrawFn, returns: st.SearchStrategy[T], min_size: int = 1) -> list[T]:
    """A return series sized from the facts above."""
    return draw(st.lists(returns, min_size=min_size, max_size=SERIES_MAX))


class TestSortinoRatioContract:
    """
    Type, shape, and lazy/eager guarantees.
    """


class TestSortinoRatioEdge:
    """
    Validation, boundaries, and null / NaN handling.
    """

    def test_periods_per_year_below_one_raises(self) -> None:
        """
        Verifies that ``periods_per_year < 1`` raises ``ValueError``.
        """
        with pytest.raises(ValueError, match="periods_per_year must be >= 1"):
            sortino_ratio(pl.col(COLUMN_X), periods_per_year=0)

    def test_non_finite_risk_free_rate_raises(self) -> None:
        """
        Verifies that a non-finite ``risk_free_rate`` raises ``ValueError``.
        """
        for invalid in (math.nan, math.inf, -math.inf):
            with pytest.raises(ValueError, match="risk_free_rate must be a finite number"):
                sortino_ratio(pl.col(COLUMN_X), periods_per_year=PERIODS, risk_free_rate=invalid)

    def test_single_row(self) -> None:
        """
        Verifies that a one-element loss resolves to its annualized ratio (the population downside is defined for one
        observation).
        """
        values = [-0.02]
        assert_matches(
            apply_expr(values, sortino_ratio(pl.col(COLUMN_X), periods_per_year=PERIODS)),
            [sortino_ratio_reference(values, PERIODS, 0.0)],
        )

    def test_no_downside_is_inf(self) -> None:
        """
        Verifies that returns all at or above the target have zero downside with a positive mean, so the ratio is
        ``+inf``.
        """
        assert_matches(
            apply_expr([0.01, 0.02, 0.03], sortino_ratio(pl.col(COLUMN_X), periods_per_year=PERIODS)), [math.inf]
        )

    def test_null_skipped(self) -> None:
        """
        Verifies that a ``null`` observation is skipped (excluded from the reduction), matching the reference.
        """
        values = [0.012, -0.008, 0.02, None, 0.005, 0.0, -0.02, 0.018]
        assert_matches(
            apply_expr(values, sortino_ratio(pl.col(COLUMN_X), periods_per_year=PERIODS, risk_free_rate=0.02)),
            [sortino_ratio_reference(values, PERIODS, 0.02)],
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    def test_nan_poisons(self) -> None:
        """
        Verifies that a NaN return poisons the result to NaN.
        """
        values = [0.01, math.nan, -0.02, 0.03]
        assert_matches(apply_expr(values, sortino_ratio(pl.col(COLUMN_X), periods_per_year=PERIODS)), [math.nan])


class TestSortinoRatioCorrectness:
    """
    Against the naive reference oracle and frozen golden-master values.
    """

    def test_matches_reference(self) -> None:
        """
        Verifies agreement with the naive closed-form reference over a representative return series.
        """
        values = [0.012, -0.008, 0.02, -0.015, 0.005, 0.0, -0.02, 0.018]
        assert_matches(
            apply_expr(values, sortino_ratio(pl.col(COLUMN_X), periods_per_year=PERIODS, risk_free_rate=0.02)),
            [sortino_ratio_reference(values, PERIODS, 0.02)],
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference: the daily-annualized Sortino of the series at a zero risk-free rate is 4.4567.
        """
        values = [0.03, -0.01, 0.02, -0.015, 0.01, 0.005, -0.02]
        assert_matches(apply_expr(values, sortino_ratio(pl.col(COLUMN_X), periods_per_year=252).round(4)), [4.4567])


class TestSortinoRatioProperties:
    """
    Invariants that must hold for all inputs (property-based).
    """

    @given(case=_cases(subnormal_safe_floats(bound=1e3)), periods=_PERIODS, risk_free=_RISK_FREE)
    def test_matches_reference_for_any_input(self, case: list[float], periods: int, risk_free: float) -> None:
        """
        Verifies that, for any return series, the implementation matches the naive reference.
        """
        assert_matches(
            apply_expr(case, sortino_ratio(pl.col(COLUMN_X), periods_per_year=periods, risk_free_rate=risk_free)),
            [sortino_ratio_reference(case, periods, risk_free)],
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    @given(case=_cases(missing_data_floats(min_magnitude=1e-3), min_size=0), periods=_PERIODS, risk_free=_RISK_FREE)
    def test_matches_reference_under_missing_data(
        self, case: list[float | None], periods: int, risk_free: float
    ) -> None:
        """
        Verifies that, for inputs freely mixing null / NaN / finite, the implementation matches the naive reference.
        """
        assert_matches(
            apply_expr(case, sortino_ratio(pl.col(COLUMN_X), periods_per_year=periods, risk_free_rate=risk_free)),
            [sortino_ratio_reference(case, periods, risk_free)],
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    @given(case=_cases(subnormal_safe_floats(bound=1e3), min_size=2), exponent=st.sampled_from([-4, -2, -1, 1, 2, 4]))
    def test_scale_invariance(self, case: list[float], exponent: int) -> None:
        """
        Verifies that a positive rescale of the returns leaves the Sortino ratio at a zero risk-free rate unchanged (a
        mean over a downside deviation), using powers of two so the rescaling is lossless.
        """
        k = 2.0**exponent
        base = apply_expr(case, sortino_ratio(pl.col(COLUMN_X), periods_per_year=PERIODS))
        scaled = apply_expr([value * k for value in case], sortino_ratio(pl.col(COLUMN_X), periods_per_year=PERIODS))
        assert_matches(scaled, base, rel_tol=RELATIVE_TOLERANCE_SCALE, abs_tol=ABSOLUTE_TOLERANCE_REFERENCE)
