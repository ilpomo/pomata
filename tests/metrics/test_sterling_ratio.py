"""
Tests for ``pomata.metrics.sterling_ratio`` — the excess CAGR per unit of average drawdown plus a cushion.

``sterling_ratio`` is single-input and REDUCING (an equity series → one scalar), so tests read the single output row of
``apply_expr``; ``assert_matches`` and the naive ``sterling_ratio_reference`` oracle are shared across the suite. It
reads a normalized growth-factor curve, so it is neither scale-homogeneous nor scale-invariant -- its correctness is
pinned by the reference, a golden master, and the metamorphic identity linking it to :func:`cagr` and
:func:`pain_index`.

The ladder is the canonical one: contract (type / reduces-to-scalar / lazy-eager / ``.over`` per-group independence),
edge (validation / empty / single-row / null / NaN), correctness (vs the closed-form reference and a frozen golden
master), and properties (reference agreement incl. missing data, the component-definition identity). Categories are
split into classes; cross-cutting categories use markers.
"""

import math

import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st
from tests.metrics.oracles import sterling_ratio_reference
from tests.support import (
    ABSOLUTE_TOLERANCE_REFERENCE,
    COLUMN_X,
    RELATIVE_TOLERANCE_PROPERTY,
    RELATIVE_TOLERANCE_REFERENCE,
    apply_expr,
    assert_matches,
)

from pomata.metrics import cagr, pain_index, sterling_ratio

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- sterling_ratio is windowless and REDUCING (M = 0); a case is just a positive equity series. Facts:
#   1. shape   reducing: the output is one scalar (one row in ``select``; one per group under ``.over``)
#   2. domain  modest positive growth factors in [0.1, 10] so the annualizing power stays finite; missing mixes null/NaN
#   3. scale   neither (a normalized growth factor over a scale-invariant average drawdown) -> reference + identity
# PERIODS / risk-free vary over realistic sets in the fuzz. Repetitions N are the shared CI profile.
# ----------------------------------------------------------------------------------------------------------------------
SERIES_MAX = 50
PERIODS = 252
_EQUITY = st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False)
_EQUITY_MISSING = st.one_of(st.none(), st.just(math.nan), _EQUITY)
_PERIODS = st.sampled_from([1, 4, 12, 52, 252])
_RISK_FREE = st.sampled_from([0.0, 0.02, 0.05])


@st.composite
def _cases[T](draw: st.DrawFn, equities: st.SearchStrategy[T], min_size: int = 1) -> list[T]:
    """A positive equity series sized from the facts above."""
    return draw(st.lists(equities, min_size=min_size, max_size=SERIES_MAX))


class TestSterlingRatioContract:
    """
    Type, shape, and lazy/eager guarantees.
    """


class TestSterlingRatioEdge:
    """
    Validation, boundaries, and null / NaN handling.
    """

    def test_periods_per_year_below_one_raises(self) -> None:
        """
        Verifies that ``periods_per_year < 1`` raises ``ValueError``.
        """
        with pytest.raises(ValueError, match="periods_per_year must be >= 1"):
            sterling_ratio(pl.col(COLUMN_X), periods_per_year=0)

    def test_non_finite_parameters_raise(self) -> None:
        """
        Verifies that a non-finite ``risk_free_rate`` or ``excess`` raises ``ValueError``.
        """
        for invalid in (math.nan, math.inf, -math.inf):
            with pytest.raises(ValueError, match="risk_free_rate must be a finite number"):
                sterling_ratio(pl.col(COLUMN_X), periods_per_year=PERIODS, risk_free_rate=invalid)
            with pytest.raises(ValueError, match="excess must be a finite number"):
                sterling_ratio(pl.col(COLUMN_X), periods_per_year=PERIODS, excess=invalid)

    def test_null_skipped(self) -> None:
        """
        Verifies that a ``null`` observation is skipped (excluded from the reduction), matching the reference.
        """
        values = [1.0, 1.1, 1.05, None, 1.15, 1.3, 1.25]
        assert_matches(
            apply_expr(values, sterling_ratio(pl.col(COLUMN_X), periods_per_year=4, risk_free_rate=0.02)),
            [sterling_ratio_reference(values, 4, 0.02, 0.10)],
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    def test_nan_poisons(self) -> None:
        """
        Verifies that a NaN equity poisons the result to NaN.
        """
        assert_matches(
            apply_expr([1.1, math.nan, 1.2], sterling_ratio(pl.col(COLUMN_X), periods_per_year=PERIODS)), [math.nan]
        )

    def test_no_drawdown_is_zero(self) -> None:
        """
        Verifies that a flat single-period growth has zero drawdown and zero growth, so the ratio is ``0`` (the cushion
        keeps the denominator finite).
        """
        assert_matches(apply_expr([1.0], sterling_ratio(pl.col(COLUMN_X), periods_per_year=PERIODS)), [0.0])


class TestSterlingRatioCorrectness:
    """
    Against the naive reference oracle and frozen golden-master values.
    """

    def test_matches_reference(self) -> None:
        """
        Verifies agreement with the naive closed-form reference over a representative equity curve.
        """
        values = [1.0, 1.1, 1.05, 1.2, 1.15, 1.3, 1.25]
        assert_matches(
            apply_expr(values, sterling_ratio(pl.col(COLUMN_X), periods_per_year=4, risk_free_rate=0.02)),
            [sterling_ratio_reference(values, 4, 0.02, 0.10)],
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference: the annual growth over the average drawdown plus the 10% cushion is 0.4175.
        """
        values = [1.1, 1.05, 1.2, 1.15, 1.3, 1.25, 1.4]
        assert_matches(apply_expr(values, sterling_ratio(pl.col(COLUMN_X), periods_per_year=1).round(4)), [0.4175])


class TestSterlingRatioProperties:
    """
    Invariants that must hold for all inputs (property-based).
    """

    @given(case=_cases(_EQUITY), periods=_PERIODS, risk_free=_RISK_FREE)
    def test_matches_reference_for_any_input(self, case: list[float], periods: int, risk_free: float) -> None:
        """
        Verifies that, for any positive equity series, the implementation matches the naive reference.
        """
        assert_matches(
            apply_expr(case, sterling_ratio(pl.col(COLUMN_X), periods_per_year=periods, risk_free_rate=risk_free)),
            [sterling_ratio_reference(case, periods, risk_free, 0.10)],
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    @given(case=_cases(_EQUITY_MISSING, min_size=0), periods=_PERIODS, risk_free=_RISK_FREE)
    def test_matches_reference_under_missing_data(
        self, case: list[float | None], periods: int, risk_free: float
    ) -> None:
        """
        Verifies that, for inputs freely mixing null / NaN / finite, the implementation matches the naive reference.
        """
        assert_matches(
            apply_expr(case, sterling_ratio(pl.col(COLUMN_X), periods_per_year=periods, risk_free_rate=risk_free)),
            [sterling_ratio_reference(case, periods, risk_free, 0.10)],
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    @given(case=_cases(_EQUITY), periods=_PERIODS, risk_free=_RISK_FREE)
    def test_matches_component_definition(self, case: list[float], periods: int, risk_free: float) -> None:
        """
        Verifies the metamorphic identity: ``sterling_ratio`` equals (:func:`cagr` minus the risk-free rate) divided by
        (:func:`pain_index` plus the cushion), computed as separate metrics.
        """
        direct = apply_expr(case, sterling_ratio(pl.col(COLUMN_X), periods_per_year=periods, risk_free_rate=risk_free))
        composed = apply_expr(
            case, (cagr(pl.col(COLUMN_X), periods_per_year=periods) - risk_free) / (pain_index(pl.col(COLUMN_X)) + 0.10)
        )
        assert_matches(direct, composed, rel_tol=RELATIVE_TOLERANCE_PROPERTY, abs_tol=ABSOLUTE_TOLERANCE_REFERENCE)
