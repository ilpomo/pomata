"""
Tests for ``pomata.metrics.calmar_ratio`` — the compound annual growth rate per unit of maximum drawdown.

``calmar_ratio`` is single-input and REDUCING (an equity series → one scalar), so tests read the single output row of
``apply_expr``; ``assert_matches`` and the naive ``calmar_ratio_reference`` oracle (CAGR over the drawdown magnitude)
are shared across the suite. It reads a normalized growth-factor curve, so it is neither scale-homogeneous nor
scale-invariant -- its correctness is pinned by the reference, a golden master, and the metamorphic identity linking it
to :func:`cagr` and :func:`max_drawdown`.

The ladder is the canonical one: contract (type / reduces-to-scalar / lazy-eager / ``.over`` per-group independence),
edge (validation / empty / single-row / no-drawdown / null / NaN), correctness (vs the closed-form reference and a
frozen golden master), and properties (reference agreement incl. missing data, the component-definition identity).
Categories are split into classes; cross-cutting categories use markers.
"""

import math

import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st
from tests.metrics.oracles import calmar_ratio_reference
from tests.support import (
    ABSOLUTE_TOLERANCE_REFERENCE,
    COLUMN_X,
    RELATIVE_TOLERANCE_PROPERTY,
    RELATIVE_TOLERANCE_REFERENCE,
    apply_expr,
    assert_matches,
)

from pomata.metrics import cagr, calmar_ratio, max_drawdown

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- calmar is windowless and REDUCING (M = 0); a case is just a positive equity series. Facts:
#   1. shape   reducing: the output is one scalar (one row in ``select``; one per group under ``.over``)
#   2. domain  modest positive growth factors in [0.1, 10] so the annualizing power stays finite; missing mixes null/NaN
#   3. scale   neither (a normalized growth factor over a scale-invariant drawdown) -> pinned by reference + identity
# PERIODS is drawn from a realistic set of annualization factors. Repetitions N are the shared CI profile.
# ----------------------------------------------------------------------------------------------------------------------
SERIES_MAX = 50
PERIODS = 252
_EQUITY = st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False)
_EQUITY_MISSING = st.one_of(st.none(), st.just(math.nan), _EQUITY)
_PERIODS = st.sampled_from([1, 4, 12, 52, 252])


@st.composite
def _cases[T](draw: st.DrawFn, equities: st.SearchStrategy[T], min_size: int = 1) -> list[T]:
    """A positive equity series sized from the facts above."""
    return draw(st.lists(equities, min_size=min_size, max_size=SERIES_MAX))


class TestCalmarRatioContract:
    """
    Type, shape, and lazy/eager guarantees.
    """


class TestCalmarRatioEdge:
    """
    Validation, boundaries, and null / NaN handling.
    """

    def test_periods_per_year_below_one_raises(self) -> None:
        """
        Verifies that ``periods_per_year < 1`` raises ``ValueError``.
        """
        with pytest.raises(ValueError, match="periods_per_year must be >= 1"):
            calmar_ratio(pl.col(COLUMN_X), periods_per_year=0)

    def test_single_row_is_nan(self) -> None:
        """
        Verifies that a one-element series has zero growth and zero drawdown, so the ratio is ``0 / 0``, i.e. ``NaN``.
        """
        assert_matches(apply_expr([1.0], calmar_ratio(pl.col(COLUMN_X), periods_per_year=PERIODS)), [math.nan])

    def test_no_drawdown_is_inf(self) -> None:
        """
        Verifies that a monotonically rising curve has zero maximum drawdown with positive growth, so the ratio is
        ``+inf``.
        """
        assert_matches(apply_expr([1.0, 1.1, 1.21], calmar_ratio(pl.col(COLUMN_X), periods_per_year=1)), [math.inf])

    def test_null_skipped(self) -> None:
        """
        Verifies that a ``null`` observation is skipped (excluded from the reduction), matching the reference.
        """
        values = [1.0, 1.1, 1.05, None, 1.15, 1.3, 1.25]
        assert_matches(
            apply_expr(values, calmar_ratio(pl.col(COLUMN_X), periods_per_year=4)),
            [calmar_ratio_reference(values, 4)],
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    def test_nan_poisons(self) -> None:
        """
        Verifies that a NaN equity poisons the result to NaN.
        """
        assert_matches(
            apply_expr([1.1, math.nan, 1.2], calmar_ratio(pl.col(COLUMN_X), periods_per_year=PERIODS)), [math.nan]
        )


class TestCalmarRatioCorrectness:
    """
    Against the naive reference oracle and frozen golden-master values.
    """

    def test_matches_reference(self) -> None:
        """
        Verifies agreement with the naive closed-form reference over a representative equity curve.
        """
        values = [1.0, 1.1, 1.05, 1.2, 1.15, 1.3, 1.25]
        assert_matches(
            apply_expr(values, calmar_ratio(pl.col(COLUMN_X), periods_per_year=4)),
            [calmar_ratio_reference(values, 4)],
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference: the annual growth (0.0492) over the worst drawdown magnitude (0.0455) is 1.0833.
        """
        values = [1.1, 1.05, 1.2, 1.15, 1.3, 1.25, 1.4]
        assert_matches(apply_expr(values, calmar_ratio(pl.col(COLUMN_X), periods_per_year=1).round(4)), [1.0833])


class TestCalmarRatioProperties:
    """
    Invariants that must hold for all inputs (property-based).
    """

    @given(case=_cases(_EQUITY), periods=_PERIODS)
    def test_matches_reference_for_any_input(self, case: list[float], periods: int) -> None:
        """
        Verifies that, for any positive equity series, the implementation matches the naive reference.
        """
        assert_matches(
            apply_expr(case, calmar_ratio(pl.col(COLUMN_X), periods_per_year=periods)),
            [calmar_ratio_reference(case, periods)],
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    @given(case=_cases(_EQUITY_MISSING, min_size=0), periods=_PERIODS)
    def test_matches_reference_under_missing_data(self, case: list[float | None], periods: int) -> None:
        """
        Verifies that, for inputs freely mixing null / NaN / finite, the implementation matches the naive reference.
        """
        assert_matches(
            apply_expr(case, calmar_ratio(pl.col(COLUMN_X), periods_per_year=periods)),
            [calmar_ratio_reference(case, periods)],
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    @given(case=_cases(_EQUITY), periods=_PERIODS)
    def test_matches_component_definition(self, case: list[float], periods: int) -> None:
        """
        Verifies the metamorphic identity: ``calmar_ratio`` equals :func:`cagr` divided by the magnitude of
        :func:`max_drawdown`, computed as separate metrics.
        """
        direct = apply_expr(case, calmar_ratio(pl.col(COLUMN_X), periods_per_year=periods))
        composed = apply_expr(
            case, cagr(pl.col(COLUMN_X), periods_per_year=periods) / max_drawdown(pl.col(COLUMN_X)).abs()
        )
        assert_matches(direct, composed, rel_tol=RELATIVE_TOLERANCE_PROPERTY, abs_tol=ABSOLUTE_TOLERANCE_REFERENCE)
