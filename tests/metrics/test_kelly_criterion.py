"""
Tests for ``pomata.metrics.kelly_criterion`` — the growth-optimal fraction of capital to risk per bet.

``kelly_criterion`` is single-input and REDUCING (a return series → one scalar), so tests read the single output row of
``apply_expr``; ``assert_matches`` and the naive ``kelly_criterion_reference`` oracle (from the win rate and payoff
ratio) are shared across the suite. It is scale-invariant, so it carries a scale-invariance tier.

The ladder is the canonical one: contract (type / reduces-to-scalar / lazy-eager / ``.over`` per-group independence),
edge (empty / single-row / one-sided / null / NaN), correctness (vs the closed-form reference and a frozen golden
master), and properties (reference agreement incl. missing data, scale invariance). Categories are split into classes;
cross-cutting categories use markers.
"""

import math

import polars as pl
from hypothesis import given
from hypothesis import strategies as st
from tests.metrics.oracles import kelly_criterion_reference
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

from pomata.metrics import kelly_criterion

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- kelly_criterion is windowless and REDUCING (M = 0); a case is just a return series. Facts:
#   1. shape   reducing: the output is one scalar (one row in ``select``; one per group under ``.over``)
#   2. domain  subnormal_safe_floats(bound): finite returns floored off subnormal; the missing variant mixes null / NaN
#   3. scale   invariant (win rate and payoff ratio are scale-invariant) -> scale-invariance tier
# Repetitions N are the shared CI profile (tests/conftest.py).
# ----------------------------------------------------------------------------------------------------------------------
SERIES_MAX = 50


@st.composite
def _cases[T](draw: st.DrawFn, returns: st.SearchStrategy[T], min_size: int = 1) -> list[T]:
    """A return series sized from the facts above."""
    return draw(st.lists(returns, min_size=min_size, max_size=SERIES_MAX))


class TestKellyCriterionContract:
    """
    Type, shape, and lazy/eager guarantees.
    """


class TestKellyCriterionEdge:
    """
    Boundaries and null / NaN handling.
    """

    def test_single_row(self) -> None:
        """
        Verifies that a one-element series is one-sided (a win or a loss, never both), so the payoff ratio is
        undefined and the fraction is ``null``.
        """
        assert_matches(apply_expr([0.02], kelly_criterion(pl.col(COLUMN_X))), [None])

    def test_null_skipped(self) -> None:
        """
        Verifies that null returns are skipped, matching the reference.
        """
        values = [0.01, None, 0.02, -0.03, 0.04, None, -0.01]
        assert_matches(
            apply_expr(values, kelly_criterion(pl.col(COLUMN_X))),
            [kelly_criterion_reference(values)],
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
        )

    def test_nan_poisons(self) -> None:
        """
        Verifies that a NaN return poisons the result to NaN.
        """
        assert_matches(apply_expr([0.01, math.nan, -0.02, 0.03], kelly_criterion(pl.col(COLUMN_X))), [math.nan])

    def test_no_losses_is_null(self) -> None:
        """
        Verifies that an all-positive series has an undefined payoff ratio, so the fraction is ``null``.
        """
        assert_matches(apply_expr([0.01, 0.02, 0.03], kelly_criterion(pl.col(COLUMN_X))), [None])

    def test_no_wins_is_null(self) -> None:
        """
        Verifies that an all-negative series has an undefined payoff ratio, so the fraction is ``null``.
        """
        assert_matches(apply_expr([-0.01, -0.02, -0.03], kelly_criterion(pl.col(COLUMN_X))), [None])


class TestKellyCriterionCorrectness:
    """
    Against the naive reference oracle and frozen golden-master values.
    """

    def test_matches_reference(self) -> None:
        """
        Verifies agreement with the naive closed-form reference over a representative return series.
        """
        values = [0.012, -0.008, 0.02, -0.015, 0.005, -0.02, 0.018]
        assert_matches(
            apply_expr(values, kelly_criterion(pl.col(COLUMN_X))),
            [kelly_criterion_reference(values)],
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference: a win rate of 0.5714 and a payoff ratio of 1.0833 give a Kelly fraction of
        0.1758.
        """
        values = [0.03, -0.01, 0.02, -0.015, 0.01, 0.005, -0.02]
        assert_matches(apply_expr(values, kelly_criterion(pl.col(COLUMN_X)).round(4)), [0.1758])


class TestKellyCriterionProperties:
    """
    Invariants that must hold for all inputs (property-based).
    """

    @given(case=_cases(subnormal_safe_floats(bound=1e3)))
    def test_matches_reference_for_any_input(self, case: list[float]) -> None:
        """
        Verifies that, for any return series, the implementation matches the naive reference.
        """
        assert_matches(
            apply_expr(case, kelly_criterion(pl.col(COLUMN_X))),
            [kelly_criterion_reference(case)],
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    @given(case=_cases(missing_data_floats(min_magnitude=1e-3), min_size=0))
    def test_matches_reference_under_missing_data(self, case: list[float | None]) -> None:
        """
        Verifies that, for inputs freely mixing null / NaN / finite, the implementation matches the naive reference.
        """
        assert_matches(
            apply_expr(case, kelly_criterion(pl.col(COLUMN_X))),
            [kelly_criterion_reference(case)],
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    @given(
        case=_cases(subnormal_safe_floats(bound=1e3), min_size=2),
        exponent=st.sampled_from([-4, -3, -2, -1, 1, 2, 3, 4]),
    )
    def test_scale_invariance(self, case: list[float], exponent: int) -> None:
        """
        Verifies that ``kelly_criterion`` is scale-invariant: scaling every input value by a constant ``k`` leaves
        the output unchanged -- ``kelly_criterion(k * x) == kelly_criterion(x)``. ``k`` is a power of two, so the
        rescale is exact and adds no floating-point error.
        """
        k = 2.0**exponent
        base = apply_expr(case, kelly_criterion(pl.col(COLUMN_X)))
        scaled = apply_expr([value * k for value in case], kelly_criterion(pl.col(COLUMN_X)))
        assert_matches(scaled, base, rel_tol=RELATIVE_TOLERANCE_SCALE, abs_tol=ABSOLUTE_TOLERANCE_REFERENCE)
