"""
Tests for ``pomata.metrics.risk_of_ruin`` — the symmetric gambler's-ruin probability.

``risk_of_ruin`` is single-input and REDUCING (a return series → one scalar), so tests read the single output row of
``apply_expr``; ``assert_matches`` and the naive ``risk_of_ruin_reference`` oracle (from the win rate and bet count) are
shared across the suite. It is scale-invariant, so it carries a scale-invariance tier.

The ladder is the canonical one: contract (type / reduces-to-scalar / lazy-eager / ``.over`` per-group independence),
edge (empty / single-row / all-wins / all-losses / null / NaN), correctness (vs the closed-form reference and a frozen
golden master), and properties (reference agreement incl. missing data, scale invariance). Categories are split into
classes; cross-cutting categories use markers.
"""

import math

import polars as pl
from hypothesis import given
from hypothesis import strategies as st
from tests.metrics.oracles import risk_of_ruin_reference
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

from pomata.metrics import risk_of_ruin

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- risk_of_ruin is windowless and REDUCING (M = 0); a case is just a return series. Facts:
#   1. shape   reducing: the output is one scalar (one row in ``select``; one per group under ``.over``)
#   2. domain  subnormal_safe_floats(bound): finite returns floored off subnormal; the missing variant mixes null / NaN
#   3. scale   invariant (win rate and the bet count are scale-invariant) -> scale-invariance tier
# Repetitions N are the shared CI profile (tests/conftest.py).
# ----------------------------------------------------------------------------------------------------------------------
SERIES_MAX = 50


@st.composite
def _cases[T](draw: st.DrawFn, returns: st.SearchStrategy[T], min_size: int = 1) -> list[T]:
    """A return series sized from the facts above."""
    return draw(st.lists(returns, min_size=min_size, max_size=SERIES_MAX))


class TestRiskOfRuinContract:
    """
    Type, shape, and lazy/eager guarantees.
    """


class TestRiskOfRuinEdge:
    """
    Boundaries and null / NaN handling.
    """

    def test_all_wins_is_zero(self) -> None:
        """
        Verifies that an all-winning series has no ruin risk, so the probability is ``0``.
        """
        assert_matches(apply_expr([0.01, 0.02, 0.03], risk_of_ruin(pl.col(COLUMN_X))), [0.0])

    def test_all_losses_is_one(self) -> None:
        """
        Verifies that an all-losing series is certain ruin, so the probability is ``1``.
        """
        assert_matches(apply_expr([-0.01, -0.02, -0.03], risk_of_ruin(pl.col(COLUMN_X))), [1.0])

    def test_all_zero_is_null(self) -> None:
        """
        Verifies that an all-zero series has no decisive returns, so the probability is ``null``.
        """
        assert_matches(apply_expr([0.0, 0.0, 0.0], risk_of_ruin(pl.col(COLUMN_X))), [None])

    def test_nan_poisons(self) -> None:
        """
        Verifies that a NaN return poisons the result to NaN.
        """
        assert_matches(apply_expr([0.01, math.nan, -0.02, 0.03], risk_of_ruin(pl.col(COLUMN_X))), [math.nan])

    def test_null_skipped(self) -> None:
        """
        Verifies that null returns are skipped, matching the reference.
        """
        values = [0.01, None, 0.02, -0.03, 0.04, None, -0.01]
        assert_matches(
            apply_expr(values, risk_of_ruin(pl.col(COLUMN_X))),
            [risk_of_ruin_reference(values)],
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
        )


class TestRiskOfRuinCorrectness:
    """
    Against the naive reference oracle and frozen golden-master values.
    """

    def test_matches_reference(self) -> None:
        """
        Verifies agreement with the naive closed-form reference over a representative return series.
        """
        values = [0.012, -0.008, 0.02, -0.015, 0.005, -0.02, 0.018]
        assert_matches(
            apply_expr(values, risk_of_ruin(pl.col(COLUMN_X))),
            [risk_of_ruin_reference(values)],
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference: a 0.5 win rate has no edge, so the odds ratio is one and ruin is certain (1.0).
        """
        values = [0.02, -0.01, 0.03, -0.02]
        assert_matches(apply_expr(values, risk_of_ruin(pl.col(COLUMN_X)).round(4)), [1.0])


class TestRiskOfRuinProperties:
    """
    Invariants that must hold for all inputs (property-based).
    """

    @given(case=_cases(subnormal_safe_floats(bound=1e3)))
    def test_matches_reference_for_any_input(self, case: list[float]) -> None:
        """
        Verifies that, for any return series, the implementation matches the naive reference.
        """
        assert_matches(
            apply_expr(case, risk_of_ruin(pl.col(COLUMN_X))),
            [risk_of_ruin_reference(case)],
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    @given(case=_cases(missing_data_floats(min_magnitude=1e-3), min_size=0))
    def test_matches_reference_under_missing_data(self, case: list[float | None]) -> None:
        """
        Verifies that, for inputs freely mixing null / NaN / finite, the implementation matches the naive reference.
        """
        assert_matches(
            apply_expr(case, risk_of_ruin(pl.col(COLUMN_X))),
            [risk_of_ruin_reference(case)],
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    @given(
        case=_cases(subnormal_safe_floats(bound=1e3), min_size=2),
        exponent=st.sampled_from([-4, -3, -2, -1, 1, 2, 3, 4]),
    )
    def test_scale_invariance(self, case: list[float], exponent: int) -> None:
        """
        Verifies that ``risk_of_ruin`` is scale-invariant: scaling every input value by a constant ``k`` leaves the
        output unchanged -- ``risk_of_ruin(k * x) == risk_of_ruin(x)``. ``k`` is a power of two, so the rescale is
        exact and adds no floating-point error.
        """
        k = 2.0**exponent
        base = apply_expr(case, risk_of_ruin(pl.col(COLUMN_X)))
        scaled = apply_expr([value * k for value in case], risk_of_ruin(pl.col(COLUMN_X)))
        assert_matches(scaled, base, rel_tol=RELATIVE_TOLERANCE_SCALE, abs_tol=ABSOLUTE_TOLERANCE_REFERENCE)
