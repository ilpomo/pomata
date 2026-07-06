"""
Tests for ``pomata.metrics.pain_index`` — the average depth of drawdown over an equity curve.

``pain_index`` is single-input and REDUCING (an equity series → one scalar), so tests read the single output row of
``apply_expr``; ``assert_matches`` and the naive ``pain_index_reference`` oracle (the mean absolute drawdown) are shared
across the suite. It is scale-invariant under a positive rescale (the peak ratio cancels), so it carries a
scale-invariance tier.

The ladder is the canonical one: contract (type / reduces-to-scalar / lazy-eager / ``.over`` per-group independence),
edge (empty / single-row / no-drawdown / null / NaN), correctness (vs the closed-form reference and a frozen golden
master), and properties (reference agreement incl. missing data, scale invariance). Categories are split into classes;
cross-cutting categories use markers.
"""

import math

import polars as pl
from hypothesis import given
from hypothesis import strategies as st
from tests.metrics.oracles import pain_index_reference
from tests.support import (
    ABSOLUTE_TOLERANCE_REFERENCE,
    COLUMN_X,
    RELATIVE_TOLERANCE_PROPERTY,
    RELATIVE_TOLERANCE_REFERENCE,
    RELATIVE_TOLERANCE_SCALE,
    apply_expr,
    assert_matches,
)

from pomata.metrics import pain_index

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- pain_index is windowless and REDUCING (M = 0); a case is just a positive equity series. Facts:
#   1. shape   reducing: the output is one scalar (one row in ``select``; one per group under ``.over``)
#   2. domain  positive equities (a growth factor is > 0); the missing variant mixes null / NaN
#   3. scale   invariant under a positive rescale (the peak ratio cancels) -> scale-invariance tier
# Repetitions N are the shared CI profile (tests/conftest.py).
# ----------------------------------------------------------------------------------------------------------------------
SERIES_MAX = 50
_EQUITY = st.floats(min_value=1e-2, max_value=1e4, allow_nan=False, allow_infinity=False)
_EQUITY_MISSING = st.one_of(st.none(), st.just(math.nan), _EQUITY)


@st.composite
def _cases[T](draw: st.DrawFn, equities: st.SearchStrategy[T], min_size: int = 1) -> list[T]:
    """A positive equity series sized from the facts above."""
    return draw(st.lists(equities, min_size=min_size, max_size=SERIES_MAX))


class TestPainIndexContract:
    """
    Type, shape, and lazy/eager guarantees.
    """


class TestPainIndexEdge:
    """
    Boundaries and null / NaN handling.
    """

    def test_single_row_is_zero(self) -> None:
        """
        Verifies that a one-element series is at its own peak, so the pain index is ``0``.
        """
        assert_matches(apply_expr([1.0], pain_index(pl.col(COLUMN_X))), [0.0])

    def test_no_drawdown_is_zero(self) -> None:
        """
        Verifies that a monotonically rising curve is never below its peak, so the pain index is ``0``.
        """
        assert_matches(apply_expr([1.0, 1.1, 1.21], pain_index(pl.col(COLUMN_X))), [0.0])

    def test_nan_poisons(self) -> None:
        """
        Verifies that a NaN equity poisons the result to NaN.
        """
        assert_matches(apply_expr([1.1, math.nan, 1.2], pain_index(pl.col(COLUMN_X))), [math.nan])

    def test_null_skipped(self) -> None:
        """
        Verifies that null equities are skipped, matching the reference.
        """
        values = [1.0, None, 0.9, 0.95, None, 1.1, 1.05]
        assert_matches(
            apply_expr(values, pain_index(pl.col(COLUMN_X))),
            [pain_index_reference(values)],
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
        )


class TestPainIndexCorrectness:
    """
    Against the naive reference oracle and frozen golden-master values.
    """

    def test_matches_reference(self) -> None:
        """
        Verifies agreement with the naive closed-form reference over a representative equity curve.
        """
        values = [1.0, 1.1, 1.05, 1.2, 1.15, 1.3, 1.25]
        assert_matches(
            apply_expr(values, pain_index(pl.col(COLUMN_X))),
            [pain_index_reference(values)],
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference: the mean absolute drawdown of the curve is 0.0179.
        """
        values = [1.1, 1.05, 1.2, 1.15, 1.3, 1.25, 1.4]
        assert_matches(apply_expr(values, pain_index(pl.col(COLUMN_X)).round(4)), [0.0179])


class TestPainIndexProperties:
    """
    Invariants that must hold for all inputs (property-based).
    """

    @given(case=_cases(_EQUITY))
    def test_matches_reference_for_any_input(self, case: list[float]) -> None:
        """
        Verifies that, for any positive equity series, the implementation matches the naive reference.
        """
        assert_matches(
            apply_expr(case, pain_index(pl.col(COLUMN_X))),
            [pain_index_reference(case)],
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    @given(case=_cases(_EQUITY_MISSING, min_size=0))
    def test_matches_reference_under_missing_data(self, case: list[float | None]) -> None:
        """
        Verifies that, for inputs freely mixing null / NaN / finite, the implementation matches the naive reference.
        """
        assert_matches(
            apply_expr(case, pain_index(pl.col(COLUMN_X))),
            [pain_index_reference(case)],
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    @given(case=_cases(_EQUITY), exponent=st.sampled_from([-4, -3, -2, -1, 1, 2, 3, 4]))
    def test_scale_invariance(self, case: list[float], exponent: int) -> None:
        """
        Verifies that ``pain_index`` is scale-invariant: scaling every input value by a constant ``k`` leaves the
        output unchanged -- ``pain_index(k * x) == pain_index(x)``. ``k`` is a power of two, so the rescale is exact
        and adds no floating-point error.
        """
        k = 2.0**exponent
        base = apply_expr(case, pain_index(pl.col(COLUMN_X)))
        scaled = apply_expr([value * k for value in case], pain_index(pl.col(COLUMN_X)))
        assert_matches(scaled, base, rel_tol=RELATIVE_TOLERANCE_SCALE, abs_tol=ABSOLUTE_TOLERANCE_REFERENCE)
