"""
Tests for ``pomata.metrics.total_return`` — the overall compounded return of an equity curve.

``total_return`` is single-input and REDUCING (an equity series → one scalar), so tests read the single output row of
``apply_expr``; ``assert_matches`` and the naive ``total_return_reference`` oracle are shared across the suite. It reads
a growth-factor series normalized to a unit start (so the result is the final value minus one) rather than a raw price
series, and is therefore neither scale-homogeneous nor scale-invariant — its correctness is pinned by the reference and
the golden master, not a scale tier.

The ladder is the canonical one: contract (type / reduces-to-scalar / lazy-eager / ``.over`` per-group independence),
edge (empty / single-row / null / leading-null / NaN poison), correctness (vs the closed-form reference and a frozen
golden master), and properties (reference agreement incl. missing data). Categories are split into classes;
cross-cutting categories use markers (see ``tests/README.md``).
"""

import math

import polars as pl
from hypothesis import given
from hypothesis import strategies as st
from tests.metrics.oracles import total_return_reference
from tests.support import (
    ABSOLUTE_TOLERANCE_REFERENCE,
    COLUMN_X,
    RELATIVE_TOLERANCE_PROPERTY,
    RELATIVE_TOLERANCE_REFERENCE,
    apply_expr,
    assert_matches,
)

from pomata.metrics import total_return

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- total_return is windowless and REDUCING (M = 0); a case is just a positive equity series. Facts:
#   1. shape   reducing: the output is one scalar (one row in ``select``; one per group under ``.over``)
#   2. domain  positive growth factors (a unit-start equity curve); the missing variant mixes null / NaN
#   3. scale   neither (a normalized growth factor) -> no scale tier; correctness pinned by the reference + golden
# Repetitions N are the shared CI profile (tests/conftest.py).
# ----------------------------------------------------------------------------------------------------------------------
SERIES_MAX = 50
_EQUITY = st.floats(min_value=1e-2, max_value=1e4, allow_nan=False, allow_infinity=False)
_EQUITY_MISSING = st.one_of(st.none(), st.just(math.nan), _EQUITY)


@st.composite
def _cases[T](draw: st.DrawFn, equities: st.SearchStrategy[T], min_size: int = 1) -> list[T]:
    """A positive equity series sized from the facts above."""
    return draw(st.lists(equities, min_size=min_size, max_size=SERIES_MAX))


class TestTotalReturnContract:
    """
    Type, shape, and lazy/eager guarantees.
    """


class TestTotalReturnEdge:
    """
    Boundaries and null / NaN handling.
    """

    def test_single_row(self) -> None:
        """
        Verifies that a one-element series resolves to the final growth minus one.
        """
        assert_matches(apply_expr([1.21], total_return(pl.col(COLUMN_X)).round(4)), [0.21])

    def test_leading_null_uses_last_defined(self) -> None:
        """
        Verifies that leading warm-up nulls are skipped; the result uses the last defined equity.
        """
        values = [None, 1.1, 1.21]
        assert_matches(apply_expr(values, total_return(pl.col(COLUMN_X))), [total_return_reference(values)])

    def test_nan_poisons(self) -> None:
        """
        Verifies that a NaN equity poisons the result to NaN.
        """
        assert_matches(apply_expr([1.1, math.nan, 1.2], total_return(pl.col(COLUMN_X))), [math.nan])


class TestTotalReturnCorrectness:
    """
    Against the naive reference oracle and frozen golden-master values.
    """

    def test_matches_reference(self) -> None:
        """
        Verifies agreement with the naive closed-form reference over a representative equity curve.
        """
        values = [1.02, 1.0, 1.08, 1.05, 1.15, 1.1, 1.2]
        assert_matches(
            apply_expr(values, total_return(pl.col(COLUMN_X))),
            [total_return_reference(values)],
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference: the equity reaches 1.3794, a 37.94% total return.
        """
        result = apply_expr([1.1, 1.045, 1.254, 1.3794], total_return(pl.col(COLUMN_X)).round(4))
        assert_matches(result, [0.3794])


class TestTotalReturnProperties:
    """
    Invariants that must hold for all inputs (property-based).
    """

    @given(case=_cases(_EQUITY))
    def test_matches_reference_for_any_input(self, case: list[float]) -> None:
        """
        Verifies that, for any positive equity series, the implementation matches the naive reference.
        """
        assert_matches(
            apply_expr(case, total_return(pl.col(COLUMN_X))),
            [total_return_reference(case)],
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    @given(case=_cases(_EQUITY_MISSING, min_size=0))
    def test_matches_reference_under_missing_data(self, case: list[float | None]) -> None:
        """
        Verifies that, for inputs freely mixing null / NaN / finite, the implementation matches the naive reference.
        """
        assert_matches(
            apply_expr(case, total_return(pl.col(COLUMN_X))),
            [total_return_reference(case)],
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )
