"""
Tests for ``pomata.metrics.max_drawdown_duration`` — the longest underwater stretch of an equity curve, in bars.

``max_drawdown_duration`` is single-input and REDUCING (an equity series → one scalar), so tests read the single output
row of ``apply_expr``; ``assert_matches`` and the naive ``max_drawdown_duration_reference`` oracle (the longest run of
strictly negative drawdown) are shared across the suite. It is a bar count returned as ``Float64``, scale-invariant
under a positive rescale.

The ladder is the canonical one: contract (type / reduces-to-scalar / lazy-eager / ``.over`` per-group independence),
edge (empty / single-row / no-drawdown / null / NaN), correctness (vs the closed-form reference and a frozen golden
master), and properties (reference agreement incl. missing data, scale invariance). Categories are split into classes;
cross-cutting categories use markers.
"""

import math

import polars as pl
from hypothesis import given
from hypothesis import strategies as st
from tests.metrics.oracles import max_drawdown_duration_reference
from tests.support import (
    COLUMN_X,
    RELATIVE_TOLERANCE_REFERENCE,
    apply_expr,
    assert_matches,
)

from pomata.metrics import max_drawdown_duration

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- max_drawdown_duration is windowless and REDUCING (M = 0); a case is a positive equity series. Facts:
#   1. shape   reducing: the output is one scalar (a Float64 bar count; one per group under ``.over``)
#   2. domain  positive equities (a growth factor is > 0); the missing variant mixes null / NaN
#   3. scale   invariant under a positive rescale (the underwater pattern is unchanged) -> scale-invariance tier
# Repetitions N are the shared CI profile (tests/conftest.py).
# ----------------------------------------------------------------------------------------------------------------------
SERIES_MAX = 50
_EQUITY = st.floats(min_value=1e-2, max_value=1e4, allow_nan=False, allow_infinity=False)
_EQUITY_MISSING = st.one_of(st.none(), st.just(math.nan), _EQUITY)


@st.composite
def _cases[T](draw: st.DrawFn, equities: st.SearchStrategy[T], min_size: int = 1) -> list[T]:
    """A positive equity series sized from the facts above."""
    return draw(st.lists(equities, min_size=min_size, max_size=SERIES_MAX))


class TestMaxDrawdownDurationContract:
    """
    Type, shape, and lazy/eager guarantees.
    """


class TestMaxDrawdownDurationEdge:
    """
    Boundaries and null / NaN handling.
    """

    def test_single_row_is_zero(self) -> None:
        """
        Verifies that a one-element series is never underwater, so the duration is ``0``.
        """
        assert_matches(apply_expr([1.0], max_drawdown_duration(pl.col(COLUMN_X))), [0.0])

    def test_no_drawdown_is_zero(self) -> None:
        """
        Verifies that a monotonically rising curve is never underwater, so the duration is ``0``.
        """
        assert_matches(apply_expr([1.0, 1.1, 1.21], max_drawdown_duration(pl.col(COLUMN_X))), [0.0])

    def test_null_skipped(self) -> None:
        """
        Verifies that null equities are skipped, matching the reference (the run is over the retained observations).
        """
        values = [1.0, None, 0.9, 0.8, None, 1.2, 0.95]
        assert_matches(
            apply_expr(values, max_drawdown_duration(pl.col(COLUMN_X))),
            [max_drawdown_duration_reference(values)],
        )

    def test_nan_poisons(self) -> None:
        """
        Verifies that a NaN equity poisons the result to NaN.
        """
        assert_matches(apply_expr([1.0, math.nan, 0.9], max_drawdown_duration(pl.col(COLUMN_X))), [math.nan])


class TestMaxDrawdownDurationCorrectness:
    """
    Against the naive reference oracle and frozen golden-master values.
    """

    def test_matches_reference(self) -> None:
        """
        Verifies agreement with the naive closed-form reference over a representative equity curve.
        """
        values = [1.0, 1.1, 1.05, 1.2, 1.15, 1.1, 1.25, 1.3]
        assert_matches(
            apply_expr(values, max_drawdown_duration(pl.col(COLUMN_X))),
            [max_drawdown_duration_reference(values)],
        )

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference: the curve stays below its peak for three consecutive bars, so the duration is 3.
        """
        values = [1.0, 0.9, 0.8, 0.85, 1.1, 1.05]
        assert_matches(apply_expr(values, max_drawdown_duration(pl.col(COLUMN_X))), [3.0])


class TestMaxDrawdownDurationProperties:
    """
    Invariants that must hold for all inputs (property-based).
    """

    @given(case=_cases(_EQUITY))
    def test_matches_reference_for_any_input(self, case: list[float]) -> None:
        """
        Verifies that, for any positive equity series, the implementation matches the naive reference.
        """
        assert_matches(
            apply_expr(case, max_drawdown_duration(pl.col(COLUMN_X))),
            [max_drawdown_duration_reference(case)],
        )

    @given(case=_cases(_EQUITY_MISSING, min_size=0))
    def test_matches_reference_under_missing_data(self, case: list[float | None]) -> None:
        """
        Verifies that, for inputs freely mixing null / NaN / finite, the implementation matches the naive reference.
        """
        assert_matches(
            apply_expr(case, max_drawdown_duration(pl.col(COLUMN_X))),
            [max_drawdown_duration_reference(case)],
        )

    @given(case=_cases(_EQUITY), exponent=st.sampled_from([-4, -3, -2, -1, 1, 2, 3, 4]))
    def test_scale_invariance(self, case: list[float], exponent: int) -> None:
        """
        Verifies that ``max_drawdown_duration`` is scale-invariant: scaling every input value by a constant ``k``
        leaves the output unchanged -- ``max_drawdown_duration(k * x) == max_drawdown_duration(x)``. ``k`` is a
        power of two, so the rescale is exact and adds no floating-point error.
        """
        k = 2.0**exponent
        base = apply_expr(case, max_drawdown_duration(pl.col(COLUMN_X)))
        scaled = apply_expr([value * k for value in case], max_drawdown_duration(pl.col(COLUMN_X)))
        assert_matches(scaled, base, rel_tol=RELATIVE_TOLERANCE_REFERENCE)
