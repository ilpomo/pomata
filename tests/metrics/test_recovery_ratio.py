"""
Tests for ``pomata.metrics.recovery_ratio`` — the total return per unit of maximum drawdown.

``recovery_ratio`` is single-input and REDUCING (an equity series → one scalar), so tests read the single output row of
``apply_expr``; ``assert_matches`` and the naive ``recovery_ratio_reference`` oracle are shared across the suite. It
reads a normalized growth-factor curve, so it is neither scale-homogeneous nor scale-invariant -- its correctness is
pinned by the reference, a golden master, and the metamorphic identity linking it to :func:`total_return` and
:func:`max_drawdown`.

The ladder is the canonical one: contract (type / reduces-to-scalar / lazy-eager / ``.over`` per-group independence),
edge (empty / single-row / no-drawdown / null / NaN), correctness (vs the closed-form reference and a frozen golden
master), and properties (reference agreement incl. missing data, the component-definition identity). Categories are
split into classes; cross-cutting categories use markers.
"""

import math

import polars as pl
from hypothesis import given
from hypothesis import strategies as st
from polars.testing import assert_frame_equal
from tests.metrics.oracles import recovery_ratio_reference
from tests.support import (
    ABSOLUTE_TOLERANCE_REFERENCE,
    COLUMN_X,
    GROUP_KEY,
    RELATIVE_TOLERANCE_PROPERTY,
    RELATIVE_TOLERANCE_REFERENCE,
    apply_expr,
    assert_matches,
)

from pomata.metrics import max_drawdown, recovery_ratio, total_return

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- recovery_ratio is windowless and REDUCING (M = 0); a case is just a positive equity series. Facts:
#   1. shape   reducing: the output is one scalar (one row in ``select``; one per group under ``.over``)
#   2. domain  positive growth factors in [0.1, 10]; the missing variant mixes null / NaN
#   3. scale   neither (a normalized growth factor over a scale-invariant drawdown) -> pinned by reference + identity
# Repetitions N are the shared CI profile (tests/conftest.py).
# ----------------------------------------------------------------------------------------------------------------------
SERIES_MAX = 50
_EQUITY = st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False)
_EQUITY_MISSING = st.one_of(st.none(), st.just(math.nan), _EQUITY)


@st.composite
def _cases[T](draw: st.DrawFn, equities: st.SearchStrategy[T], min_size: int = 1) -> list[T]:
    """A positive equity series sized from the facts above."""
    return draw(st.lists(equities, min_size=min_size, max_size=SERIES_MAX))


class TestRecoveryFactorContract:
    """
    Type, shape, and lazy/eager guarantees.
    """

    def test_returns_expr(self) -> None:
        """
        Verifies that the factory returns a ``pl.Expr`` without touching a frame.
        """
        assert isinstance(recovery_ratio(pl.col(COLUMN_X)), pl.Expr)

    def test_reduces_to_scalar(self) -> None:
        """
        Verifies that the metric reduces a series to one ``Float64`` row.
        """
        frame = pl.DataFrame({COLUMN_X: pl.Series(COLUMN_X, [1.1, 1.05, 1.2, 1.15], dtype=pl.Float64)})
        result = frame.select(recovery_ratio(pl.col(COLUMN_X)).alias("r"))
        assert result.height == 1
        assert result.schema["r"] == pl.Float64

    def test_lazy_eager_parity(self) -> None:
        """
        Verifies that eager and lazy application produce identical materialized output.
        """
        frame = pl.DataFrame({COLUMN_X: pl.Series(COLUMN_X, [1.1, 1.05, 1.2, 1.15], dtype=pl.Float64)})
        expr = recovery_ratio(pl.col(COLUMN_X)).alias("r")
        assert_frame_equal(frame.select(expr), frame.lazy().select(expr).collect())

    def test_over_partitions_independently(self) -> None:
        """
        Verifies that under ``.over`` the ratio is computed per group (broadcast) and never spans boundaries.
        """
        group_a = [1.1, 1.05, 1.2, 1.15, 1.3]
        group_b = [1.0, 0.9, 1.05, 1.1]
        frame = pl.DataFrame({GROUP_KEY: ["a"] * len(group_a) + ["b"] * len(group_b), COLUMN_X: group_a + group_b})
        grouped = frame.select(recovery_ratio(pl.col(COLUMN_X)).over(GROUP_KEY).alias("r"))["r"].to_list()
        expected_a = recovery_ratio_reference(group_a)
        expected_b = recovery_ratio_reference(group_b)
        assert_matches(
            grouped, [expected_a] * len(group_a) + [expected_b] * len(group_b), rel_tol=RELATIVE_TOLERANCE_REFERENCE
        )


class TestRecoveryFactorEdge:
    """
    Boundaries and null / NaN handling.
    """

    def test_empty(self) -> None:
        """
        Verifies that an empty series yields ``null``.
        """
        assert_matches(apply_expr([], recovery_ratio(pl.col(COLUMN_X))), [None])

    def test_single_row_is_nan(self) -> None:
        """
        Verifies that a one-element series has zero growth and zero drawdown, so the ratio is ``0 / 0``, i.e. ``NaN``.
        """
        assert_matches(apply_expr([1.0], recovery_ratio(pl.col(COLUMN_X))), [math.nan])

    def test_no_drawdown_is_inf(self) -> None:
        """
        Verifies that a monotonically rising curve has zero maximum drawdown with positive growth, so the ratio is
        ``+inf``.
        """
        assert_matches(apply_expr([1.0, 1.1, 1.21], recovery_ratio(pl.col(COLUMN_X))), [math.inf])

    def test_losing_curve_is_negative(self) -> None:
        """
        Verifies that a curve ending below its start reports a negative recovery factor: the total-return numerator
        keeps its sign (here ``-0.3``) over the magnitude of the drawdown (``0.3``), giving ``-1.0``.
        """
        assert_matches(apply_expr([1.0, 0.9, 0.95, 0.7], recovery_ratio(pl.col(COLUMN_X))), [-1.0])

    def test_all_null(self) -> None:
        """
        Verifies that an all-null series yields ``null``.
        """
        assert_matches(apply_expr([None, None], recovery_ratio(pl.col(COLUMN_X))), [None])

    def test_nan_poisons(self) -> None:
        """
        Verifies that a NaN equity poisons the result to NaN.
        """
        assert_matches(apply_expr([1.1, math.nan, 1.2], recovery_ratio(pl.col(COLUMN_X))), [math.nan])


class TestRecoveryFactorCorrectness:
    """
    Against the naive reference oracle and frozen golden-master values.
    """

    def test_matches_reference(self) -> None:
        """
        Verifies agreement with the naive closed-form reference over a representative equity curve.
        """
        values = [1.0, 1.1, 1.05, 1.2, 1.15, 1.3, 1.25]
        assert_matches(
            apply_expr(values, recovery_ratio(pl.col(COLUMN_X))),
            [recovery_ratio_reference(values)],
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference: total return (0.4) over the worst drawdown magnitude (~0.0455) is 8.8.
        """
        values = [1.1, 1.05, 1.2, 1.15, 1.3, 1.25, 1.4]
        assert_matches(apply_expr(values, recovery_ratio(pl.col(COLUMN_X)).round(4)), [8.8])


class TestRecoveryFactorProperties:
    """
    Invariants that must hold for all inputs (property-based).
    """

    @given(case=_cases(_EQUITY))
    def test_matches_reference_for_any_input(self, case: list[float]) -> None:
        """
        Verifies that, for any positive equity series, the implementation matches the naive reference.
        """
        assert_matches(
            apply_expr(case, recovery_ratio(pl.col(COLUMN_X))),
            [recovery_ratio_reference(case)],
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    @given(case=_cases(_EQUITY_MISSING, min_size=0))
    def test_matches_reference_under_missing_data(self, case: list[float | None]) -> None:
        """
        Verifies that, for inputs freely mixing null / NaN / finite, the implementation matches the naive reference.
        """
        assert_matches(
            apply_expr(case, recovery_ratio(pl.col(COLUMN_X))),
            [recovery_ratio_reference(case)],
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    @given(case=_cases(_EQUITY))
    def test_matches_component_definition(self, case: list[float]) -> None:
        """
        Verifies the metamorphic identity: ``recovery_ratio`` equals the signed :func:`total_return` divided by the
        magnitude of :func:`max_drawdown`, computed as separate metrics.
        """
        direct = apply_expr(case, recovery_ratio(pl.col(COLUMN_X)))
        composed = apply_expr(case, total_return(pl.col(COLUMN_X)) / max_drawdown(pl.col(COLUMN_X)).abs())
        assert_matches(direct, composed, rel_tol=RELATIVE_TOLERANCE_PROPERTY, abs_tol=ABSOLUTE_TOLERANCE_REFERENCE)
