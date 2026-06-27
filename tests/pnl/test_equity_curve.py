"""
Tests for ``pomata.pnl.equity_curve`` — the compounded growth of one unit of capital.

``equity_curve`` is single-input and cumulative (a running product of one-plus-returns), so tests use the shared
``apply_expr`` helper to materialize the factory over a one-column ``Float64`` frame; ``assert_matches`` and the naive
``equity_curve_reference`` oracle are shared across the suite. The curve is a nonlinear transform of the returns
(neither scale-invariant nor homogeneous), so it carries a compounding metamorphic in place of the scale-homogeneity /
large-magnitude tiers.

The ladder is the canonical one: contract (type / shape / lazy-eager / ``.over`` per-group independence), edge
(warm-up / single-row / null / NaN / interior-null continuity), correctness (vs the closed-form reference and a frozen
golden master), and properties (reference agreement incl. missing data, the compounding identity). Categories are split
into classes; cross-cutting categories use markers (see ``tests/README.md``).
"""

import math

import polars as pl
from hypothesis import given
from hypothesis import strategies as st
from polars.testing import assert_frame_equal
from tests.pnl.oracles import equity_curve_reference
from tests.support import (
    ABSOLUTE_TOLERANCE_REFERENCE,
    COLUMN_X,
    GROUP_KEY,
    RELATIVE_TOLERANCE_PROPERTY,
    RELATIVE_TOLERANCE_REFERENCE,
    apply_expr,
    assert_matches,
)

from pomata.pnl import equity_curve

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- derived, not chosen; the rationale is the shared method in CORRECTNESS.md, the numbers are this
# primitive's. equity_curve is a windowless cumulative product (M = 0); a case is just the return series. Returns are
# drawn from a modest range so one-plus-return stays positive and the product never overflows; the curve is nonlinear,
# so there is no scale-homogeneity or large-magnitude VALUE test -- a compounding metamorphic stands in their place.
# Repetitions N are the shared CI profile (tests/conftest.py); override per-test only if its parameter space is larger.
# ----------------------------------------------------------------------------------------------------------------------
SERIES_MAX = 50

# Modest per-bar returns: one-plus-return stays in [0.1, 1.9], so the cumulative product stays finite and positive.
_RETURNS = st.floats(min_value=-0.9, max_value=0.9, allow_nan=False, allow_infinity=False)
_RETURNS_MISSING = st.one_of(st.none(), st.just(math.nan), _RETURNS)


@st.composite
def _cases[T](draw: st.DrawFn, returns: st.SearchStrategy[T], min_size: int = 1) -> list[T]:
    """
    A return series sized from the facts above. equity_curve is windowless, so a case is just the series.
    """
    return draw(st.lists(returns, min_size=min_size, max_size=SERIES_MAX))


class TestEquityCurveContract:
    """
    Type, shape, and lazy/eager guarantees.
    """

    def test_returns_expr(self) -> None:
        """
        Verifies that the factory returns a ``pl.Expr`` without touching a frame.
        """
        assert isinstance(equity_curve(pl.col(COLUMN_X)), pl.Expr)

    def test_preserves_length_and_dtype(self) -> None:
        """
        Verifies that the output has one value per input row and is ``Float64``.
        """
        frame = pl.DataFrame({COLUMN_X: pl.Series(COLUMN_X, [0.1, -0.05, 0.2, 0.1], dtype=pl.Float64)})
        result = frame.select(equity_curve(pl.col(COLUMN_X)).alias("y"))
        assert result.height == frame.height
        assert result.schema["y"] == pl.Float64

    def test_lazy_eager_parity(self) -> None:
        """
        Verifies that eager and lazy application produce identical materialized output.
        """
        frame = pl.DataFrame({COLUMN_X: pl.Series(COLUMN_X, [0.1, -0.05, 0.2, 0.1], dtype=pl.Float64)})
        expr = equity_curve(pl.col(COLUMN_X)).alias("y")
        result_eager = frame.select(expr)
        result_lazy = frame.lazy().select(expr).collect()
        assert_frame_equal(result_eager, result_lazy)

    def test_over_partitions_independently(self) -> None:
        """
        Verifies that under ``.over`` the running product restarts per group and never carries across boundaries.
        """
        frame = pl.DataFrame({GROUP_KEY: ["a"] * 3 + ["b"] * 3, COLUMN_X: [0.1, 0.2, -0.05, 0.0, 0.1, 0.1]})
        expr = equity_curve(pl.col(COLUMN_X)).over(GROUP_KEY)
        grouped = frame.select(expr.alias("y"))["y"].to_list()
        group_a = apply_expr([0.1, 0.2, -0.05], equity_curve(pl.col(COLUMN_X)))
        group_b = apply_expr([0.0, 0.1, 0.1], equity_curve(pl.col(COLUMN_X)))
        assert_matches(grouped, group_a + group_b)


class TestEquityCurveEdge:
    """
    Boundaries, warm-up, and null / NaN handling.
    """

    def test_warmup_leading_null(self) -> None:
        """
        Verifies a leading warm-up null (as produced by returns_simple) stays null and the curve begins at the first
        defined return.
        """
        result = apply_expr([None, 0.1, 0.2, -0.05], equity_curve(pl.col(COLUMN_X)))
        assert result[0] is None
        assert result[1] is not None

    def test_single_row(self) -> None:
        """
        Verifies that a one-element series resolves to ``1 + return`` (no warm-up of its own).
        """
        assert_matches(apply_expr([0.1], equity_curve(pl.col(COLUMN_X)).round(4)), [1.1])

    def test_empty(self) -> None:
        """
        Verifies that an empty series yields an empty result.
        """
        assert_matches(apply_expr([], equity_curve(pl.col(COLUMN_X))), [])

    def test_all_null(self) -> None:
        """
        Verifies that an all-null series stays null.
        """
        assert_matches(apply_expr([None, None, None], equity_curve(pl.col(COLUMN_X))), [None, None, None])

    def test_null_carries_across(self) -> None:
        """
        Verifies that an interior null emits null at that row while the product carries across it unchanged (matching
        the naive reference).
        """
        values = [0.1, None, 0.2, -0.05]
        assert_matches(apply_expr(values, equity_curve(pl.col(COLUMN_X))), equity_curve_reference(values))

    def test_nan_propagates(self) -> None:
        """
        Verifies that a NaN enters the running product and every later row is NaN (matching the naive reference).
        """
        values = [0.1, math.nan, 0.2, -0.05]
        assert_matches(apply_expr(values, equity_curve(pl.col(COLUMN_X))), equity_curve_reference(values))


class TestEquityCurveCorrectness:
    """
    Against the naive reference oracle and frozen golden-master values.
    """

    def test_matches_reference(self) -> None:
        """
        Verifies agreement with the naive closed-form reference over a representative return series.
        """
        values = [0.1, -0.05, 0.2, 0.1, -0.02, 0.03, -0.1, 0.04]
        assert_matches(
            apply_expr(values, equity_curve(pl.col(COLUMN_X))),
            equity_curve_reference(values),
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference over a four-bar return series.
        """
        result = apply_expr([0.1, -0.05, 0.2, 0.1], equity_curve(pl.col(COLUMN_X)).round(4))
        assert_matches(result, [1.1, 1.045, 1.254, 1.3794])


class TestEquityCurveProperties:
    """
    Invariants that must hold for all inputs (property-based).
    """

    @given(case=_cases(_RETURNS, min_size=0))
    def test_matches_reference_for_any_input(
        self,
        case: list[float],
    ) -> None:
        """
        Verifies that, for any modest return series, the implementation matches the naive reference.
        """
        values = case
        assert_matches(
            apply_expr(values, equity_curve(pl.col(COLUMN_X))),
            equity_curve_reference(values),
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    @given(case=_cases(_RETURNS_MISSING, min_size=0))
    def test_matches_reference_under_missing_data(
        self,
        case: list[float | None],
    ) -> None:
        """
        Verifies that, for inputs freely mixing null / NaN / finite, the implementation matches the naive reference.
        """
        values = case
        assert_matches(
            apply_expr(values, equity_curve(pl.col(COLUMN_X))),
            equity_curve_reference(values),
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    @given(case=_cases(_RETURNS))
    def test_compounds_consecutive_returns(
        self,
        case: list[float],
    ) -> None:
        """
        Verifies the defining identity of a compounded curve: the first row is ``1 + return`` and each later row over
        its predecessor recovers ``1 + return`` (the running product compounds each bar's return).
        """
        values = case
        equity = apply_expr(values, equity_curve(pl.col(COLUMN_X)))
        assert equity[0] is not None
        assert math.isclose(
            equity[0], 1.0 + values[0], rel_tol=RELATIVE_TOLERANCE_PROPERTY, abs_tol=ABSOLUTE_TOLERANCE_REFERENCE
        )
        for index in range(1, len(values)):
            current = equity[index]
            previous = equity[index - 1]
            assert current is not None
            assert previous is not None
            assert math.isclose(
                current / previous,
                1.0 + values[index],
                rel_tol=RELATIVE_TOLERANCE_PROPERTY,
                abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
            )
