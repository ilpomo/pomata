"""
Tests for ``pomata.metrics.drawdown`` — the running fractional decline of an equity curve from its prior peak.

``drawdown`` is single-input and series-valued (one drawdown per row), so tests use the shared ``apply_expr`` helper to
materialize the factory over a one-column ``Float64`` frame; ``assert_matches`` and the naive ``drawdown_reference``
oracle are shared across the suite. It is invariant under a positive rescaling of the equity (the peak ratio cancels),
so it carries a scale-invariance tier rather than scale-homogeneity / large-magnitude.

The ladder is the canonical one: contract (type / shape / lazy-eager / ``.over`` per-group independence), edge (warm-up
null / single-row / null / NaN / monotonic), correctness (vs the closed-form reference and a frozen golden master), and
properties (reference agreement incl. missing data, scale invariance). Categories are split into classes; cross-cutting
categories use markers (see ``tests/README.md``).
"""

import math

import polars as pl
from hypothesis import given
from hypothesis import strategies as st
from polars.testing import assert_frame_equal
from tests.metrics.oracles import drawdown_reference
from tests.support import (
    COLUMN_X,
    GROUP_KEY,
    RELATIVE_TOLERANCE_PROPERTY,
    RELATIVE_TOLERANCE_REFERENCE,
    apply_expr,
    assert_matches,
)

from pomata.metrics import drawdown

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- derived, not chosen; the rationale is the shared method in CORRECTNESS.md, the numbers are this
# metric's. drawdown is windowless and SERIES-valued (M = 0); a case is just a positive equity series. Its facts:
#   1. shape   series: one drawdown per row, the same length as the input
#   2. domain  positive equities (a growth factor is > 0); the missing variant mixes null / NaN
#   3. scale   invariant under a positive rescale (the peak ratio cancels) -> scale-invariance tier
# Repetitions N are the shared CI profile (tests/conftest.py).
# ----------------------------------------------------------------------------------------------------------------------
SERIES_MAX = 50
_EQUITY = st.floats(min_value=1e-2, max_value=1e4, allow_nan=False, allow_infinity=False)
_EQUITY_MISSING = st.one_of(st.none(), st.just(math.nan), _EQUITY)


@st.composite
def _cases[T](draw: st.DrawFn, equities: st.SearchStrategy[T], min_size: int = 1) -> list[T]:
    """A positive equity series sized from the facts above. drawdown is windowless, so a case is just the series."""
    return draw(st.lists(equities, min_size=min_size, max_size=SERIES_MAX))


class TestDrawdownContract:
    """
    Type, shape, and lazy/eager guarantees.
    """

    def test_returns_expr(self) -> None:
        """
        Verifies that the factory returns a ``pl.Expr`` without touching a frame.
        """
        assert isinstance(drawdown(pl.col(COLUMN_X)), pl.Expr)

    def test_preserves_length_and_dtype(self) -> None:
        """
        Verifies that the output has one value per input row and is ``Float64``.
        """
        frame = pl.DataFrame({COLUMN_X: pl.Series(COLUMN_X, [1.0, 1.1, 1.05, 1.2], dtype=pl.Float64)})
        result = frame.select(drawdown(pl.col(COLUMN_X)).alias("d"))
        assert result.height == frame.height
        assert result.schema["d"] == pl.Float64

    def test_lazy_eager_parity(self) -> None:
        """
        Verifies that eager and lazy application produce identical materialized output.
        """
        frame = pl.DataFrame({COLUMN_X: pl.Series(COLUMN_X, [1.0, 1.1, 1.05, 1.2], dtype=pl.Float64)})
        expr = drawdown(pl.col(COLUMN_X)).alias("d")
        assert_frame_equal(frame.select(expr), frame.lazy().select(expr).collect())

    def test_over_partitions_independently(self) -> None:
        """
        Verifies that under ``.over`` the running peak restarts per group and never carries across boundaries.
        """
        group_a = [1.0, 1.1, 1.05, 1.2]
        group_b = [2.0, 1.8, 2.2, 2.0]
        frame = pl.DataFrame({GROUP_KEY: ["a"] * 4 + ["b"] * 4, COLUMN_X: group_a + group_b})
        grouped = frame.select(drawdown(pl.col(COLUMN_X)).over(GROUP_KEY).alias("d"))["d"].to_list()
        assert_matches(grouped, drawdown_reference(group_a) + drawdown_reference(group_b))


class TestDrawdownEdge:
    """
    Boundaries, warm-up, and null / NaN handling.
    """

    def test_empty(self) -> None:
        """
        Verifies that an empty series yields an empty result.
        """
        assert_matches(apply_expr([], drawdown(pl.col(COLUMN_X))), [])

    def test_single_row(self) -> None:
        """
        Verifies that a one-element series is at its own peak, so the drawdown is ``0``.
        """
        assert_matches(apply_expr([1.0], drawdown(pl.col(COLUMN_X))), [0.0])

    def test_all_null(self) -> None:
        """
        Verifies that an all-null series stays null.
        """
        assert_matches(apply_expr([None, None, None], drawdown(pl.col(COLUMN_X))), [None, None, None])

    def test_leading_null(self) -> None:
        """
        Verifies that a leading warm-up null stays null and the curve begins at the first defined equity.
        """
        values = [None, 1.0, 1.1, 0.99, 1.2]
        assert_matches(apply_expr(values, drawdown(pl.col(COLUMN_X))), drawdown_reference(values))

    def test_interior_null_carries_peak(self) -> None:
        """
        Verifies that an interior null yields null at that row while the running peak carries across it (matching the
        naive reference).
        """
        values = [1.0, 1.2, None, 1.1, 1.3]
        assert_matches(apply_expr(values, drawdown(pl.col(COLUMN_X))), drawdown_reference(values))

    def test_nan_row(self) -> None:
        """
        Verifies that a NaN equity yields NaN at that row while the running peak ignores it (matching the reference).
        """
        values = [1.0, 1.1, math.nan, 0.9, 1.2]
        assert_matches(apply_expr(values, drawdown(pl.col(COLUMN_X))), drawdown_reference(values))


class TestDrawdownCorrectness:
    """
    Against the naive reference oracle and frozen golden-master values.
    """

    def test_matches_reference(self) -> None:
        """
        Verifies agreement with the naive closed-form reference over a representative equity curve.
        """
        values = [1.0, 1.05, 1.2, 1.1, 1.3, 0.95, 1.0, 1.4]
        assert_matches(
            apply_expr(values, drawdown(pl.col(COLUMN_X))),
            drawdown_reference(values),
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
        )

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference over a six-bar equity curve with a peak, a dip, and a deeper trough.
        """
        result = apply_expr([1.0, 1.1, 1.05, 1.2, 0.9, 1.0], drawdown(pl.col(COLUMN_X)).round(4))
        assert_matches(result, [0.0, 0.0, -0.0455, 0.0, -0.25, -0.1667])


class TestDrawdownProperties:
    """
    Invariants that must hold for all inputs (property-based).
    """

    @given(case=_cases(_EQUITY))
    def test_matches_reference_for_any_input(self, case: list[float]) -> None:
        """
        Verifies that, for any positive equity series, the implementation matches the naive reference.
        """
        assert_matches(
            apply_expr(case, drawdown(pl.col(COLUMN_X))),
            drawdown_reference(case),
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
        )

    @given(case=_cases(_EQUITY_MISSING, min_size=0))
    def test_matches_reference_under_missing_data(self, case: list[float | None]) -> None:
        """
        Verifies that, for inputs freely mixing null / NaN / finite, the implementation matches the naive reference.
        """
        assert_matches(
            apply_expr(case, drawdown(pl.col(COLUMN_X))),
            drawdown_reference(case),
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
        )

    @given(case=_cases(_EQUITY), exponent=st.sampled_from([-4, -3, -2, -1, 1, 2, 3, 4]))
    def test_scale_invariance(self, case: list[float], exponent: int) -> None:
        """
        Verifies that a positive rescale of the equity leaves the drawdown unchanged (the peak ratio cancels), using
        powers of two so the rescaling is lossless.
        """
        k = 2.0**exponent
        base = apply_expr(case, drawdown(pl.col(COLUMN_X)))
        scaled = apply_expr([value * k for value in case], drawdown(pl.col(COLUMN_X)))
        assert_matches(scaled, base, rel_tol=RELATIVE_TOLERANCE_PROPERTY)
