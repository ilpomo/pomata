"""
Tests for ``pomata.pnl.returns_net`` — Net Strategy Returns.

``returns_net`` is two-input (``returns_gross`` / ``cost``) and elementwise (a pure per-row subtraction, no window, no
lag). Tests use a local ``apply_returns_net`` helper to materialize it over a two-column ``Float64`` frame;
``assert_matches`` and the naive ``returns_net_reference`` oracle are shared across the suite. The difference is
degree-1 homogeneous when both inputs are scaled together, so it carries the scale-homogeneity and large-magnitude
tiers.

The ladder, adapted to an elementwise two-input difference: contract (type / shape / lazy-eager / ``.over`` identity),
edge (empty / single-row / null / NaN / null-precedence), correctness (closed-form reference + frozen golden master),
and properties (reference agreement incl. missing data, scale-homogeneity, large-magnitude). Categories are split into
classes; cross-cutting categories use markers (see ``tests/README.md``).
"""

import math
from collections.abc import Sequence

import polars as pl
from hypothesis import given
from hypothesis import strategies as st
from polars.testing import assert_frame_equal
from tests.pnl.oracles import returns_net_reference
from tests.support import (
    ABSOLUTE_TOLERANCE_REFERENCE,
    ABSOLUTE_TOLERANCE_STREAMING,
    RELATIVE_TOLERANCE_PROPERTY,
    RELATIVE_TOLERANCE_REFERENCE,
    RELATIVE_TOLERANCE_SCALE,
    assert_matches,
    assert_scale_homogeneous,
    finite_floats,
    materialize,
    missing_data_floats,
)

from pomata.pnl import returns_net

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- derived, not chosen; the rationale is the shared method in CORRECTNESS.md, the numbers are this
# primitive's. returns_net is a windowless elementwise difference of two aligned series (W = 0, M = 0): a case is a pair
# of equal-length series, every row a defined output. It is degree-1 homogeneous when both inputs are scaled together,
# so it keeps the scale-homogeneity and large-magnitude tiers. Repetitions N are the shared CI profile
# (tests/conftest.py); override per-test only if its parameter space is larger.
# ----------------------------------------------------------------------------------------------------------------------
SERIES_MAX = 50
RETURNS_GROSS = "returns_gross"
COST = "cost"


@st.composite
def _cases[T](
    draw: st.DrawFn,
    gross: st.SearchStrategy[T],
    cost: st.SearchStrategy[T],
    min_size: int = 1,
) -> tuple[list[T], list[T]]:
    """
    A pair of equal-length series (returns_gross, cost) sized from the facts above. returns_net is windowless, so a case
    is just the aligned pair: every row is a defined output.
    """
    length = draw(st.integers(min_value=min_size, max_value=SERIES_MAX))
    gross_values = draw(st.lists(gross, min_size=length, max_size=length))
    cost_values = draw(st.lists(cost, min_size=length, max_size=length))
    return gross_values, cost_values


def apply_returns_net(
    returns_gross: Sequence[float | None],
    cost: Sequence[float | None],
) -> list[float | None]:
    """
    Materialize ``returns_net`` over a two-column ``Float64`` frame from the aligned input lists.
    """
    return materialize(
        {RETURNS_GROSS: returns_gross, COST: cost},
        returns_net(pl.col(RETURNS_GROSS), pl.col(COST)),
    )


class TestReturnsNetContract:
    """
    Type, shape, and lazy/eager guarantees.
    """

    def test_returns_expr(self) -> None:
        """
        Verifies that the factory returns a ``pl.Expr`` without touching a frame.
        """
        assert isinstance(returns_net(pl.col(RETURNS_GROSS), pl.col(COST)), pl.Expr)

    def test_preserves_length_and_dtype(self) -> None:
        """
        Verifies that the output has one value per input row and is ``Float64``.
        """
        frame = pl.DataFrame(
            {
                RETURNS_GROSS: pl.Series(RETURNS_GROSS, [0.05, -0.02, 0.03, 0.01, 0.0], dtype=pl.Float64),
                COST: pl.Series(COST, [0.0005, 0.0015, 0.0005, 0.0, 0.0005], dtype=pl.Float64),
            }
        )
        result = frame.select(returns_net(pl.col(RETURNS_GROSS), pl.col(COST)).alias("y"))
        assert result.height == frame.height
        assert result.schema["y"] == pl.Float64

    def test_lazy_eager_parity(self) -> None:
        """
        Verifies that eager and lazy application produce identical materialized output.
        """
        frame = pl.DataFrame(
            {
                RETURNS_GROSS: pl.Series(RETURNS_GROSS, [0.05, -0.02, 0.03, 0.01, 0.0], dtype=pl.Float64),
                COST: pl.Series(COST, [0.0005, 0.0015, 0.0005, 0.0, 0.0005], dtype=pl.Float64),
            }
        )
        expr = returns_net(pl.col(RETURNS_GROSS), pl.col(COST)).alias("y")
        result_eager = frame.select(expr)
        result_lazy = frame.lazy().select(expr).collect()
        assert_frame_equal(result_eager, result_lazy)

    def test_over_is_identity(self) -> None:
        """
        Verifies that ``.over`` is optional for this elementwise difference: partitioning by group is identical to the
        un-partitioned call (no cross-bar state can leak across group boundaries).
        """
        frame = pl.DataFrame(
            {
                "ticker": ["a", "a", "a", "b", "b", "b"],
                RETURNS_GROSS: pl.Series(RETURNS_GROSS, [0.05, -0.02, 0.03, 0.01, 0.04, -0.01], dtype=pl.Float64),
                COST: pl.Series(COST, [0.0005, 0.0015, 0.0005, 0.0, 0.001, 0.0005], dtype=pl.Float64),
            }
        )
        plain = frame.select(returns_net(pl.col(RETURNS_GROSS), pl.col(COST)).alias("y"))["y"].to_list()
        grouped = frame.select(returns_net(pl.col(RETURNS_GROSS), pl.col(COST)).over("ticker").alias("y"))[
            "y"
        ].to_list()
        assert_matches(plain, grouped)


class TestReturnsNetEdge:
    """
    Boundaries and null / NaN handling.
    """

    def test_empty(self) -> None:
        """
        Verifies that an empty input yields an empty output.
        """
        assert_matches(apply_returns_net([], []), [])

    def test_single_row(self) -> None:
        """
        Verifies that a one-row series resolves to the single difference (no window, no warm-up).
        """
        assert_matches(apply_returns_net([0.05], [0.0005]), [0.0495])

    def test_all_null(self) -> None:
        """
        Verifies that an all-null input yields an all-null output (the difference propagates ``null``).
        """
        assert_matches(apply_returns_net([None, None], [None, None]), [None, None])

    def test_null_propagates(self) -> None:
        """
        Verifies that a ``null`` in either input makes that row ``null`` (matching the naive reference).
        """
        returns_gross = [0.05, None, 0.03, 0.01]
        cost = [0.0005, 0.0015, 0.0005, 0.0]
        assert_matches(apply_returns_net(returns_gross, cost), returns_net_reference(returns_gross, cost))

    def test_nan_propagates(self) -> None:
        """
        Verifies that a ``NaN`` in either input makes that row ``NaN`` (matching the naive reference).
        """
        returns_gross = [0.05, 0.02, math.nan, 0.01]
        cost = [0.0005, 0.0015, 0.0005, 0.0]
        assert_matches(apply_returns_net(returns_gross, cost), returns_net_reference(returns_gross, cost))

    def test_null_takes_precedence_over_nan(self) -> None:
        """
        Verifies that a row with a ``null`` in one input and a ``NaN`` in the other yields ``null`` — ``null`` takes
        precedence over ``NaN``.
        """
        assert_matches(apply_returns_net([None, 0.05], [math.nan, 0.0005]), [None, 0.0495])


class TestReturnsNetCorrectness:
    """
    Against the naive reference oracle and frozen golden-master values.
    """

    def test_matches_reference(self) -> None:
        """
        Verifies agreement with the naive closed-form reference over a representative series.
        """
        returns_gross = [0.05, -0.02, 0.03, 0.01, 0.0, 0.04, -0.03, 0.02]
        cost = [0.0005, 0.0015, 0.0005, 0.0, 0.0005, 0.001, 0.0015, 0.0005]
        assert_matches(
            apply_returns_net(returns_gross, cost),
            returns_net_reference(returns_gross, cost),
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference over a five-bar series.
        """
        result = materialize(
            {
                RETURNS_GROSS: [0.05, -0.02, 0.03, 0.01, 0.0],
                COST: [0.0005, 0.0015, 0.0005, 0.0, 0.0005],
            },
            returns_net(pl.col(RETURNS_GROSS), pl.col(COST)).round(4),
        )
        assert_matches(result, [0.0495, -0.0215, 0.0295, 0.01, -0.0005])


class TestReturnsNetProperties:
    """
    Invariants that must hold for all inputs (property-based).
    """

    @given(case=_cases(finite_floats(), finite_floats(), min_size=0))
    def test_matches_reference_for_any_input(
        self,
        case: tuple[list[float], list[float]],
    ) -> None:
        """
        Verifies that, for any aligned input series, the implementation matches the naive reference.
        """
        returns_gross, cost = case
        assert_matches(
            apply_returns_net(returns_gross, cost),
            returns_net_reference(returns_gross, cost),
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    @given(case=_cases(missing_data_floats(), missing_data_floats(), min_size=0))
    def test_matches_reference_under_missing_data(
        self,
        case: tuple[list[float | None], list[float | None]],
    ) -> None:
        """
        Verifies that, for inputs freely mixing null / NaN / finite, the implementation matches the naive reference.
        """
        returns_gross, cost = case
        assert_matches(
            apply_returns_net(returns_gross, cost),
            returns_net_reference(returns_gross, cost),
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    @given(case=_cases(finite_floats(), finite_floats()), exponent=st.sampled_from([-4, -3, -2, -1, 1, 2, 3, 4]))
    def test_scale_homogeneity(
        self,
        case: tuple[list[float], list[float]],
        exponent: int,
    ) -> None:
        """
        Verifies degree-1 homogeneity: scaling both inputs by a constant scales the net return by the same constant.
        ``k`` is a power of two so the rescaling is lossless.
        """
        k = 2.0**exponent
        returns_gross, cost = case
        result_base = apply_returns_net(returns_gross, cost)
        result_scaled = apply_returns_net([value * k for value in returns_gross], [value * k for value in cost])
        assert_scale_homogeneous(result_scaled, result_base, k=k, degree=1)

    @given(case=_cases(finite_floats(), finite_floats()), scale=st.sampled_from([1e-6, 1e6, 1e9]))
    def test_matches_reference_at_large_magnitude(
        self,
        case: tuple[list[float], list[float]],
        scale: float,
    ) -> None:
        """
        Verifies that at extreme magnitudes the implementation stays finite where the reference is and agrees.
        """
        gross_base, cost_base = case
        returns_gross = [value * scale for value in gross_base]
        cost = [value * scale for value in cost_base]
        assert_matches(
            apply_returns_net(returns_gross, cost),
            returns_net_reference(returns_gross, cost),
            rel_tol=RELATIVE_TOLERANCE_SCALE,
            abs_tol=ABSOLUTE_TOLERANCE_STREAMING,
        )
