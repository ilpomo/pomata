"""
Tests for ``pomata.pnl.cumulative_pnl`` — the additive running total of a return series.

``cumulative_pnl`` is single-input and cumulative (a running sum), so tests use the shared ``apply_expr`` helper to
materialize the factory over a one-column ``Float64`` frame; ``assert_matches`` and the naive
``cumulative_pnl_reference`` oracle are shared across the suite. The running sum is degree-1 homogeneous, so it
carries the scale-homogeneity and large-magnitude tiers, plus a running-difference metamorphic.

The ladder is the canonical one: contract (type / shape / lazy-eager / ``.over`` per-group independence), edge
(single-row / interior-null continuity / NaN / warm-up), correctness (vs the closed-form reference and a frozen
golden master), and properties (reference agreement incl. missing data, scale-homogeneity, large-magnitude, the
running-difference identity). Categories are split into classes; cross-cutting categories use markers (see
``tests/README.md``).
"""

import math

import polars as pl
from hypothesis import given
from hypothesis import strategies as st
from tests.pnl.oracles import cumulative_pnl_reference
from tests.support import (
    ABSOLUTE_TOLERANCE_REFERENCE,
    ABSOLUTE_TOLERANCE_STREAMING,
    COLUMN_X,
    EXACT_TOLERANCE_FACTOR,
    GROUP_KEY,
    RELATIVE_TOLERANCE_PROPERTY,
    RELATIVE_TOLERANCE_REFERENCE,
    RELATIVE_TOLERANCE_SCALE,
    apply_expr,
    assert_matches,
    assert_scale_homogeneous,
    finite_floats,
    input_scale,
    missing_data_floats,
    subnormal_safe_floats,
)

from pomata.pnl import cumulative_pnl

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- derived, not chosen; the rationale is the shared method in CORRECTNESS.md, the numbers are this
# primitive's. cumulative_pnl is a windowless cumulative sum (M = 0); a case is just the return series. It is degree-1
# homogeneous, so it keeps the scale-homogeneity and large-magnitude tiers; the running-difference metamorphic uses a
# magnitude-relative floor because differencing two large partial sums cancels (the same conditioning as the
# accumulation/distribution line). Repetitions N are the shared CI profile (tests/conftest.py); override per-test only
# if its parameter space is larger.
# ----------------------------------------------------------------------------------------------------------------------
SERIES_MAX = 50


@st.composite
def _cases[T](draw: st.DrawFn, returns: st.SearchStrategy[T], min_size: int = 1) -> list[T]:
    """
    A return series sized from the facts above. cumulative_pnl is windowless, so a case is just the series.
    """
    return draw(st.lists(returns, min_size=min_size, max_size=SERIES_MAX))


class TestCumulativePnlContract:
    """
    Type, shape, and lazy/eager guarantees.
    """

    def test_over_partitions_independently(self) -> None:
        """
        Verifies that under ``.over`` the running sum restarts per group and never carries across boundaries.
        """
        frame = pl.DataFrame({GROUP_KEY: ["a"] * 3 + ["b"] * 3, COLUMN_X: [0.1, 0.2, -0.05, 0.0, 0.1, 0.1]})
        expr = cumulative_pnl(pl.col(COLUMN_X)).over(GROUP_KEY)
        grouped = frame.select(expr.alias("y"))["y"].to_list()
        group_a = apply_expr([0.1, 0.2, -0.05], cumulative_pnl(pl.col(COLUMN_X)))
        group_b = apply_expr([0.0, 0.1, 0.1], cumulative_pnl(pl.col(COLUMN_X)))
        assert_matches(grouped, group_a + group_b)


class TestCumulativePnlEdge:
    """
    Boundaries, warm-up, and null / NaN handling.
    """

    def test_single_row(self) -> None:
        """
        Verifies that a one-element series resolves to that single return (no warm-up of its own).
        """
        assert_matches(apply_expr([0.1], cumulative_pnl(pl.col(COLUMN_X)).round(4)), [0.1])

    def test_null_carries_across(self) -> None:
        """
        Verifies that an interior null emits null at that row while the running sum carries across it unchanged
        (matching the naive reference).
        """
        values = [0.1, None, 0.2, -0.05]
        assert_matches(apply_expr(values, cumulative_pnl(pl.col(COLUMN_X))), cumulative_pnl_reference(values))

    def test_nan_latches(self) -> None:
        """
        Verifies that a NaN enters the running sum and every later row is NaN (matching the naive reference).
        """
        values = [0.1, math.nan, 0.2, -0.05]
        assert_matches(apply_expr(values, cumulative_pnl(pl.col(COLUMN_X))), cumulative_pnl_reference(values))

    def test_warmup_leading_null(self) -> None:
        """
        Verifies a leading warm-up null (as produced by returns_simple) stays null and the sum begins at the first
        defined return.
        """
        result = apply_expr([None, 0.1, 0.2, -0.05], cumulative_pnl(pl.col(COLUMN_X)))
        assert result[0] is None
        assert result[1] is not None


class TestCumulativePnlCorrectness:
    """
    Against the naive reference oracle and frozen golden-master values.
    """

    def test_matches_reference(self) -> None:
        """
        Verifies agreement with the naive closed-form reference over a representative return series.
        """
        values = [0.1, -0.05, 0.2, 0.1, -0.02, 0.03, -0.1, 0.04]
        assert_matches(
            apply_expr(values, cumulative_pnl(pl.col(COLUMN_X))),
            cumulative_pnl_reference(values),
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference over a four-bar return series.
        """
        result = apply_expr([0.1, -0.05, 0.2, 0.1], cumulative_pnl(pl.col(COLUMN_X)).round(4))
        assert_matches(result, [0.1, 0.05, 0.25, 0.35])


class TestCumulativePnlProperties:
    """
    Invariants that must hold for all inputs (property-based).
    """

    @given(case=_cases(finite_floats(), min_size=0))
    def test_matches_reference_for_any_input(
        self,
        case: list[float],
    ) -> None:
        """
        Verifies that, for any return series, the implementation matches the naive reference.
        """
        values = case
        assert_matches(
            apply_expr(values, cumulative_pnl(pl.col(COLUMN_X))),
            cumulative_pnl_reference(values),
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    @given(case=_cases(missing_data_floats(), min_size=0))
    def test_matches_reference_under_missing_data(
        self,
        case: list[float | None],
    ) -> None:
        """
        Verifies that, for inputs freely mixing null / NaN / finite, the implementation matches the naive reference.
        """
        values = case
        assert_matches(
            apply_expr(values, cumulative_pnl(pl.col(COLUMN_X))),
            cumulative_pnl_reference(values),
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    @given(case=_cases(subnormal_safe_floats()), exponent=st.sampled_from([-4, -3, -2, -1, 1, 2, 3, 4]))
    def test_scale_homogeneity(
        self,
        case: list[float],
        exponent: int,
    ) -> None:
        """
        Verifies that ``cumulative_pnl`` is homogeneous of degree 1: scaling every input value by a constant ``k``
        scales the output by the same ``k`` -- ``cumulative_pnl(k * x) == k * cumulative_pnl(x)``. ``k`` is a power
        of two, so the rescale is exact and adds no floating-point error.
        """
        k = 2.0**exponent
        values = case
        result_base = apply_expr(values, cumulative_pnl(pl.col(COLUMN_X)))
        result_scaled = apply_expr([value * k for value in values], cumulative_pnl(pl.col(COLUMN_X)))
        assert_scale_homogeneous(result_scaled, result_base, k=k, degree=1)

    @given(case=_cases(finite_floats()), scale=st.sampled_from([1e-6, 1e6, 1e9]))
    def test_matches_reference_at_large_magnitude(
        self,
        case: list[float],
        scale: float,
    ) -> None:
        """
        Verifies that at extreme magnitudes the implementation stays finite where the reference is and agrees.
        """
        values = [value * scale for value in case]
        assert_matches(
            apply_expr(values, cumulative_pnl(pl.col(COLUMN_X))),
            cumulative_pnl_reference(values),
            rel_tol=RELATIVE_TOLERANCE_SCALE,
            abs_tol=ABSOLUTE_TOLERANCE_STREAMING,
        )

    @given(case=_cases(finite_floats()))
    def test_running_difference_recovers_returns(
        self,
        case: list[float],
    ) -> None:
        """
        Verifies the running-difference identity: the first row is the first return and each later row minus its
        predecessor recovers that bar's return. The floor is magnitude-relative because differencing two large partial
        sums cancels (the same conditioning as the accumulation/distribution line).
        """
        values = case
        cumulative = apply_expr(values, cumulative_pnl(pl.col(COLUMN_X)))
        tolerance = input_scale(cumulative) * EXACT_TOLERANCE_FACTOR
        assert cumulative[0] is not None
        assert math.isclose(cumulative[0], values[0], rel_tol=RELATIVE_TOLERANCE_PROPERTY, abs_tol=tolerance)
        for index in range(1, len(values)):
            current = cumulative[index]
            previous = cumulative[index - 1]
            assert current is not None
            assert previous is not None
            assert math.isclose(
                current - previous, values[index], rel_tol=RELATIVE_TOLERANCE_PROPERTY, abs_tol=tolerance
            )
