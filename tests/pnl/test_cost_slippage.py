"""
Tests for ``pomata.pnl.cost_slippage`` — the fixed bid-ask half-spread transaction cost.

``cost_slippage`` is single-input (``weight``) plus a scalar ``half_spread``; it scales :func:`turnover`, so it
inherits the flat start (first row ``|weight_0| * half_spread``) and turnover's null / NaN rule. Tests use the shared
``apply_expr`` helper over a one-column ``Float64`` frame; ``assert_matches`` and the naive ``cost_slippage_reference``
oracle are shared. The cost is degree-1 homogeneous in the weight, so it carries the scale-homogeneity and
large-magnitude tiers.

The ladder is the canonical one: contract (type / shape / lazy-eager / ``.over`` per-group independence), edge
(negative-half-spread guard / single-row / null / NaN / flat-start / consecutive-infinities), correctness (vs the
closed-form reference and a frozen golden master), and properties (reference agreement incl. missing data,
scale-homogeneity, large-magnitude). Categories are split into classes; cross-cutting categories use markers (see
``tests/README.md``).
"""

import math

import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st
from tests.pnl.oracles import cost_slippage_reference
from tests.support import (
    ABSOLUTE_TOLERANCE_REFERENCE,
    ABSOLUTE_TOLERANCE_STREAMING,
    COLUMN_X,
    GROUP_KEY,
    RELATIVE_TOLERANCE_PROPERTY,
    RELATIVE_TOLERANCE_REFERENCE,
    RELATIVE_TOLERANCE_SCALE,
    apply_expr,
    assert_matches,
    assert_scale_homogeneous,
    finite_floats,
    missing_data_floats,
    subnormal_safe_floats,
)

from pomata.pnl import cost_slippage

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- derived, not chosen; the rationale is the shared method in CORRECTNESS.md, the numbers are this
# primitive's. cost_slippage scales turnover (W = 0, flat start, M = 0); a case is just the weight series plus a
# scalar half-spread. It is degree-1 homogeneous in the weight, so it keeps the scale-homogeneity and large-magnitude
# tiers. Repetitions N are the shared CI profile (tests/conftest.py); override per-test only if its space is larger.
# ----------------------------------------------------------------------------------------------------------------------
SERIES_MAX = 50
HALF_SPREAD = 0.002  # the deterministic-test half-spread (20 bps)

# Realistic non-negative half-spreads for the property tiers.
_HALF_SPREADS = st.floats(min_value=0.0, max_value=0.1, allow_nan=False, allow_infinity=False)


@st.composite
def _cases[T](draw: st.DrawFn, weights: st.SearchStrategy[T], min_size: int = 1) -> list[T]:
    """
    A weight series sized from the facts above. cost_slippage is windowless with a flat start, so a case is just the
    series; every row is a defined output.
    """
    return draw(st.lists(weights, min_size=min_size, max_size=SERIES_MAX))


class TestCostSlippageContract:
    """
    Type, shape, and lazy/eager guarantees.
    """

    def test_over_partitions_independently(self) -> None:
        """
        Verifies that under ``.over`` the turnover resets per group (each group gets its own flat start) and never
        reaches across group boundaries.
        """
        frame = pl.DataFrame({GROUP_KEY: ["a"] * 3 + ["b"] * 3, COLUMN_X: [0.5, 1.0, -0.5, 1.0, 1.0, 0.0]})
        expr = cost_slippage(pl.col(COLUMN_X), half_spread=HALF_SPREAD).over(GROUP_KEY)
        grouped = frame.select(expr.alias("y"))["y"].to_list()
        group_a = apply_expr([0.5, 1.0, -0.5], cost_slippage(pl.col(COLUMN_X), half_spread=HALF_SPREAD))
        group_b = apply_expr([1.0, 1.0, 0.0], cost_slippage(pl.col(COLUMN_X), half_spread=HALF_SPREAD))
        assert_matches(grouped, group_a + group_b)


class TestCostSlippageEdge:
    """
    Boundaries, the flat start, null / NaN handling, and the half-spread guard.
    """

    def test_invalid_half_spread_raises(self) -> None:
        """
        Verifies that a half-spread that is not a finite number ``>= 0`` (negative, ``NaN``, or ``±inf``) raises
        ``ValueError`` -- a spread is a finite non-negative number, so a non-finite value fails fast at the call site
        rather than silently poisoning the output with ``NaN`` / ``inf``.
        """
        for invalid in (-0.002, math.nan, math.inf, -math.inf):
            with pytest.raises(ValueError, match="half_spread must be a finite number >= 0"):
                cost_slippage(pl.col(COLUMN_X), half_spread=invalid)

    def test_single_row(self) -> None:
        """
        Verifies that a one-element series resolves to ``|weight_0| * half_spread`` (the entry trade), not null.
        """
        assert_matches(apply_expr([0.5], cost_slippage(pl.col(COLUMN_X), half_spread=HALF_SPREAD)), [0.001])

    def test_null_propagates(self) -> None:
        """
        Verifies that a null voids its own row and the next (via turnover), matching the naive reference.
        """
        values = [0.5, None, 1.0, -0.5]
        assert_matches(
            apply_expr(values, cost_slippage(pl.col(COLUMN_X), half_spread=HALF_SPREAD)),
            cost_slippage_reference(values, HALF_SPREAD),
        )

    def test_nan_propagates(self) -> None:
        """
        Verifies that a NaN propagates to its own row and the next (matching the naive reference).
        """
        values = [0.5, math.nan, 1.0, -0.5]
        assert_matches(
            apply_expr(values, cost_slippage(pl.col(COLUMN_X), half_spread=HALF_SPREAD)),
            cost_slippage_reference(values, HALF_SPREAD),
        )

    def test_flat_start_first_row(self) -> None:
        """
        Verifies the first row is ``|weight_0| * half_spread`` (the cost of the entry trade from a flat start).
        """
        assert_matches(
            apply_expr([0.5, 1.0, -0.5], cost_slippage(pl.col(COLUMN_X), half_spread=HALF_SPREAD)),
            [0.001, 0.001, 0.003],
        )

    def test_consecutive_infinities_make_nan(self) -> None:
        """
        Verifies the turnover basis carries Polars' IEEE result into the cost: two consecutive equal-sign infinities
        make ``inf - inf = NaN`` turnover at the second bar, so the cost there is ``NaN`` (matching the reference). The
        property tiers cannot reach this (their strategies set ``allow_infinity=False``), so it is pinned here.
        """
        values = [math.inf, math.inf, 1.0, -math.inf]
        assert_matches(
            apply_expr(values, cost_slippage(pl.col(COLUMN_X), half_spread=HALF_SPREAD)),
            cost_slippage_reference(values, HALF_SPREAD),
        )


class TestCostSlippageCorrectness:
    """
    Against the naive reference oracle and frozen golden-master values.
    """

    def test_matches_reference(self) -> None:
        """
        Verifies agreement with the naive closed-form reference over a representative weight series.
        """
        values = [0.5, 1.0, -0.5, -0.5, 0.0, 1.5, -1.0, 0.25]
        assert_matches(
            apply_expr(values, cost_slippage(pl.col(COLUMN_X), half_spread=HALF_SPREAD)),
            cost_slippage_reference(values, HALF_SPREAD),
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference over a five-bar weight series at a 20 bps half-spread.
        """
        result = apply_expr(
            [0.5, 1.0, -0.5, -0.5, 0.0], cost_slippage(pl.col(COLUMN_X), half_spread=HALF_SPREAD).round(4)
        )
        assert_matches(result, [0.001, 0.001, 0.003, 0.0, 0.001])


class TestCostSlippageProperties:
    """
    Invariants that must hold for all inputs (property-based).
    """

    @given(case=_cases(finite_floats(), min_size=0), half_spread=_HALF_SPREADS)
    def test_matches_reference_for_any_input(
        self,
        case: list[float],
        half_spread: float,
    ) -> None:
        """
        Verifies that, for any weight series and non-negative half-spread, the implementation matches the reference.
        """
        values = case
        assert_matches(
            apply_expr(values, cost_slippage(pl.col(COLUMN_X), half_spread=half_spread)),
            cost_slippage_reference(values, half_spread),
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    @given(case=_cases(missing_data_floats(), min_size=0), half_spread=_HALF_SPREADS)
    def test_matches_reference_under_missing_data(
        self,
        case: list[float | None],
        half_spread: float,
    ) -> None:
        """
        Verifies that, for inputs freely mixing null / NaN / finite, the implementation matches the naive reference.
        """
        values = case
        assert_matches(
            apply_expr(values, cost_slippage(pl.col(COLUMN_X), half_spread=half_spread)),
            cost_slippage_reference(values, half_spread),
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
        Verifies that ``cost_slippage`` is homogeneous of degree 1: scaling every input value by a constant ``k``
        scales the output by the same ``k`` -- ``cost_slippage(k * x) == k * cost_slippage(x)``. ``k`` is a power of
        two, so the rescale is exact and adds no floating-point error.
        """
        k = 2.0**exponent
        values = case
        result_base = apply_expr(values, cost_slippage(pl.col(COLUMN_X), half_spread=HALF_SPREAD))
        result_scaled = apply_expr(
            [value * k for value in values], cost_slippage(pl.col(COLUMN_X), half_spread=HALF_SPREAD)
        )
        assert_scale_homogeneous(result_scaled, result_base, k=k, degree=1)

    @given(case=_cases(finite_floats()), scale=st.sampled_from([1e-6, 1e6, 1e9]), half_spread=_HALF_SPREADS)
    def test_matches_reference_at_large_magnitude(
        self,
        case: list[float],
        scale: float,
        half_spread: float,
    ) -> None:
        """
        Verifies that at extreme weight magnitudes the implementation stays finite where the reference is and agrees.
        """
        values = [value * scale for value in case]
        assert_matches(
            apply_expr(values, cost_slippage(pl.col(COLUMN_X), half_spread=half_spread)),
            cost_slippage_reference(values, half_spread),
            rel_tol=RELATIVE_TOLERANCE_SCALE,
            abs_tol=ABSOLUTE_TOLERANCE_STREAMING,
        )
