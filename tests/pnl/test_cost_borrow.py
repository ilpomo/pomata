"""
Tests for ``pomata.pnl.cost_borrow`` — the per-bar short-borrow cost.

``cost_borrow`` is two-input (``quantity`` / ``price``) plus a scalar ``rate`` and elementwise (a per-row charge on the
short part of the position, no window, no lag). Tests use a local ``apply_cost_borrow`` helper to materialize it over a
two-column ``Float64`` frame; ``assert_matches`` and the naive ``cost_borrow_reference`` oracle are shared. The cost is
degree-1 homogeneous in both ``quantity`` and ``price``, so it carries the scale-homogeneity and large-magnitude tiers.

The ladder, adapted to an elementwise two-input charge: contract (type / shape / lazy-eager / ``.over`` identity), edge
(long-or-flat zero / single-row / null / NaN / null-precedence / negative-rate guard), correctness (closed-form
reference + frozen golden master), and properties (reference agreement incl. missing data, scale-homogeneity,
large-magnitude). Categories are split into classes; cross-cutting categories use markers (see ``tests/README.md``).
"""

import math
from collections.abc import Sequence

import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st
from tests.pnl.oracles import cost_borrow_reference
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

from pomata.pnl import cost_borrow

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- cost_borrow is a windowless elementwise charge on the short part of two aligned series (W = 0, M = 0):
# a case is a pair of equal-length series (quantity, price) plus a scalar rate. Degree-1 homogeneous in quantity and in
# price, so it keeps the scale-homogeneity and large-magnitude tiers. Repetitions N are the shared CI profile.
# ----------------------------------------------------------------------------------------------------------------------
SERIES_MAX = 50
QUANTITY = "quantity"
PRICE = "price"
RATE = 0.0001  # the deterministic-test per-bar borrow rate

_RATES = st.floats(min_value=0.0, max_value=0.01, allow_nan=False, allow_infinity=False)


@st.composite
def _cases[T](
    draw: st.DrawFn,
    quantities: st.SearchStrategy[T],
    prices: st.SearchStrategy[T],
    min_size: int = 1,
) -> tuple[list[T], list[T]]:
    """
    A pair of equal-length series (quantities, prices); windowless, so every row is a defined output.
    """
    length = draw(st.integers(min_value=min_size, max_value=SERIES_MAX))
    quantity = draw(st.lists(quantities, min_size=length, max_size=length))
    price = draw(st.lists(prices, min_size=length, max_size=length))
    return quantity, price


def apply_cost_borrow(
    quantity: Sequence[float | None],
    price: Sequence[float | None],
    rate: float = RATE,
) -> list[float | None]:
    """
    Materialize ``cost_borrow`` over a two-column ``Float64`` frame from the aligned input lists.
    """
    return materialize({QUANTITY: quantity, PRICE: price}, cost_borrow(pl.col(QUANTITY), pl.col(PRICE), rate))


class TestCostBorrowContract:
    """
    Type, shape, and lazy/eager guarantees.
    """

    def test_over_is_identity(self) -> None:
        """
        Verifies that ``.over`` is optional for this elementwise charge: partitioning by group is identical to the
        un-partitioned call.
        """
        frame = pl.DataFrame(
            {
                "ticker": ["a", "a", "a", "b", "b", "b"],
                QUANTITY: pl.Series(QUANTITY, [-50.0, -50.0, 0.0, -20.0, -20.0, 30.0], dtype=pl.Float64),
                PRICE: pl.Series(PRICE, [10.0, 11.0, 12.0, 13.0, 14.0, 15.0], dtype=pl.Float64),
            }
        )
        plain = frame.select(cost_borrow(pl.col(QUANTITY), pl.col(PRICE), RATE).alias("y"))["y"].to_list()
        grouped = frame.select(cost_borrow(pl.col(QUANTITY), pl.col(PRICE), RATE).over("ticker").alias("y"))[
            "y"
        ].to_list()
        assert_matches(plain, grouped)


class TestCostBorrowEdge:
    """
    Boundaries, the long/flat zero, null / NaN handling, and the rate guard.
    """

    def test_long_or_flat_is_zero(self) -> None:
        """
        Verifies that a long or flat quantity has zero borrow cost (only the short part is charged).
        """
        assert_matches(apply_cost_borrow([100.0, 0.0, 50.0], [10.0, 11.0, 12.0]), [0.0, 0.0, 0.0])

    def test_single_row(self) -> None:
        """
        Verifies that a one-row short series resolves to its borrow cost.
        """
        assert_matches(apply_cost_borrow([-50.0], [10.0]), [0.05])

    def test_null_propagates(self) -> None:
        """
        Verifies that a ``null`` in the quantity or the price makes that row ``null`` (matching the reference).
        """
        quantity = [-50.0, None, -50.0, -20.0]
        price = [10.0, 11.0, 12.0, 13.0]
        assert_matches(apply_cost_borrow(quantity, price), cost_borrow_reference(quantity, price, RATE))

    def test_nan_propagates(self) -> None:
        """
        Verifies that a ``NaN`` in either input makes that row ``NaN`` (matching the reference).
        """
        quantity = [-50.0, -50.0, math.nan, -20.0]
        price = [10.0, 11.0, 12.0, 13.0]
        assert_matches(apply_cost_borrow(quantity, price), cost_borrow_reference(quantity, price, RATE))

    def test_null_takes_precedence_over_nan(self) -> None:
        """
        Verifies that a row with a ``null`` in one input and a ``NaN`` in the other yields ``null``.
        """
        assert_matches(apply_cost_borrow([None, -50.0], [math.nan, 10.0]), [None, 0.05])

    def test_invalid_rate_raises(self) -> None:
        """
        Verifies that a rate that is not a finite number ``>= 0`` (negative, ``NaN``, or ``±inf``) raises
        ``ValueError`` -- a borrow rate is a finite non-negative number, so a non-finite value fails fast at the call
        site rather than silently poisoning the output with ``NaN`` / ``inf``.
        """
        for invalid in (-0.0001, math.nan, math.inf, -math.inf):
            with pytest.raises(ValueError, match="rate must be a finite number >= 0"):
                cost_borrow(pl.col(QUANTITY), pl.col(PRICE), invalid)


class TestCostBorrowCorrectness:
    """
    Against the naive reference oracle and frozen golden-master values.
    """

    def test_matches_reference(self) -> None:
        """
        Verifies agreement with the naive closed-form reference over a representative series.
        """
        quantity = [100.0, -50.0, -50.0, -20.0, -20.0, 0.0, -80.0, 40.0]
        price = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0]
        assert_matches(
            apply_cost_borrow(quantity, price),
            cost_borrow_reference(quantity, price, RATE),
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference over a five-bar series (long bar is zero, short bars charged).
        """
        result = materialize(
            {QUANTITY: [100.0, -50.0, -50.0, -20.0, -20.0], PRICE: [10.0, 11.0, 12.0, 13.0, 14.0]},
            cost_borrow(pl.col(QUANTITY), pl.col(PRICE), RATE).round(6),
        )
        assert_matches(result, [0.0, 0.055, 0.06, 0.026, 0.028])


class TestCostBorrowProperties:
    """
    Invariants that must hold for all inputs (property-based).
    """

    @given(case=_cases(finite_floats(), finite_floats(), min_size=0), rate=_RATES)
    def test_matches_reference_for_any_input(
        self,
        case: tuple[list[float], list[float]],
        rate: float,
    ) -> None:
        """
        Verifies that, for any aligned input series and non-negative rate, the implementation matches the reference.
        """
        quantity, price = case
        assert_matches(
            apply_cost_borrow(quantity, price, rate),
            cost_borrow_reference(quantity, price, rate),
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    @given(case=_cases(missing_data_floats(), missing_data_floats(), min_size=0), rate=_RATES)
    def test_matches_reference_under_missing_data(
        self,
        case: tuple[list[float | None], list[float | None]],
        rate: float,
    ) -> None:
        """
        Verifies that, for inputs freely mixing null / NaN / finite, the implementation matches the naive reference.
        """
        quantity, price = case
        assert_matches(
            apply_cost_borrow(quantity, price, rate),
            cost_borrow_reference(quantity, price, rate),
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    @given(case=_cases(finite_floats(), finite_floats()), exponent=st.sampled_from([-4, -3, -2, -1, 1, 2, 3, 4]))
    def test_scale_homogeneity_in_quantity(
        self,
        case: tuple[list[float], list[float]],
        exponent: int,
    ) -> None:
        """
        Verifies degree-1 homogeneity in the quantity: scaling it by a positive constant scales the borrow cost by the
        same constant (the short part scales linearly). ``k`` is a power of two so the rescaling is lossless.
        """
        k = 2.0**exponent
        quantity, price = case
        result_base = apply_cost_borrow(quantity, price)
        result_scaled = apply_cost_borrow([value * k for value in quantity], price)
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
        quantity_base, price_base = case
        quantity = [value * scale for value in quantity_base]
        price = [value * scale for value in price_base]
        assert_matches(
            apply_cost_borrow(quantity, price),
            cost_borrow_reference(quantity, price, RATE),
            rel_tol=RELATIVE_TOLERANCE_SCALE,
            abs_tol=ABSOLUTE_TOLERANCE_STREAMING,
        )
