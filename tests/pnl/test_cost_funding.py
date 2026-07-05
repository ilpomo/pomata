"""
Tests for ``pomata.pnl.cost_funding`` — the per-bar perpetual-swap funding cost of a held position.

``cost_funding`` is three-input (``quantity`` / ``price`` / ``rate``) and elementwise (a pure per-row product, no
window, no lag). Tests use a local ``apply_cost_funding`` helper to materialize it over a three-column ``Float64``
frame; ``assert_matches`` and the naive ``cost_funding_reference`` oracle are shared across the suite. The product is
degree-1 homogeneous in each input, so it carries the scale-homogeneity and large-magnitude tiers; the funding rate is
signed, so the sign of the cost follows ``sign(quantity) * sign(rate)``.

The ladder, adapted to an elementwise three-input product: contract (type / shape / lazy-eager / ``.over`` identity),
edge (empty / single-row / null / NaN / null-precedence / sign / zero-rate), correctness (closed-form reference + frozen
golden master), and properties (reference agreement incl. missing data, scale-homogeneity, large-magnitude). Categories
are split into classes; cross-cutting categories use markers (see ``tests/README.md``).
"""

import math
from collections.abc import Sequence

import polars as pl
from hypothesis import given
from hypothesis import strategies as st
from tests.pnl.oracles import cost_funding_reference
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

from pomata.pnl import cost_funding

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- cost_funding is a windowless elementwise product of three aligned series (W = 0, M = 0): a case is a
# triple of equal-length series. Degree-1 homogeneous in each input, so it keeps the scale-homogeneity and large-
# magnitude tiers. Repetitions N are the shared CI profile (tests/conftest.py).
# ----------------------------------------------------------------------------------------------------------------------
SERIES_MAX = 50
QUANTITY = "quantity"
PRICE = "price"
RATE = "rate"


@st.composite
def _cases[T](
    draw: st.DrawFn,
    quantities: st.SearchStrategy[T],
    prices: st.SearchStrategy[T],
    rates: st.SearchStrategy[T],
    min_size: int = 1,
) -> tuple[list[T], list[T], list[T]]:
    """
    A triple of equal-length series (quantities, prices, funding rates); windowless, so every row is a defined output.
    """
    length = draw(st.integers(min_value=min_size, max_value=SERIES_MAX))
    quantity = draw(st.lists(quantities, min_size=length, max_size=length))
    price = draw(st.lists(prices, min_size=length, max_size=length))
    rate = draw(st.lists(rates, min_size=length, max_size=length))
    return quantity, price, rate


def apply_cost_funding(
    quantity: Sequence[float | None],
    price: Sequence[float | None],
    rate: Sequence[float | None],
) -> list[float | None]:
    """
    Materialize ``cost_funding`` over a three-column ``Float64`` frame from the aligned input lists.
    """
    return materialize(
        {QUANTITY: quantity, PRICE: price, RATE: rate},
        cost_funding(pl.col(QUANTITY), pl.col(PRICE), pl.col(RATE)),
    )


class TestCostFundingContract:
    """
    Type, shape, and lazy/eager guarantees.
    """

    def test_over_is_identity(self) -> None:
        """
        Verifies that ``.over`` is optional for this elementwise product: partitioning by group is identical to the
        un-partitioned call.
        """
        frame = pl.DataFrame(
            {
                "ticker": ["a", "a", "a", "b", "b", "b"],
                QUANTITY: pl.Series(QUANTITY, [10.0, -5.0, -5.0, 30.0, 30.0, 30.0], dtype=pl.Float64),
                PRICE: pl.Series(PRICE, [100.0, 101.0, 104.0, 50.0, 51.0, 49.0], dtype=pl.Float64),
                RATE: pl.Series(RATE, [0.0001, 0.0001, -0.0001, 0.0001, -0.0001, 0.0001], dtype=pl.Float64),
            }
        )
        plain = frame.select(cost_funding(pl.col(QUANTITY), pl.col(PRICE), pl.col(RATE)).alias("y"))["y"].to_list()
        grouped = frame.select(cost_funding(pl.col(QUANTITY), pl.col(PRICE), pl.col(RATE)).over("ticker").alias("y"))[
            "y"
        ].to_list()
        assert_matches(plain, grouped)


class TestCostFundingEdge:
    """
    Boundaries, null / NaN handling, and the funding-sign convention.
    """

    def test_single_row(self) -> None:
        """
        Verifies that a one-row series resolves to the single product.
        """
        assert_matches(apply_cost_funding([10.0], [100.0], [0.0001]), [0.1])

    def test_sign_follows_quantity_and_rate(self) -> None:
        """
        Verifies the funding-sign convention over the 2x2 sign matrix: a long pays a positive rate and is rebated by a
        negative one; a short is the mirror image. The cost sign is ``sign(quantity) * sign(rate)``.
        """
        quantity = [10.0, -5.0, 10.0, -5.0]
        price = [100.0, 100.0, 100.0, 100.0]
        rate = [0.0001, 0.0001, -0.0001, -0.0001]
        assert_matches(apply_cost_funding(quantity, price, rate), [0.1, -0.05, -0.1, 0.05])

    def test_zero_rate_is_free(self) -> None:
        """
        Verifies that an off-funding bar (``rate = 0``) costs nothing, whatever the position and price.
        """
        assert_matches(apply_cost_funding([10.0, -5.0, 20.0], [100.0, 101.0, 102.0], [0.0, 0.0, 0.0]), [0.0, 0.0, 0.0])

    def test_null_propagates(self) -> None:
        """
        Verifies that a ``null`` in any input makes that row ``null`` (matching the naive reference).
        """
        quantity = [10.0, None, -5.0, 20.0]
        price = [100.0, 102.0, 101.0, 104.0]
        rate = [0.0001, 0.0001, 0.0001, 0.0001]
        assert_matches(apply_cost_funding(quantity, price, rate), cost_funding_reference(quantity, price, rate))

    def test_nan_propagates(self) -> None:
        """
        Verifies that a ``NaN`` in any input makes that row ``NaN`` (matching the naive reference).
        """
        quantity = [10.0, 10.0, math.nan, 20.0]
        price = [100.0, 102.0, 101.0, 104.0]
        rate = [0.0001, 0.0001, 0.0001, 0.0001]
        assert_matches(apply_cost_funding(quantity, price, rate), cost_funding_reference(quantity, price, rate))

    def test_null_takes_precedence_over_nan(self) -> None:
        """
        Verifies that a row with a ``null`` in one input and a ``NaN`` in another yields ``null``.
        """
        assert_matches(apply_cost_funding([None, 10.0], [math.nan, 100.0], [0.0001, 0.0001]), [None, 0.1])


class TestCostFundingCorrectness:
    """
    Against the naive reference oracle and frozen golden-master values.
    """

    def test_matches_reference(self) -> None:
        """
        Verifies agreement with the naive closed-form reference over a representative series.
        """
        quantity = [10.0, 10.0, -5.0, -5.0, 20.0, 20.0, -8.0, -8.0]
        price = [100.0, 102.0, 101.0, 104.0, 103.0, 105.0, 99.0, 98.0]
        rate = [0.0001, 0.0001, 0.0001, -0.0001, 0.0001, -0.0002, 0.0001, 0.0]
        assert_matches(
            apply_cost_funding(quantity, price, rate),
            cost_funding_reference(quantity, price, rate),
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference over a five-bar series (long pays, short is rebated, one rate flips sign).
        """
        result = materialize(
            {
                QUANTITY: [10.0, 10.0, -5.0, -5.0, 20.0],
                PRICE: [100.0, 102.0, 101.0, 104.0, 103.0],
                RATE: [0.0001, 0.0001, 0.0001, -0.0001, 0.0001],
            },
            cost_funding(pl.col(QUANTITY), pl.col(PRICE), pl.col(RATE)).round(6),
        )
        assert_matches(result, [0.1, 0.102, -0.0505, 0.052, 0.206])


class TestCostFundingProperties:
    """
    Invariants that must hold for all inputs (property-based).
    """

    @given(case=_cases(finite_floats(), finite_floats(), finite_floats(), min_size=0))
    def test_matches_reference_for_any_input(
        self,
        case: tuple[list[float], list[float], list[float]],
    ) -> None:
        """
        Verifies that, for any aligned input series, the implementation matches the naive reference.
        """
        quantity, price, rate = case
        assert_matches(
            apply_cost_funding(quantity, price, rate),
            cost_funding_reference(quantity, price, rate),
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    @given(
        case=_cases(missing_data_floats(), missing_data_floats(), missing_data_floats(), min_size=0),
    )
    def test_matches_reference_under_missing_data(
        self,
        case: tuple[list[float | None], list[float | None], list[float | None]],
    ) -> None:
        """
        Verifies that, for inputs freely mixing null / NaN / finite, the implementation matches the naive reference.
        """
        quantity, price, rate = case
        assert_matches(
            apply_cost_funding(quantity, price, rate),
            cost_funding_reference(quantity, price, rate),
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    @given(
        case=_cases(finite_floats(), finite_floats(), finite_floats()),
        axis=st.sampled_from([QUANTITY, PRICE, RATE]),
        exponent=st.sampled_from([-4, -3, -2, -1, 1, 2, 3, 4]),
    )
    def test_scale_homogeneity_in_each_input(
        self,
        case: tuple[list[float], list[float], list[float]],
        axis: str,
        exponent: int,
    ) -> None:
        """
        Verifies degree-1 homogeneity in each input: scaling any one of quantity, price, or rate by a constant scales
        the funding cost by the same constant. ``k`` is a power of two so the rescaling is lossless.
        """
        k = 2.0**exponent
        quantity, price, rate = case
        scaled = {
            QUANTITY: [value * k for value in quantity] if axis == QUANTITY else quantity,
            PRICE: [value * k for value in price] if axis == PRICE else price,
            RATE: [value * k for value in rate] if axis == RATE else rate,
        }
        result_base = apply_cost_funding(quantity, price, rate)
        result_scaled = apply_cost_funding(scaled[QUANTITY], scaled[PRICE], scaled[RATE])
        assert_scale_homogeneous(result_scaled, result_base, k=k, degree=1)

    @given(case=_cases(finite_floats(), finite_floats(), finite_floats()), scale=st.sampled_from([1e-6, 1e6, 1e9]))
    def test_matches_reference_at_large_magnitude(
        self,
        case: tuple[list[float], list[float], list[float]],
        scale: float,
    ) -> None:
        """
        Verifies that at extreme magnitudes the implementation stays finite where the reference is and agrees.
        """
        quantity_base, price_base, rate_base = case
        quantity = [value * scale for value in quantity_base]
        price = [value * scale for value in price_base]
        rate = [value * scale for value in rate_base]
        assert_matches(
            apply_cost_funding(quantity, price, rate),
            cost_funding_reference(quantity, price, rate),
            rel_tol=RELATIVE_TOLERANCE_SCALE,
            abs_tol=ABSOLUTE_TOLERANCE_STREAMING,
        )
