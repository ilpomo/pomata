"""
Tests for ``pomata.pnl.cost_notional`` — the bps-of-traded-notional commission (currency).

``cost_notional`` is two-input (``quantity`` / ``price``) plus a scalar ``rate``; it is ``turnover(quantity) * price *
rate``, so it inherits the flat start and turnover's null / NaN rule and adds the price multiply. Tests use a local
``apply_cost_notional`` helper to materialize it over a two-column ``Float64`` frame; ``assert_matches`` and the naive
``cost_notional_reference`` oracle are shared. The cost is degree-1 homogeneous in both ``quantity`` and ``price``,
so it carries the scale-homogeneity and large-magnitude tiers.

The ladder is the canonical one: contract (type / shape / lazy-eager / ``.over`` per-group independence), edge
(flat-start / single-row / null / NaN / negative-rate guard), correctness (vs the closed-form reference and a frozen
golden master), and properties (reference agreement incl. missing data, scale-homogeneity, large-magnitude). Categories
are split into classes; cross-cutting categories use markers (see ``tests/README.md``).
"""

import math
from collections.abc import Sequence

import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st
from polars.testing import assert_frame_equal
from tests.pnl.oracles import cost_notional_reference
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

from pomata.pnl import cost_notional

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- cost_notional scales turnover by the price (W = 0, flat start, M = 0); a case is a pair of equal-length
# series (quantity, price) plus a scalar rate. Degree-1 homogeneous in quantity and in price, so it keeps the
# scale-homogeneity and large-magnitude tiers. Repetitions N are the shared CI profile (tests/conftest.py).
# ----------------------------------------------------------------------------------------------------------------------
SERIES_MAX = 50
QUANTITY = "quantity"
PRICE = "price"
RATE = 0.001  # the deterministic-test rate (10 bps of notional)

_RATES = st.floats(min_value=0.0, max_value=0.1, allow_nan=False, allow_infinity=False)


@st.composite
def _cases[T](
    draw: st.DrawFn,
    quantities: st.SearchStrategy[T],
    prices: st.SearchStrategy[T],
    min_size: int = 1,
) -> tuple[list[T], list[T]]:
    """
    A pair of equal-length series (quantities, prices); windowless with a flat start, so every row is a defined output.
    """
    length = draw(st.integers(min_value=min_size, max_value=SERIES_MAX))
    quantity = draw(st.lists(quantities, min_size=length, max_size=length))
    price = draw(st.lists(prices, min_size=length, max_size=length))
    return quantity, price


def apply_cost_notional(
    quantity: Sequence[float | None],
    price: Sequence[float | None],
    rate: float = RATE,
) -> list[float | None]:
    """
    Materialize ``cost_notional`` over a two-column ``Float64`` frame from the aligned input lists.
    """
    return materialize({QUANTITY: quantity, PRICE: price}, cost_notional(pl.col(QUANTITY), pl.col(PRICE), rate))


class TestCostNotionalContract:
    """
    Type, shape, and lazy/eager guarantees.
    """

    def test_returns_expr(self) -> None:
        """
        Verifies that the factory returns a ``pl.Expr`` without touching a frame.
        """
        assert isinstance(cost_notional(pl.col(QUANTITY), pl.col(PRICE), RATE), pl.Expr)

    def test_preserves_length_and_dtype(self) -> None:
        """
        Verifies that the output has one value per input row and is ``Float64``.
        """
        frame = pl.DataFrame(
            {
                QUANTITY: pl.Series(QUANTITY, [10.0, 10.0, -5.0, -5.0, 20.0], dtype=pl.Float64),
                PRICE: pl.Series(PRICE, [100.0, 102.0, 101.0, 104.0, 103.0], dtype=pl.Float64),
            }
        )
        result = frame.select(cost_notional(pl.col(QUANTITY), pl.col(PRICE), RATE).alias("y"))
        assert result.height == frame.height
        assert result.schema["y"] == pl.Float64

    def test_lazy_eager_parity(self) -> None:
        """
        Verifies that eager and lazy application produce identical materialized output.
        """
        frame = pl.DataFrame(
            {
                QUANTITY: pl.Series(QUANTITY, [10.0, 10.0, -5.0, -5.0, 20.0], dtype=pl.Float64),
                PRICE: pl.Series(PRICE, [100.0, 102.0, 101.0, 104.0, 103.0], dtype=pl.Float64),
            }
        )
        expr = cost_notional(pl.col(QUANTITY), pl.col(PRICE), RATE).alias("y")
        result_eager = frame.select(expr)
        result_lazy = frame.lazy().select(expr).collect()
        assert_frame_equal(result_eager, result_lazy)

    def test_over_partitions_independently(self) -> None:
        """
        Verifies that under ``.over`` the turnover resets per group (each group gets its own flat start).
        """
        frame = pl.DataFrame(
            {
                "ticker": ["a"] * 3 + ["b"] * 3,
                QUANTITY: pl.Series(QUANTITY, [10.0, 10.0, -5.0, 2.0, 2.0, 2.0], dtype=pl.Float64),
                PRICE: pl.Series(PRICE, [100.0, 102.0, 101.0, 50.0, 51.0, 49.0], dtype=pl.Float64),
            }
        )
        grouped = frame.select(cost_notional(pl.col(QUANTITY), pl.col(PRICE), RATE).over("ticker").alias("y"))[
            "y"
        ].to_list()
        group_a = apply_cost_notional([10.0, 10.0, -5.0], [100.0, 102.0, 101.0])
        group_b = apply_cost_notional([2.0, 2.0, 2.0], [50.0, 51.0, 49.0])
        assert_matches(grouped, group_a + group_b)


class TestCostNotionalEdge:
    """
    Boundaries, the flat start, null / NaN handling, and the rate guard.
    """

    def test_flat_start_first_row(self) -> None:
        """
        Verifies the first row charges on ``|quantity_0| * price_0`` (the entry trade from a flat start).
        """
        assert_matches(apply_cost_notional([10.0, 10.0, -5.0], [100.0, 102.0, 101.0]), [1.0, 0.0, 1.515])

    def test_single_row(self) -> None:
        """
        Verifies that a one-row series charges on the entry trade (not null).
        """
        assert_matches(apply_cost_notional([10.0], [100.0]), [1.0])

    def test_empty(self) -> None:
        """
        Verifies that an empty input yields an empty output.
        """
        assert_matches(apply_cost_notional([], []), [])

    def test_all_null(self) -> None:
        """
        Verifies that an all-null input yields an all-null output.
        """
        assert_matches(apply_cost_notional([None, None, None], [None, None, None]), [None, None, None])

    def test_null_propagates(self) -> None:
        """
        Verifies that a ``null`` in the quantity or the price voids the affected rows (matching the reference).
        """
        quantity = [10.0, None, -5.0, 20.0]
        price = [100.0, 102.0, 101.0, 104.0]
        assert_matches(apply_cost_notional(quantity, price), cost_notional_reference(quantity, price, RATE))

    def test_nan_propagates(self) -> None:
        """
        Verifies that a ``NaN`` propagates to the rows that reference it (matching the reference).
        """
        quantity = [10.0, 10.0, -5.0, 20.0]
        price = [100.0, 102.0, math.nan, 104.0]
        assert_matches(apply_cost_notional(quantity, price), cost_notional_reference(quantity, price, RATE))

    def test_invalid_rate_raises(self) -> None:
        """
        Verifies that a rate that is not a finite number ``>= 0`` (negative, ``NaN``, or ``±inf``) raises
        ``ValueError`` -- a cost rate is a finite non-negative number, so a non-finite value fails fast at the call site
        rather than silently poisoning the output with ``NaN`` / ``inf``.
        """
        for invalid in (-0.001, math.nan, math.inf, -math.inf):
            with pytest.raises(ValueError, match="rate must be a finite number >= 0"):
                cost_notional(pl.col(QUANTITY), pl.col(PRICE), invalid)


class TestCostNotionalCorrectness:
    """
    Against the naive reference oracle and frozen golden-master values.
    """

    def test_matches_reference(self) -> None:
        """
        Verifies agreement with the naive closed-form reference over a representative series.
        """
        quantity = [10.0, 10.0, -5.0, -5.0, 20.0, 15.0, -8.0, 12.0]
        price = [100.0, 102.0, 101.0, 104.0, 103.0, 106.0, 105.0, 108.0]
        assert_matches(
            apply_cost_notional(quantity, price),
            cost_notional_reference(quantity, price, RATE),
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference over a five-bar series at 10 bps of notional.
        """
        result = materialize(
            {QUANTITY: [10.0, 10.0, -5.0, -5.0, 20.0], PRICE: [100.0, 102.0, 101.0, 104.0, 103.0]},
            cost_notional(pl.col(QUANTITY), pl.col(PRICE), RATE).round(4),
        )
        assert_matches(result, [1.0, 0.0, 1.515, 0.0, 2.575])


class TestCostNotionalProperties:
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
            apply_cost_notional(quantity, price, rate),
            cost_notional_reference(quantity, price, rate),
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
            apply_cost_notional(quantity, price, rate),
            cost_notional_reference(quantity, price, rate),
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
        Verifies degree-1 homogeneity in the quantity: scaling it by a constant scales the cost by the same constant
        (the price and rate held fixed). ``k`` is a power of two so the rescaling is lossless.
        """
        k = 2.0**exponent
        quantity, price = case
        result_base = apply_cost_notional(quantity, price)
        result_scaled = apply_cost_notional([value * k for value in quantity], price)
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
            apply_cost_notional(quantity, price),
            cost_notional_reference(quantity, price, RATE),
            rel_tol=RELATIVE_TOLERANCE_SCALE,
            abs_tol=ABSOLUTE_TOLERANCE_STREAMING,
        )
