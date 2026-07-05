"""
Tests for ``pomata.pnl.pnl_gross`` — Gross Position PnL.

``pnl_gross`` is two-input (``quantity`` / ``price``) plus a scalar ``multiplier``, with a one-bar price lag, so tests
use a local ``apply_pnl_gross`` helper to materialize it over a two-column ``Float64`` frame; ``assert_matches`` and the
naive ``pnl_gross_reference`` oracle are shared across the suite. The PnL is degree-1 homogeneous in both ``quantity``
and ``price``, so it carries the scale-homogeneity and large-magnitude tiers.

The ladder is the canonical one: contract (type / shape / lazy-eager / ``.over`` per-group independence), edge
(warm-up / single-row / null / NaN / multiplier guard / multiplier scaling), correctness (vs the closed-form reference
and a frozen golden master), and properties (reference agreement incl. missing data, scale-homogeneity,
large-magnitude). Categories are split into classes; cross-cutting categories use markers (see ``tests/README.md``).
"""

import math
from collections.abc import Sequence

import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st
from tests.pnl.oracles import pnl_gross_reference
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
    subnormal_safe_floats,
)

from pomata.pnl import pnl_gross

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- derived, not chosen; the rationale is the shared method in CORRECTNESS.md, the numbers are this
# primitive's. pnl_gross has a one-bar price lag (W = 1 warm-up, M = 0): a case is a pair of equal-length series
# (quantity, price). It is degree-1 homogeneous in quantity and in price, so it keeps the scale-homogeneity and
# large-magnitude tiers. Repetitions N are the shared CI profile (tests/conftest.py).
# ----------------------------------------------------------------------------------------------------------------------
SERIES_MAX = 50
QUANTITY = "quantity"
PRICE = "price"
MULTIPLIER = 50.0  # the deterministic futures-multiplier test value

# Positive contract multipliers for the property tiers.
_MULTIPLIERS = st.floats(min_value=0.01, max_value=1000.0, allow_nan=False, allow_infinity=False)


@st.composite
def _cases[T](
    draw: st.DrawFn,
    quantities: st.SearchStrategy[T],
    prices: st.SearchStrategy[T],
    min_size: int = 1,
) -> tuple[list[T], list[T]]:
    """
    A pair of equal-length series (quantities, prices) sized from the facts above; one-bar warm-up, so the first row is
    always null and the rest are defined output.
    """
    length = draw(st.integers(min_value=min_size, max_value=SERIES_MAX))
    quantity = draw(st.lists(quantities, min_size=length, max_size=length))
    price = draw(st.lists(prices, min_size=length, max_size=length))
    return quantity, price


def apply_pnl_gross(
    quantity: Sequence[float | None],
    price: Sequence[float | None],
    multiplier: float = 1.0,
) -> list[float | None]:
    """
    Materialize ``pnl_gross`` over a two-column ``Float64`` frame from the aligned input lists.
    """
    return materialize(
        {QUANTITY: quantity, PRICE: price},
        pnl_gross(pl.col(QUANTITY), pl.col(PRICE), multiplier=multiplier),
    )


class TestPnlGrossContract:
    """
    Type, shape, and lazy/eager guarantees.
    """

    def test_over_partitions_independently(self) -> None:
        """
        Verifies that under ``.over`` the one-bar price change resets per group and never reaches across group
        boundaries (so the first row of each group is null).
        """
        frame = pl.DataFrame(
            {
                "ticker": ["a"] * 3 + ["b"] * 3,
                QUANTITY: pl.Series(QUANTITY, [10.0, 10.0, -5.0, 2.0, 2.0, 2.0], dtype=pl.Float64),
                PRICE: pl.Series(PRICE, [100.0, 102.0, 101.0, 50.0, 51.0, 49.0], dtype=pl.Float64),
            }
        )
        grouped = frame.select(pnl_gross(pl.col(QUANTITY), pl.col(PRICE)).over("ticker").alias("y"))["y"].to_list()
        group_a = apply_pnl_gross([10.0, 10.0, -5.0], [100.0, 102.0, 101.0])
        group_b = apply_pnl_gross([2.0, 2.0, 2.0], [50.0, 51.0, 49.0])
        assert_matches(grouped, group_a + group_b)


class TestPnlGrossEdge:
    """
    Boundaries, warm-up, null / NaN handling, and the multiplier guard.
    """

    def test_warmup_null_count(self) -> None:
        """
        Verifies the warm-up is exactly one row: the first PnL is null (no previous price), the second is defined.
        """
        result = apply_pnl_gross([10.0, 10.0, -5.0], [100.0, 102.0, 101.0])
        assert result[0] is None
        assert result[1] is not None

    def test_single_row(self) -> None:
        """
        Verifies that a one-row series is all warm-up (no previous price to difference against).
        """
        assert_matches(apply_pnl_gross([10.0], [100.0]), [None])

    def test_null_propagates(self) -> None:
        """
        Verifies that a ``null`` in the quantity or the price makes the affected rows ``null`` (matching the reference).
        """
        quantity = [10.0, None, -5.0, 20.0]
        price = [100.0, 102.0, 101.0, 104.0]
        assert_matches(apply_pnl_gross(quantity, price), pnl_gross_reference(quantity, price))

    def test_nan_propagates(self) -> None:
        """
        Verifies that a ``NaN`` propagates to the rows that reference it (matching the reference).
        """
        quantity = [10.0, 10.0, -5.0, 20.0]
        price = [100.0, 102.0, math.nan, 104.0]
        assert_matches(apply_pnl_gross(quantity, price), pnl_gross_reference(quantity, price))

    def test_null_takes_precedence_over_nan(self) -> None:
        """
        Verifies ``null`` takes precedence over ``NaN`` when the two meet at one output row: a ``null`` quantity against
        a ``NaN`` price -- and the reverse -- yields ``null``, not ``NaN`` (Polars treats ``null`` as missing, so it
        dominates), matching the reference.
        """
        null_quantity = [1.0, None, 2.0]
        nan_price = [100.0, math.nan, 110.0]
        result = apply_pnl_gross(null_quantity, nan_price)
        assert result[1] is None
        assert_matches(result, pnl_gross_reference(null_quantity, nan_price))

        nan_quantity = [1.0, math.nan, 2.0]
        null_price = [100.0, None, 110.0]
        result_reverse = apply_pnl_gross(nan_quantity, null_price)
        assert result_reverse[1] is None
        assert_matches(result_reverse, pnl_gross_reference(nan_quantity, null_price))

    def test_short_on_flat_price_is_negative_zero(self) -> None:
        """
        Verifies a short position over a flat price yields IEEE ``-0.0``, not ``+0.0``: ``Δprice`` is exactly ``0.0``
        and ``negative_quantity x 0.0`` carries the sign bit, matching the reference.
        The property tiers cannot see this, since ``-0.0 == 0.0``.
        """
        quantity = [-5.0, -5.0]
        price = [100.0, 100.0]
        result = apply_pnl_gross(quantity, price)
        value = result[1]
        assert value is not None
        assert value == 0.0
        assert math.copysign(1.0, value) == -1.0
        assert_matches(result, pnl_gross_reference(quantity, price))

    def test_multiplier_scales(self) -> None:
        """
        Verifies the contract multiplier scales the PnL linearly (a ``50`` multiplier is ``50x`` the unit multiplier).
        """
        quantity = [10.0, 10.0, -5.0, -5.0, 20.0]
        price = [100.0, 102.0, 101.0, 104.0, 103.0]
        unit = apply_pnl_gross(quantity, price, multiplier=1.0)
        scaled = apply_pnl_gross(quantity, price, multiplier=MULTIPLIER)
        for value_scaled, value_unit in zip(scaled, unit, strict=True):
            if value_unit is None:
                assert value_scaled is None
            else:
                assert value_scaled is not None
                assert math.isclose(value_scaled, value_unit * MULTIPLIER, rel_tol=RELATIVE_TOLERANCE_REFERENCE)

    def test_invalid_multiplier_raises(self) -> None:
        """
        Verifies that a multiplier that is not a finite number ``> 0`` (zero, negative, ``NaN``, or ``±inf``) raises
        ``ValueError`` -- a contract multiplier is a finite positive number, so a non-finite value fails fast at the
        call site rather than silently poisoning the output with ``NaN`` / ``inf``.
        """
        for invalid in (0.0, -5.0, math.nan, math.inf, -math.inf):
            with pytest.raises(ValueError, match="multiplier must be a finite number > 0"):
                pnl_gross(pl.col(QUANTITY), pl.col(PRICE), multiplier=invalid)


class TestPnlGrossCorrectness:
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
            apply_pnl_gross(quantity, price, multiplier=MULTIPLIER),
            pnl_gross_reference(quantity, price, MULTIPLIER),
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference at the unit multiplier and at a 50x futures multiplier.
        """
        quantity = [10.0, 10.0, -5.0, -5.0, 20.0]
        price = [100.0, 102.0, 101.0, 104.0, 103.0]
        unit = materialize({QUANTITY: quantity, PRICE: price}, pnl_gross(pl.col(QUANTITY), pl.col(PRICE)).round(4))
        assert_matches(unit, [None, 20.0, 5.0, -15.0, -20.0])
        futures = materialize(
            {QUANTITY: quantity, PRICE: price}, pnl_gross(pl.col(QUANTITY), pl.col(PRICE), multiplier=50.0).round(4)
        )
        assert_matches(futures, [None, 1000.0, 250.0, -750.0, -1000.0])


class TestPnlGrossProperties:
    """
    Invariants that must hold for all inputs (property-based).
    """

    @given(case=_cases(finite_floats(), finite_floats(), min_size=0), multiplier=_MULTIPLIERS)
    def test_matches_reference_for_any_input(
        self,
        case: tuple[list[float], list[float]],
        multiplier: float,
    ) -> None:
        """
        Verifies that, for any aligned input series and positive multiplier, the implementation matches the reference.
        """
        quantity, price = case
        assert_matches(
            apply_pnl_gross(quantity, price, multiplier=multiplier),
            pnl_gross_reference(quantity, price, multiplier),
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    @given(case=_cases(missing_data_floats(), missing_data_floats(), min_size=0), multiplier=_MULTIPLIERS)
    def test_matches_reference_under_missing_data(
        self,
        case: tuple[list[float | None], list[float | None]],
        multiplier: float,
    ) -> None:
        """
        Verifies that, for inputs freely mixing null / NaN / finite, the implementation matches the naive reference.
        """
        quantity, price = case
        assert_matches(
            apply_pnl_gross(quantity, price, multiplier=multiplier),
            pnl_gross_reference(quantity, price, multiplier),
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    @given(
        case=_cases(subnormal_safe_floats(), subnormal_safe_floats()),
        exponent=st.sampled_from([-4, -3, -2, -1, 1, 2, 3, 4]),
    )
    def test_scale_homogeneity_in_quantity(
        self,
        case: tuple[list[float], list[float]],
        exponent: int,
    ) -> None:
        """
        Verifies degree-1 homogeneity in the quantity: scaling the quantity by a constant scales the PnL by the same
        constant (the price held fixed). ``k`` is a power of two so the rescaling is lossless.
        """
        k = 2.0**exponent
        quantity, price = case
        result_base = apply_pnl_gross(quantity, price)
        result_scaled = apply_pnl_gross([value * k for value in quantity], price)
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
            apply_pnl_gross(quantity, price),
            pnl_gross_reference(quantity, price),
            rel_tol=RELATIVE_TOLERANCE_SCALE,
            abs_tol=ABSOLUTE_TOLERANCE_STREAMING,
        )
