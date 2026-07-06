"""
Tests for ``pomata.pnl.pnl_gross_inverse`` — Gross Inverse-Contract (coin-margined) PnL.

``pnl_gross_inverse`` is two-input (``quantity`` / ``price``) plus a scalar ``multiplier``, with a one-bar price lag, so
tests use a local ``apply_pnl_gross_inverse`` helper to materialize it over a two-column ``Float64`` frame;
``assert_matches`` and the naive ``pnl_gross_inverse_reference`` oracle are shared across the suite. The payoff is the
one-bar change in the price reciprocal, so it is degree-1 homogeneous in ``quantity`` (it carries the scale-homogeneity
and large-magnitude tiers there) but degree-(-1) homogeneous in ``price`` — scaling the price series by ``k`` scales the
coin PnL by ``1 / k`` — which stands in as the distinctive nonlinear metamorphic. It is defined on strictly positive
prices; the IEEE-754 reciprocal boundaries are pinned deterministically in the edge tier.

The ladder is the canonical one: contract (type / shape / lazy-eager / ``.over`` per-group independence), edge
(warm-up / single-row / null / NaN / multiplier guard / multiplier scaling / domain boundaries), correctness (vs the
closed-form reference and a frozen golden master), and properties (reference agreement incl. missing data,
scale-homogeneity in quantity, inverse price scaling, large-magnitude). Categories are split into classes; cross-cutting
categories use markers (see ``tests/README.md``).
"""

import math
from collections.abc import Sequence

import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st
from tests.pnl.oracles import pnl_gross_inverse_reference
from tests.support import (
    ABSOLUTE_TOLERANCE_REFERENCE,
    ABSOLUTE_TOLERANCE_SCALE,
    ABSOLUTE_TOLERANCE_STREAMING,
    RELATIVE_TOLERANCE_PROPERTY,
    RELATIVE_TOLERANCE_REFERENCE,
    RELATIVE_TOLERANCE_SCALE,
    assert_matches,
    assert_scale_homogeneous,
    finite_floats,
    materialize,
    missing_data_floats,
    positive_missing_data,
    subnormal_safe_floats,
)

from pomata.pnl import pnl_gross_inverse

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- derived, not chosen; the rationale is the shared method in CORRECTNESS.md, the numbers are this
# primitive's. pnl_gross_inverse has a one-bar price lag (W = 1 warm-up, M = 0): a case is a pair of equal-length series
# (quantity, price). It is degree-1 homogeneous in quantity (so it keeps the scale-homogeneity and large-magnitude tiers
# there) and degree-(-1) homogeneous in price (the inverse-price-scaling metamorphic). The payoff is defined on strictly
# positive prices ([1.0, PRICE_MAX]); the IEEE reciprocal boundaries are pinned in the edge tier. Repetitions N are the
# shared CI profile (tests/conftest.py).
# ----------------------------------------------------------------------------------------------------------------------
SERIES_MAX = 50
PRICE_MAX = 1e4
QUANTITY = "quantity"
PRICE = "price"
MULTIPLIER = 100.0  # the deterministic inverse-contract notional test value

# Positive contract notionals and a positive-price element strategy for the property tiers.
_MULTIPLIERS = st.floats(min_value=0.01, max_value=1000.0, allow_nan=False, allow_infinity=False)
_POSITIVE_PRICES = st.floats(min_value=1.0, max_value=PRICE_MAX, allow_nan=False, allow_infinity=False)


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


def apply_pnl_gross_inverse(
    quantity: Sequence[float | None],
    price: Sequence[float | None],
    multiplier: float = 1.0,
) -> list[float | None]:
    """
    Materialize ``pnl_gross_inverse`` over a two-column ``Float64`` frame from the aligned input lists.
    """
    return materialize(
        {QUANTITY: quantity, PRICE: price},
        pnl_gross_inverse(pl.col(QUANTITY), pl.col(PRICE), multiplier=multiplier),
    )


class TestPnlGrossInverseContract:
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
                QUANTITY: pl.Series(QUANTITY, [1.0, 1.0, -2.0, 2.0, 2.0, 2.0], dtype=pl.Float64),
                PRICE: pl.Series(PRICE, [100.0, 110.0, 105.0, 50.0, 55.0, 52.0], dtype=pl.Float64),
            }
        )
        grouped = frame.select(pnl_gross_inverse(pl.col(QUANTITY), pl.col(PRICE)).over("ticker").alias("y"))[
            "y"
        ].to_list()
        group_a = apply_pnl_gross_inverse([1.0, 1.0, -2.0], [100.0, 110.0, 105.0])
        group_b = apply_pnl_gross_inverse([2.0, 2.0, 2.0], [50.0, 55.0, 52.0])
        assert_matches(grouped, group_a + group_b)


class TestPnlGrossInverseEdge:
    """
    Boundaries, warm-up, null / NaN handling, the multiplier guard, and the positive-price domain.
    """

    def test_warmup_null_count(self) -> None:
        """
        Verifies the warm-up is exactly one row: the first PnL is null (no previous price), the second is defined.
        """
        result = apply_pnl_gross_inverse([1.0, 1.0, -2.0], [100.0, 110.0, 105.0])
        assert result[0] is None
        assert result[1] is not None

    def test_single_row(self) -> None:
        """
        Verifies that a one-row series is all warm-up (no previous price to difference against).
        """
        assert_matches(apply_pnl_gross_inverse([1.0], [100.0]), [None])

    def test_null_propagates(self) -> None:
        """
        Verifies that a ``null`` in the quantity or the price makes the affected rows ``null`` (matching the reference).
        """
        quantity = [1.0, None, -2.0, 3.0]
        price = [100.0, 110.0, 105.0, 120.0]
        assert_matches(apply_pnl_gross_inverse(quantity, price), pnl_gross_inverse_reference(quantity, price))

    def test_nan_propagates(self) -> None:
        """
        Verifies that a ``NaN`` propagates to the rows that reference it (matching the reference).
        """
        quantity = [1.0, 1.0, -2.0, 3.0]
        price = [100.0, 110.0, math.nan, 120.0]
        assert_matches(apply_pnl_gross_inverse(quantity, price), pnl_gross_inverse_reference(quantity, price))

    def test_null_takes_precedence_over_nan(self) -> None:
        """
        Verifies ``null`` takes precedence over ``NaN`` when the two meet at one output row: a ``null`` quantity against
        a ``NaN`` price -- and the reverse -- yields ``null``, not ``NaN`` (Polars treats ``null`` as missing, so it
        dominates), matching the reference.
        """
        null_quantity = [1.0, None, 2.0]
        nan_price = [100.0, math.nan, 110.0]
        result = apply_pnl_gross_inverse(null_quantity, nan_price)
        assert result[1] is None
        assert_matches(result, pnl_gross_inverse_reference(null_quantity, nan_price))

        nan_quantity = [1.0, math.nan, 2.0]
        null_price = [100.0, None, 110.0]
        result_reverse = apply_pnl_gross_inverse(nan_quantity, null_price)
        assert result_reverse[1] is None
        assert_matches(result_reverse, pnl_gross_inverse_reference(nan_quantity, null_price))

    def test_short_on_flat_price_is_negative_zero(self) -> None:
        """
        Verifies a short position over a flat price yields IEEE ``-0.0``, not ``+0.0``: the reciprocal change is exactly
        ``0.0`` and ``negative_quantity x 0.0`` carries the sign bit, matching the reference. The property tiers cannot
        see this, since ``-0.0 == 0.0``.
        """
        quantity = [-5.0, -5.0]
        price = [100.0, 100.0]
        result = apply_pnl_gross_inverse(quantity, price)
        value = result[1]
        assert value is not None
        assert value == 0.0
        assert math.copysign(1.0, value) == -1.0
        assert_matches(result, pnl_gross_inverse_reference(quantity, price))

    def test_multiplier_scales(self) -> None:
        """
        Verifies the contract notional scales the PnL linearly (a ``100`` notional is ``100x`` the unit notional).
        """
        quantity = [1.0, 1.0, -2.0, -2.0, 3.0]
        price = [100.0, 110.0, 105.0, 120.0, 115.0]
        unit = apply_pnl_gross_inverse(quantity, price, multiplier=1.0)
        scaled = apply_pnl_gross_inverse(quantity, price, multiplier=MULTIPLIER)
        for value_scaled, value_unit in zip(scaled, unit, strict=True):
            if value_unit is None:
                assert value_scaled is None
            else:
                assert value_scaled is not None
                assert math.isclose(value_scaled, value_unit * MULTIPLIER, rel_tol=RELATIVE_TOLERANCE_REFERENCE)

    def test_invalid_multiplier_raises(self) -> None:
        """
        Verifies that a multiplier that is not a finite number ``> 0`` (zero, negative, ``NaN``, or ``±inf``) raises
        ``ValueError`` -- a contract notional is a finite positive number, so a non-finite value fails fast at the call
        site rather than silently poisoning the output with ``NaN`` / ``inf``.
        """
        for invalid in (0.0, -5.0, math.nan, math.inf, -math.inf):
            with pytest.raises(ValueError, match="multiplier must be a finite number > 0"):
                pnl_gross_inverse(pl.col(QUANTITY), pl.col(PRICE), multiplier=invalid)

    def test_domain_boundaries(self) -> None:
        """
        Verifies the IEEE-754 reciprocal boundaries reproduced from Polars: a zero current price makes ``1 / P_t``
        infinite so a long bar is ``-inf``, a zero previous price makes ``1 / P_{t-1}`` infinite so the next bar is
        ``+inf``, and a negative price gives a finite but economically meaningless value.
        """
        assert_matches(
            apply_pnl_gross_inverse([1.0, 1.0, 1.0, 1.0], [100.0, 0.0, 50.0, -50.0]),
            [None, -math.inf, math.inf, 0.04],
        )


class TestPnlGrossInverseCorrectness:
    """
    Against the naive reference oracle and frozen golden-master values.
    """

    def test_matches_reference(self) -> None:
        """
        Verifies agreement with the naive closed-form reference over a representative positive-price series.
        """
        quantity = [1.0, 1.0, -2.0, -2.0, 3.0, 3.0, -1.0, 2.0]
        price = [100.0, 110.0, 105.0, 120.0, 115.0, 118.0, 112.0, 120.0]
        assert_matches(
            apply_pnl_gross_inverse(quantity, price, multiplier=MULTIPLIER),
            pnl_gross_inverse_reference(quantity, price, MULTIPLIER),
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference at the unit notional and at a ``100`` inverse-contract notional.
        """
        quantity = [1.0, 1.0, -2.0, -2.0, 3.0]
        price = [100.0, 110.0, 105.0, 120.0, 115.0]
        unit = materialize(
            {QUANTITY: quantity, PRICE: price}, pnl_gross_inverse(pl.col(QUANTITY), pl.col(PRICE)).round(6)
        )
        assert_matches(unit, [None, 0.000909, 0.000866, -0.002381, -0.001087])
        notional = materialize(
            {QUANTITY: quantity, PRICE: price},
            pnl_gross_inverse(pl.col(QUANTITY), pl.col(PRICE), multiplier=100.0).round(4),
        )
        assert_matches(notional, [None, 0.0909, 0.0866, -0.2381, -0.1087])


class TestPnlGrossInverseProperties:
    """
    Invariants that must hold for all inputs (property-based).
    """

    @given(case=_cases(finite_floats(), _POSITIVE_PRICES, min_size=0), multiplier=_MULTIPLIERS)
    def test_matches_reference_for_any_input(
        self,
        case: tuple[list[float], list[float]],
        multiplier: float,
    ) -> None:
        """
        Verifies that, for any quantity series over positive prices and a positive notional, the implementation matches
        the reference.
        """
        quantity, price = case
        assert_matches(
            apply_pnl_gross_inverse(quantity, price, multiplier=multiplier),
            pnl_gross_inverse_reference(quantity, price, multiplier),
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    @given(
        case=_cases(missing_data_floats(), positive_missing_data(PRICE_MAX), min_size=0),
        multiplier=_MULTIPLIERS,
    )
    def test_matches_reference_under_missing_data(
        self,
        case: tuple[list[float | None], list[float | None]],
        multiplier: float,
    ) -> None:
        """
        Verifies that, for inputs freely mixing null / NaN / finite (positive prices), the implementation matches the
        naive reference.
        """
        quantity, price = case
        assert_matches(
            apply_pnl_gross_inverse(quantity, price, multiplier=multiplier),
            pnl_gross_inverse_reference(quantity, price, multiplier),
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    @given(
        case=_cases(subnormal_safe_floats(), _POSITIVE_PRICES),
        exponent=st.sampled_from([-4, -3, -2, -1, 1, 2, 3, 4]),
    )
    def test_scale_homogeneity_in_quantity(
        self,
        case: tuple[list[float], list[float]],
        exponent: int,
    ) -> None:
        """
        Verifies that ``pnl_gross_inverse`` is homogeneous of degree 1 in the quantity: scaling the quantity by a
        constant ``k``, with the other inputs untouched, scales the output by the same ``k``. ``k`` is a power of
        two, so the rescale is exact and adds no floating-point error.
        """
        k = 2.0**exponent
        quantity, price = case
        result_base = apply_pnl_gross_inverse(quantity, price)
        result_scaled = apply_pnl_gross_inverse([value * k for value in quantity], price)
        assert_scale_homogeneous(result_scaled, result_base, k=k, degree=1)

    @given(
        case=_cases(finite_floats(), _POSITIVE_PRICES),
        exponent=st.sampled_from([-4, -3, -2, -1, 1, 2, 3, 4]),
    )
    def test_inverse_price_scaling(
        self,
        case: tuple[list[float], list[float]],
        exponent: int,
    ) -> None:
        """
        Verifies degree-(-1) homogeneity in the price: scaling the whole price series by a constant scales the coin PnL
        by the reciprocal of that constant (``pnl(q, k * P) == pnl(q, P) / k``), because the payoff is built from the
        price reciprocal. ``k`` is a power of two so the rescaling is lossless.
        """
        k = 2.0**exponent
        quantity, price = case
        result_base = apply_pnl_gross_inverse(quantity, price)
        result_scaled = apply_pnl_gross_inverse(quantity, [value * k for value in price])
        for value_scaled, value_base in zip(result_scaled, result_base, strict=True):
            if value_base is None:
                assert value_scaled is None
            else:
                assert value_scaled is not None
                assert math.isclose(
                    value_scaled, value_base / k, rel_tol=RELATIVE_TOLERANCE_SCALE, abs_tol=ABSOLUTE_TOLERANCE_SCALE
                )

    @given(case=_cases(finite_floats(), _POSITIVE_PRICES), scale=st.sampled_from([1e-6, 1e6, 1e9]))
    def test_matches_reference_at_large_magnitude(
        self,
        case: tuple[list[float], list[float]],
        scale: float,
    ) -> None:
        """
        Verifies that at extreme quantities the implementation stays finite where the reference is and agrees (the price
        held in its positive range, the magnitude carried by the quantity).
        """
        quantity_base, price = case
        quantity = [value * scale for value in quantity_base]
        assert_matches(
            apply_pnl_gross_inverse(quantity, price),
            pnl_gross_inverse_reference(quantity, price),
            rel_tol=RELATIVE_TOLERANCE_SCALE,
            abs_tol=ABSOLUTE_TOLERANCE_STREAMING,
        )
