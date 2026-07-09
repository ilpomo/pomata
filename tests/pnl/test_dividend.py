"""
Tests for ``pomata.pnl.dividend`` — the per-bar dividend cashflow of a held quantity.

``dividend`` is two-input (``quantity`` / ``dividend_per_share``) and elementwise (a pure per-row product, no window, no
lag). Tests use a local ``apply_dividend`` helper to materialize it over a two-column ``Float64`` frame;
``assert_matches`` and the naive ``dividend_reference`` oracle are shared across the suite. The product is degree-1
homogeneous in each input, so it carries the scale-homogeneity and large-magnitude tiers.

The ladder, adapted to an elementwise two-input product: contract (type / shape / lazy-eager / ``.over`` identity), edge
(empty / single-row / null / null-precedence / NaN), correctness (closed-form reference + frozen golden master), and
properties (reference agreement incl. missing data, scale-homogeneity, large-magnitude). Categories are split into
classes; cross-cutting categories use markers (see ``tests/README.md``).
"""

import math
from collections.abc import Sequence

import polars as pl
from hypothesis import given
from hypothesis import strategies as st
from tests.pnl.oracles import dividend_reference
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

from pomata.pnl import dividend

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- dividend is a windowless elementwise product of two aligned series (W = 0, M = 0): a case is a pair of
# equal-length series. Degree-1 homogeneous in each input, so it keeps the scale-homogeneity and large-magnitude tiers.
# Repetitions N are the shared CI profile (tests/conftest.py).
# ----------------------------------------------------------------------------------------------------------------------
SERIES_MAX = 50
QUANTITY = "quantity"
DIVIDEND_PER_SHARE = "dividend_per_share"


@st.composite
def _cases[T](
    draw: st.DrawFn,
    quantities: st.SearchStrategy[T],
    dividends: st.SearchStrategy[T],
    min_size: int = 1,
) -> tuple[list[T], list[T]]:
    """
    A pair of equal-length series (quantities, dividends per share); windowless, so every row is a defined output.
    """
    length = draw(st.integers(min_value=min_size, max_value=SERIES_MAX))
    quantity = draw(st.lists(quantities, min_size=length, max_size=length))
    dividend_per_share = draw(st.lists(dividends, min_size=length, max_size=length))
    return quantity, dividend_per_share


def apply_dividend(
    quantity: Sequence[float | None],
    dividend_per_share: Sequence[float | None],
) -> list[float | None]:
    """
    Materialize ``dividend`` over a two-column ``Float64`` frame from the aligned input lists.
    """
    return materialize(
        {QUANTITY: quantity, DIVIDEND_PER_SHARE: dividend_per_share},
        dividend(pl.col(QUANTITY), pl.col(DIVIDEND_PER_SHARE)),
    )


class TestDividendContract:
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
                QUANTITY: pl.Series(QUANTITY, [100.0, 100.0, -50.0, 30.0, 30.0, 30.0], dtype=pl.Float64),
                DIVIDEND_PER_SHARE: pl.Series(DIVIDEND_PER_SHARE, [0.0, 0.5, 0.5, 0.0, 0.2, 0.0], dtype=pl.Float64),
            }
        )
        plain = frame.select(dividend(pl.col(QUANTITY), pl.col(DIVIDEND_PER_SHARE)).alias("y"))["y"].to_list()
        grouped = frame.select(dividend(pl.col(QUANTITY), pl.col(DIVIDEND_PER_SHARE)).over("ticker").alias("y"))[
            "y"
        ].to_list()
        assert_matches(plain, grouped)


class TestDividendEdge:
    """
    Boundaries and null / NaN handling.
    """

    def test_single_row(self) -> None:
        """
        Verifies that a one-row series resolves to the single product.
        """
        assert_matches(apply_dividend([100.0], [0.5]), [50.0])

    def test_null_propagates(self) -> None:
        """
        Verifies that a ``null`` in either input makes that row ``null`` (matching the naive reference).
        """
        quantity = [100.0, None, 100.0, -50.0]
        dividend_per_share = [0.0, 0.5, 0.5, 0.5]
        assert_matches(apply_dividend(quantity, dividend_per_share), dividend_reference(quantity, dividend_per_share))

    def test_null_takes_precedence_over_nan(self) -> None:
        """
        Verifies that a row with a ``null`` in one input and a ``NaN`` in the other yields ``null``.
        """
        assert_matches(apply_dividend([None, 100.0], [math.nan, 0.5]), [None, 50.0])

    def test_nan_propagates(self) -> None:
        """
        Verifies that a ``NaN`` in either input makes that row ``NaN`` (matching the naive reference).
        """
        quantity = [100.0, 100.0, math.nan, -50.0]
        dividend_per_share = [0.0, 0.5, 0.5, 0.5]
        assert_matches(apply_dividend(quantity, dividend_per_share), dividend_reference(quantity, dividend_per_share))


class TestDividendCorrectness:
    """
    Against the naive reference oracle and frozen golden-master values.
    """

    def test_matches_reference(self) -> None:
        """
        Verifies agreement with the naive closed-form reference over a representative series.
        """
        quantity = [100.0, 100.0, 100.0, 0.0, -50.0, 80.0, 80.0, -30.0]
        dividend_per_share = [0.0, 0.0, 0.5, 0.0, 0.5, 0.0, 0.25, 0.25]
        assert_matches(
            apply_dividend(quantity, dividend_per_share),
            dividend_reference(quantity, dividend_per_share),
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference over a five-bar series (long receives, short pays).
        """
        result = materialize(
            {
                QUANTITY: [100.0, 100.0, 100.0, 0.0, -50.0],
                DIVIDEND_PER_SHARE: [0.0, 0.0, 0.5, 0.0, 0.5],
            },
            dividend(pl.col(QUANTITY), pl.col(DIVIDEND_PER_SHARE)).round(4),
        )
        assert_matches(result, [0.0, 0.0, 50.0, 0.0, -25.0])


class TestDividendProperties:
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
        quantity, dividend_per_share = case
        assert_matches(
            apply_dividend(quantity, dividend_per_share),
            dividend_reference(quantity, dividend_per_share),
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
        quantity, dividend_per_share = case
        assert_matches(
            apply_dividend(quantity, dividend_per_share),
            dividend_reference(quantity, dividend_per_share),
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
        Verifies that ``dividend`` is homogeneous of degree 1 in the quantity: scaling the quantity by a constant
        ``k``, with the other inputs untouched, scales the output by the same ``k``. ``k`` is a power of two, so the
        rescale is exact and adds no floating-point error.
        """
        k = 2.0**exponent
        quantity, dividend_per_share = case
        result_base = apply_dividend(quantity, dividend_per_share)
        result_scaled = apply_dividend([value * k for value in quantity], dividend_per_share)
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
        quantity_base, dividend_base = case
        quantity = [value * scale for value in quantity_base]
        dividend_per_share = [value * scale for value in dividend_base]
        assert_matches(
            apply_dividend(quantity, dividend_per_share),
            dividend_reference(quantity, dividend_per_share),
            rel_tol=RELATIVE_TOLERANCE_SCALE,
            abs_tol=ABSOLUTE_TOLERANCE_STREAMING,
        )
