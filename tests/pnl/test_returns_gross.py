"""
Tests for ``pomata.pnl.returns_gross`` — Gross Strategy Returns.

``returns_gross`` is two-input (``weight`` / ``asset_returns``) and elementwise (a pure per-row product, no window, no
lag). Tests use a local ``apply_returns_gross`` helper to materialize it over a two-column ``Float64`` frame;
``assert_matches`` and the naive ``returns_gross_reference`` oracle are shared across the suite. The product is degree-1
homogeneous in each input, so it carries the scale-homogeneity and large-magnitude tiers (unlike a scale-invariant
ratio).

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
from tests.pnl.oracles import returns_gross_reference
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

from pomata.pnl import returns_gross

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- derived, not chosen; the rationale is the shared method in CORRECTNESS.md, the numbers are this
# primitive's. returns_gross is a windowless elementwise product of two aligned series (W = 0, M = 0): a case is a pair
# of equal-length series, every row a defined output. It is degree-1 homogeneous in each input (so it keeps the
# scale-homogeneity and large-magnitude tiers, unlike a scale-invariant ratio). Repetitions N are the shared CI profile
# (tests/conftest.py); override per-test only if its parameter space is larger.
# ----------------------------------------------------------------------------------------------------------------------
SERIES_MAX = 50
WEIGHT = "weight"
ASSET_RETURNS = "asset_returns"


@st.composite
def _cases[T](
    draw: st.DrawFn,
    weights: st.SearchStrategy[T],
    asset_returns: st.SearchStrategy[T],
    min_size: int = 1,
) -> tuple[list[T], list[T]]:
    """
    A pair of equal-length series (weights, asset_returns) sized from the facts above. returns_gross is windowless, so
    a case is just the aligned pair: every row is a defined output.
    """
    length = draw(st.integers(min_value=min_size, max_value=SERIES_MAX))
    weight = draw(st.lists(weights, min_size=length, max_size=length))
    asset_return = draw(st.lists(asset_returns, min_size=length, max_size=length))
    return weight, asset_return


def apply_returns_gross(
    weight: Sequence[float | None],
    asset_returns: Sequence[float | None],
) -> list[float | None]:
    """
    Materialize ``returns_gross`` over a two-column ``Float64`` frame from the aligned input lists.
    """
    return materialize(
        {WEIGHT: weight, ASSET_RETURNS: asset_returns},
        returns_gross(pl.col(WEIGHT), pl.col(ASSET_RETURNS)),
    )


class TestReturnsGrossContract:
    """
    Type, shape, and lazy/eager guarantees.
    """

    def test_over_is_identity(self) -> None:
        """
        Verifies that ``.over`` is optional for this elementwise product: partitioning by group is identical to the
        un-partitioned call (no cross-bar state can leak across group boundaries).
        """
        frame = pl.DataFrame(
            {
                "ticker": ["a", "a", "a", "b", "b", "b"],
                WEIGHT: pl.Series(WEIGHT, [1.0, -1.0, 0.5, 0.5, 0.5, -1.0], dtype=pl.Float64),
                ASSET_RETURNS: pl.Series(ASSET_RETURNS, [0.02, 0.03, -0.01, 0.04, -0.02, 0.01], dtype=pl.Float64),
            }
        )
        plain = frame.select(returns_gross(pl.col(WEIGHT), pl.col(ASSET_RETURNS)).alias("y"))["y"].to_list()
        grouped = frame.select(returns_gross(pl.col(WEIGHT), pl.col(ASSET_RETURNS)).over("ticker").alias("y"))[
            "y"
        ].to_list()
        assert_matches(plain, grouped)


class TestReturnsGrossEdge:
    """
    Boundaries and null / NaN handling.
    """

    def test_single_row(self) -> None:
        """
        Verifies that a one-row series resolves to the single product (no window, no warm-up).
        """
        assert_matches(apply_returns_gross([0.5], [0.04]), [0.02])

    def test_null_propagates(self) -> None:
        """
        Verifies that a ``null`` in either input makes that row ``null`` (matching the naive reference).
        """
        weight = [1.0, None, -1.0, 0.5]
        asset_returns = [0.02, 0.03, 0.01, 0.04]
        assert_matches(apply_returns_gross(weight, asset_returns), returns_gross_reference(weight, asset_returns))

    def test_null_takes_precedence_over_nan(self) -> None:
        """
        Verifies that a row with a ``null`` in one input and a ``NaN`` in the other yields ``null`` — ``null`` takes
        precedence over ``NaN``.
        """
        assert_matches(apply_returns_gross([None, 0.5], [math.nan, 0.04]), [None, 0.02])

    def test_nan_propagates(self) -> None:
        """
        Verifies that a ``NaN`` in either input makes that row ``NaN`` (matching the naive reference).
        """
        weight = [1.0, 0.5, -1.0, 0.5]
        asset_returns = [0.02, math.nan, 0.01, 0.04]
        assert_matches(apply_returns_gross(weight, asset_returns), returns_gross_reference(weight, asset_returns))


class TestReturnsGrossCorrectness:
    """
    Against the naive reference oracle and frozen golden-master values.
    """

    def test_matches_reference(self) -> None:
        """
        Verifies agreement with the naive closed-form reference over a representative series.
        """
        weight = [1.0, 0.5, -1.0, -1.0, 0.5, 2.0, -0.5, 1.5]
        asset_returns = [0.02, -0.01, 0.03, -0.02, 0.04, -0.03, 0.01, 0.05]
        assert_matches(
            apply_returns_gross(weight, asset_returns),
            returns_gross_reference(weight, asset_returns),
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference over a five-bar series.
        """
        result = materialize(
            {
                WEIGHT: [1.0, 0.5, -1.0, -1.0, 0.5],
                ASSET_RETURNS: [0.02, -0.01, 0.03, -0.02, 0.04],
            },
            returns_gross(pl.col(WEIGHT), pl.col(ASSET_RETURNS)).round(4),
        )
        assert_matches(result, [0.02, -0.005, -0.03, 0.02, 0.02])


class TestReturnsGrossProperties:
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
        weight, asset_returns = case
        assert_matches(
            apply_returns_gross(weight, asset_returns),
            returns_gross_reference(weight, asset_returns),
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
        weight, asset_returns = case
        assert_matches(
            apply_returns_gross(weight, asset_returns),
            returns_gross_reference(weight, asset_returns),
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    @given(
        case=_cases(subnormal_safe_floats(), subnormal_safe_floats()),
        exponent=st.sampled_from([-4, -3, -2, -1, 1, 2, 3, 4]),
    )
    def test_scale_homogeneity_in_weight(
        self,
        case: tuple[list[float], list[float]],
        exponent: int,
    ) -> None:
        """
        Verifies that ``returns_gross`` is homogeneous of degree 1 in the weight: scaling the weight by a constant
        ``k``, with the other inputs untouched, scales the output by the same ``k``. ``k`` is a power of two, so the
        rescale is exact and adds no floating-point error.
        """
        k = 2.0**exponent
        weight, asset_returns = case
        result_base = apply_returns_gross(weight, asset_returns)
        result_scaled = apply_returns_gross([value * k for value in weight], asset_returns)
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
        weight_base, asset_returns_base = case
        weight = [value * scale for value in weight_base]
        asset_returns = [value * scale for value in asset_returns_base]
        assert_matches(
            apply_returns_gross(weight, asset_returns),
            returns_gross_reference(weight, asset_returns),
            rel_tol=RELATIVE_TOLERANCE_SCALE,
            abs_tol=ABSOLUTE_TOLERANCE_STREAMING,
        )
