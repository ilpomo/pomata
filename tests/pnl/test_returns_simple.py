"""
Tests for ``pomata.pnl.returns_simple`` — Simple (arithmetic) Returns.

``returns_simple`` is single-input and a fixed one-bar-lag transform (a warm-up of one row, no recursion), so tests use
the shared ``apply_expr`` helper to materialize the factory over a one-column ``Float64`` frame; ``assert_matches`` and
the naive ``returns_simple_reference`` oracle are shared across the suite. The return is a ratio, so it is
scale-INVARIANT (``returns_simple(k * P) == returns_simple(P)``): it carries a scale-invariance property in place of the
scale-homogeneity / large-magnitude tests, which are vacuous when the input scale cancels.

The ladder is the canonical one: contract (type / shape / lazy-eager / ``.over`` per-group independence), edge
(single-row / null / NaN / warm-up / zero previous price), correctness (vs the closed-form reference and a frozen
golden master), and properties (reference agreement incl. missing data, scale-invariance). Categories are split into
classes; cross-cutting categories use markers (see ``tests/README.md``).
"""

import math

import polars as pl
from hypothesis import given
from hypothesis import strategies as st
from tests.pnl.oracles import returns_simple_reference
from tests.support import (
    ABSOLUTE_TOLERANCE_REFERENCE,
    COLUMN_X,
    GROUP_KEY,
    RELATIVE_TOLERANCE_PROPERTY,
    RELATIVE_TOLERANCE_REFERENCE,
    apply_expr,
    assert_matches,
    positive_missing_data,
)

from pomata.pnl import returns_simple

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- derived, not chosen; the rationale is the shared method in CORRECTNESS.md, the numbers are this
# primitive's. To add a primitive, set its facts here; the property tier below is then the same shape as every other's.
#   1. warmup  W = 1   (the one-bar lag ``expr.shift(1)`` is undefined on row 0, so the first row is always null)
#   2. memory  the oracle is a per-row two-endpoint transform like pomata, so the property holds from the first defined
#              row (M = 0); a case is simply a series of prices -- every row past the first is a defined output
#   3. domain  positive prices in ``[1.0, PRICE_MAX]`` -- the realistic price regime shared with returns_log; the
#              degenerate zero / negative previous price is pinned deterministically in the edge tier
# returns_simple has no window parameter, so ``_cases`` draws only the series (no window to couple). The return is a
# ratio (scale-invariant, O(1) around zero), so there is no scale-homogeneity or large-magnitude VALUE test (the scale
# cancels); a scale-invariance metamorphic stands in its place. Repetitions N are the shared CI profile
# (tests/conftest.py); override per-test only if its parameter space is larger.
# ----------------------------------------------------------------------------------------------------------------------
SERIES_MAX = 50
PRICE_MAX = 1e4

# The shared positive-price element strategy for the property tiers (the realistic price regime, away from zero).
_POSITIVE_PRICES = st.floats(min_value=1.0, max_value=PRICE_MAX, allow_nan=False, allow_infinity=False)


@st.composite
def _cases[T](draw: st.DrawFn, prices: st.SearchStrategy[T], min_size: int = 2) -> list[T]:
    """
    A price series sized from the facts above. returns_simple is windowless with a one-row warm-up, so -- unlike the
    windowed indicators' ``(series, window)`` pair -- a case is just the series; ``min_size`` defaults to two so at
    least one defined return is produced.
    """
    return draw(st.lists(prices, min_size=min_size, max_size=SERIES_MAX))


class TestReturnsSimpleContract:
    """
    Type, shape, and lazy/eager guarantees.
    """

    def test_over_partitions_independently(self) -> None:
        """
        Verifies that under ``.over`` the one-bar lag resets per group and never reaches across group boundaries (so the
        first row of each group is null).
        """
        frame = pl.DataFrame(
            {GROUP_KEY: ["a"] * 4 + ["b"] * 4, COLUMN_X: [100.0, 105.0, 102.0, 108.0, 50.0, 52.0, 51.0, 55.0]}
        )
        expr = returns_simple(pl.col(COLUMN_X)).over(GROUP_KEY)
        grouped = frame.select(expr.alias("y"))["y"].to_list()
        group_a = apply_expr([100.0, 105.0, 102.0, 108.0], returns_simple(pl.col(COLUMN_X)))
        group_b = apply_expr([50.0, 52.0, 51.0, 55.0], returns_simple(pl.col(COLUMN_X)))
        assert_matches(grouped, group_a + group_b)


class TestReturnsSimpleEdge:
    """
    Boundaries, warm-up, and null / NaN handling.
    """

    def test_single_row(self) -> None:
        """
        Verifies that a one-element series is all warm-up (no previous price to difference against).
        """
        assert_matches(apply_expr([42.0], returns_simple(pl.col(COLUMN_X))), [None])

    def test_null_propagates(self) -> None:
        """
        Verifies that a null at the current or previous row yields null there (matching the naive reference).
        """
        values = [100.0, 105.0, None, 108.0, 110.0]
        assert_matches(
            apply_expr(values, returns_simple(pl.col(COLUMN_X))),
            returns_simple_reference(values),
        )

    def test_null_takes_precedence_over_nan(self) -> None:
        """
        Verifies that the return row where a ``NaN`` price meets the previous row's ``null`` yields ``null``
        (``null`` takes precedence over ``NaN``), while the next return off the ``NaN`` is ``NaN``.
        """
        values = [100.0, None, math.nan, 108.0]
        assert_matches(apply_expr(values, returns_simple(pl.col(COLUMN_X))), [None, None, None, math.nan])

    def test_nan_propagates(self) -> None:
        """
        Verifies that a NaN propagates to the positions that reference it (matching the naive reference).
        """
        values = [100.0, 105.0, math.nan, 108.0, 110.0]
        assert_matches(
            apply_expr(values, returns_simple(pl.col(COLUMN_X))),
            returns_simple_reference(values),
        )

    def test_warmup_null_count(self) -> None:
        """
        Verifies the warm-up is exactly one row: the first return is null, the second is defined.
        """
        result = apply_expr([100.0, 105.0, 102.0], returns_simple(pl.col(COLUMN_X)))
        assert result[0] is None
        assert result[1] is not None

    def test_zero_previous_price(self) -> None:
        """
        Verifies the IEEE-754 division boundaries: a non-zero change over a zero previous price is ``+/-inf`` and a zero
        change over zero (``0 / 0``) is ``NaN``; a finite change over the zero is reported as usual.
        """
        assert_matches(
            apply_expr([0.0, 10.0, 0.0, 0.0], returns_simple(pl.col(COLUMN_X))),
            [None, math.inf, -1.0, math.nan],
        )

    def test_negative_zero_previous_price(self) -> None:
        """
        Verifies the signed-zero sign branch: a ``-0.0`` previous price flips the sign of the infinite return relative
        to a ``+0.0`` one (a positive change over ``-0.0`` is ``-inf``, a negative change is ``+inf``). The property
        tiers draw prices from ``[1.0, PRICE_MAX]`` and cannot reach a zero, so this deterministic pin -- asserting both
        the implementation AND the oracle against the literal -- is what protects the oracle's ``copysign`` sign factor.
        """
        cases: list[tuple[list[float], list[float | None]]] = [
            ([-0.0, 10.0], [None, -math.inf]),
            ([-0.0, -10.0], [None, math.inf]),
        ]
        for values, expected in cases:
            assert_matches(apply_expr(values, returns_simple(pl.col(COLUMN_X))), expected)
            assert_matches(returns_simple_reference(values), expected)

    def test_consecutive_infinities_make_nan(self) -> None:
        """
        Verifies IEEE infinity handling against the reference: two consecutive equal-sign infinite prices make the
        second return ``inf / inf - 1 = NaN``. The property tiers cannot reach this (their strategies set
        ``allow_infinity=False``), so it is pinned here.
        """
        values = [math.inf, math.inf, 1.0, -math.inf]
        assert_matches(apply_expr(values, returns_simple(pl.col(COLUMN_X))), returns_simple_reference(values))


class TestReturnsSimpleCorrectness:
    """
    Against the naive reference oracle and frozen golden-master values.
    """

    def test_matches_reference(self) -> None:
        """
        Verifies agreement with the naive closed-form reference over a representative positive-price series.
        """
        values = [100.0, 102.0, 101.5, 105.0, 103.0, 108.0, 110.0, 109.0]
        assert_matches(
            apply_expr(values, returns_simple(pl.col(COLUMN_X))),
            returns_simple_reference(values),
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference over a five-bar price series.
        """
        result = apply_expr([100.0, 105.0, 102.0, 108.0, 110.0], returns_simple(pl.col(COLUMN_X)).round(4))
        assert_matches(result, [None, 0.05, -0.0286, 0.0588, 0.0185])


class TestReturnsSimpleProperties:
    """
    Invariants that must hold for all inputs (property-based).
    """

    @given(case=_cases(_POSITIVE_PRICES, min_size=0))
    def test_matches_reference_for_any_input(
        self,
        case: list[float],
    ) -> None:
        """
        Verifies that, for any positive-price series, the implementation matches the naive reference.
        """
        values = case
        assert_matches(
            apply_expr(values, returns_simple(pl.col(COLUMN_X))),
            returns_simple_reference(values),
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    @given(case=_cases(positive_missing_data(PRICE_MAX), min_size=0))
    def test_matches_reference_under_missing_data(
        self,
        case: list[float | None],
    ) -> None:
        """
        Verifies that, for inputs freely mixing null / NaN / finite positive prices, the implementation matches the
        naive reference.
        """
        values = case
        assert_matches(
            apply_expr(values, returns_simple(pl.col(COLUMN_X))),
            returns_simple_reference(values),
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    @given(
        case=_cases(_POSITIVE_PRICES),
        exponent=st.sampled_from([-4, -3, -2, -1, 1, 2, 3, 4]),
    )
    def test_scale_invariance(
        self,
        case: list[float],
        exponent: int,
    ) -> None:
        """
        Verifies that ``returns_simple`` is scale-invariant: scaling every input value by a constant ``k`` leaves
        the output unchanged -- ``returns_simple(k * x) == returns_simple(x)``. ``k`` is a power of two, so the
        rescale is exact and adds no floating-point error.
        """
        k = 2.0**exponent
        values = case
        result_base = apply_expr(values, returns_simple(pl.col(COLUMN_X)))
        result_scaled = apply_expr([value * k for value in values], returns_simple(pl.col(COLUMN_X)))
        assert_matches(
            result_scaled, result_base, rel_tol=RELATIVE_TOLERANCE_PROPERTY, abs_tol=ABSOLUTE_TOLERANCE_REFERENCE
        )
