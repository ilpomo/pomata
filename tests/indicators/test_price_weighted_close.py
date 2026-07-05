"""
Tests for ``pomata.indicators.price_weighted_close`` — Weighted Close Price.

``price_weighted_close`` is multi-input (``high`` / ``low`` / ``close``) and elementwise (no window, no cross-bar
state). Tests use a local ``apply_price_weighted_close`` helper to materialize it over a three-column ``Float64``
frame; ``assert_matches`` and the naive ``price_weighted_close_reference`` oracle are shared across the suite.

The ladder, adapted to an elementwise transform: contract (type / shape / lazy-eager / ``.over`` identity), edge
(empty / single-row / constant / null / NaN / null-precedence), correctness (closed-form reference + frozen golden
master), and properties (reference agreement incl. missing data, scale-homogeneity, and large-magnitude stability).
Categories are split into classes; cross-cutting categories use markers (see ``tests/README.md``).
"""

import math
from collections.abc import Sequence

import polars as pl
from hypothesis import given
from hypothesis import strategies as st
from tests.indicators.oracles import price_weighted_close_reference
from tests.support import (
    ABSOLUTE_TOLERANCE_REFERENCE,
    ABSOLUTE_TOLERANCE_STREAMING,
    CLOSE,
    HIGH,
    LOW,
    RELATIVE_TOLERANCE_PROPERTY,
    RELATIVE_TOLERANCE_REFERENCE,
    RELATIVE_TOLERANCE_SCALE,
    assert_matches,
    assert_scale_homogeneous,
    coherent_hlc,
    coherent_hlc_with_missing,
    materialize,
    split_triples,
)

from pomata.indicators import price_weighted_close

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- derived, not chosen; the rationale is the shared method in CORRECTNESS.md, the numbers are this
# indicator's. To add an indicator, set its three facts here; the property tier below is then the same shape as every
# other indicator's.
#   1. warmup  W = 0   (windowless and elementwise: every row is defined from row 0 -- each bar uses only its own bar)
#   2. memory  the oracle shares pomata's per-row recomputation, so the property holds from row 0 (M = 0); with W = 0
#              there is no warm-up to outlast, so a case is simply a series of bars -- every row is output
#   3. domain  coherent_hlc(): coherent positive-finite (low <= close <= high) bars; SERIES_MAX bars span several sizes
# price_weighted_close is a windowless weighted mean ``(high + low + 2 * close) / 4``, homogeneous of degree 1 (so it
# keeps the scale-homogeneity and large-magnitude tiers, unlike a scale-invariant ratio). It has no window parameter, so
# ``_cases`` draws only the series (no window to couple, hence no ``window`` in the unpacked pair). Repetitions N are
# the shared CI profile (tests/conftest.py); override per-test only if its parameter space is larger.
# ----------------------------------------------------------------------------------------------------------------------
SERIES_MAX = 50


@st.composite
def _cases[T](draw: st.DrawFn, bars: st.SearchStrategy[T], min_size: int = 1) -> list[T]:
    """
    A series of bars sized from the facts above. price_weighted_close is windowless (W = 0), so -- unlike the windowed
    indicators' ``(series, window)`` pair -- a case is just the series: every row is output, never warm-up.
    """
    # NOTE: windowless -- returns the bare series (no window to couple length to); the W + D coupling of the windowed
    # ``_cases`` is vacuous here because W = 0 and every drawn row is already a defined output.
    return draw(st.lists(bars, min_size=min_size, max_size=SERIES_MAX))


def apply_price_weighted_close(
    high: Sequence[float | None],
    low: Sequence[float | None],
    close: Sequence[float | None],
) -> list[float | None]:
    """
    Materialize ``price_weighted_close`` over a three-column ``Float64`` frame from the aligned input lists.
    """
    return materialize(
        {HIGH: high, LOW: low, CLOSE: close}, price_weighted_close(pl.col(HIGH), pl.col(LOW), pl.col(CLOSE))
    )


class TestPriceWeightedCloseContract:
    """
    Type, shape, and lazy/eager guarantees.
    """

    def test_over_is_identity(self) -> None:
        """
        Verifies that ``.over`` is optional for this elementwise transform: partitioning by group is identical to the
        un-partitioned call (no cross-bar state can leak across group boundaries).
        """
        frame = pl.DataFrame(
            {
                "ticker": ["a", "a", "a", "b", "b", "b"],
                HIGH: pl.Series(HIGH, [11.0, 12.0, 13.0, 12.5, 14.0, 15.0], dtype=pl.Float64),
                LOW: pl.Series(LOW, [9.0, 10.0, 11.0, 11.0, 12.0, 13.0], dtype=pl.Float64),
                CLOSE: pl.Series(CLOSE, [10.0, 11.5, 12.5, 11.5, 13.5, 14.5], dtype=pl.Float64),
            }
        )
        plain = frame.select(price_weighted_close(pl.col(HIGH), pl.col(LOW), pl.col(CLOSE)).alias("y"))["y"].to_list()
        grouped = frame.select(
            price_weighted_close(pl.col(HIGH), pl.col(LOW), pl.col(CLOSE)).over("ticker").alias("y")
        )["y"].to_list()
        assert_matches(plain, grouped)
        assert_matches(plain, [10.0, 11.25, 12.25, 11.625, 13.25, 14.25])


class TestPriceWeightedCloseEdge:
    """
    Boundaries and null / NaN handling.
    """

    def test_all_null(self) -> None:
        """
        Verifies that an all-null input yields an all-null output (the sum propagates ``null`` on every row).
        """
        assert_matches(
            apply_price_weighted_close([None, None, None], [None, None, None], [None, None, None]), [None, None, None]
        )

    def test_single_row(self) -> None:
        """
        Verifies that a one-row series resolves to the single representative price (no window, no warm-up).
        """
        assert_matches(apply_price_weighted_close([11.0], [9.0], [10.0]), [10.0])

    def test_constant_series(self) -> None:
        """
        Verifies that when every input price equals the same constant the result is that constant on every row.
        """
        assert_matches(apply_price_weighted_close([5.0, 5.0, 5.0], [5.0, 5.0, 5.0], [5.0, 5.0, 5.0]), [5.0, 5.0, 5.0])

    def test_null_propagates(self) -> None:
        """
        Verifies that a ``null`` in any input makes that row ``null`` (the sum propagates ``null``).
        """
        assert_matches(
            apply_price_weighted_close([11.0, None, 13.0], [9.0, 10.0, 11.0], [10.0, 11.5, 12.5]), [10.0, None, 12.25]
        )

    def test_nan_propagates(self) -> None:
        """
        Verifies that a ``NaN`` in any input makes that row ``NaN`` (the sum propagates ``NaN``).
        """
        assert_matches(
            apply_price_weighted_close([11.0, float("nan"), 13.0], [9.0, 10.0, 11.0], [10.0, 11.5, 12.5]),
            [10.0, math.nan, 12.25],
        )

    def test_null_takes_precedence_over_nan(self) -> None:
        """
        Verifies that a row carrying both a ``null`` (in ``high``) and a ``NaN`` (in ``low``) yields
        ``null`` — ``null`` takes precedence over ``NaN``.
        """
        assert_matches(apply_price_weighted_close([11.0, None], [9.0, float("nan")], [10.0, 11.5]), [10.0, None])


class TestPriceWeightedCloseCorrectness:
    """
    Against the naive reference oracle and frozen golden-master values.
    """

    def test_matches_reference(self) -> None:
        """
        Verifies agreement with the naive closed-form reference over a representative OHLC series.
        """
        high = [11.0, 12.0, 13.0, 12.5, 14.0, 15.0, 14.5, 16.0]
        low = [9.0, 10.0, 11.0, 11.0, 12.0, 13.0, 12.5, 14.0]
        close = [10.0, 11.5, 12.5, 11.5, 13.5, 14.5, 13.0, 15.5]
        assert_matches(
            apply_price_weighted_close(high, low, close),
            price_weighted_close_reference(high, low, close),
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference over a five-bar OHLC series.
        """
        assert_matches(
            apply_price_weighted_close(
                [11.0, 12.0, 13.0, 12.5, 14.0], [9.0, 10.0, 11.0, 11.0, 12.0], [10.0, 11.5, 12.5, 11.5, 13.5]
            ),
            [10.0, 11.25, 12.25, 11.625, 13.25],
        )


class TestPriceWeightedCloseProperties:
    """
    Invariants that must hold for all inputs (property-based).
    """

    # NOTE: exact transform -- implementation and oracle compute identical arithmetic, residual is zero, so a fixed
    # reference band applies here (not input_scale-sized like the sum-based degree-1 kernels).
    @given(
        case=_cases(coherent_hlc(), min_size=0),
    )
    def test_matches_reference_for_any_input(
        self,
        case: list[tuple[float, float, float]],
    ) -> None:
        """
        Verifies that, for any aligned input series, the implementation matches the naive reference.
        """
        rows = case
        high, low, close = split_triples(rows)
        assert_matches(
            apply_price_weighted_close(high, low, close),
            price_weighted_close_reference(high, low, close),
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    @given(
        case=_cases(coherent_hlc()),
        exponent=st.sampled_from([-4, -3, -2, -1, 1, 2, 3, 4]),
    )
    def test_scale_homogeneity(
        self,
        case: list[tuple[float, float, float]],
        exponent: int,
    ) -> None:
        """
        Verifies that the transform is homogeneous of degree 1: ``f(k * prices) == k * f(prices)``. ``k`` is a power of
        two so the rescaling is lossless and cannot introduce a sub-ULP drift into the comparison.
        """
        k = 2.0**exponent
        rows = case
        high, low, close = split_triples(rows)
        result_base = apply_price_weighted_close(high, low, close)
        high_scaled = [value * k for value in high]
        low_scaled = [value * k for value in low]
        close_scaled = [value * k for value in close]
        result_scaled = apply_price_weighted_close(high_scaled, low_scaled, close_scaled)
        assert_scale_homogeneous(result_scaled, result_base, k=k, degree=1)

    @given(
        case=_cases(coherent_hlc_with_missing()),
    )
    def test_matches_reference_under_missing_data(
        self,
        case: list[tuple[float | None, float | None, float | None]],
    ) -> None:
        """
        Verifies that, for inputs freely mixing null / NaN / finite, the implementation matches the naive reference.
        """
        rows = case
        high, low, close = split_triples(rows)
        assert_matches(
            apply_price_weighted_close(high, low, close),
            price_weighted_close_reference(high, low, close),
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    @given(
        case=_cases(coherent_hlc()),
        scale=st.sampled_from([1e-6, 1e6, 1e9]),
    )
    def test_matches_reference_at_large_magnitude(
        self,
        case: list[tuple[float, float, float]],
        scale: float,
    ) -> None:
        """
        Verifies that at extreme positive magnitudes the implementation stays finite where the reference is and agrees.
        """
        rows = case
        high = [row[0] * scale for row in rows]
        low = [row[1] * scale for row in rows]
        close = [row[2] * scale for row in rows]
        assert_matches(
            apply_price_weighted_close(high, low, close),
            price_weighted_close_reference(high, low, close),
            rel_tol=RELATIVE_TOLERANCE_SCALE,
            abs_tol=ABSOLUTE_TOLERANCE_STREAMING,
        )
