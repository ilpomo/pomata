"""
Tests for ``pomata.indicators.price_average`` — Average Price.

``price_average`` is multi-input (``open`` / ``high`` / ``low`` / ``close``) and elementwise (no window, no cross-bar
state). Tests use a local ``apply_price_average`` helper to materialize it over a four-column ``Float64`` frame;
``assert_matches`` and the naive ``price_average_reference`` oracle are shared across the suite.

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
from polars.testing import assert_frame_equal
from tests.indicators.oracles import price_average_reference
from tests.support import (
    ABSOLUTE_TOLERANCE_REFERENCE,
    ABSOLUTE_TOLERANCE_STREAMING,
    CLOSE,
    HIGH,
    LOW,
    OPEN,
    RELATIVE_TOLERANCE_PROPERTY,
    RELATIVE_TOLERANCE_REFERENCE,
    RELATIVE_TOLERANCE_SCALE,
    assert_matches,
    assert_scale_homogeneous,
    coherent_ohlc,
    coherent_ohlc_with_missing,
    materialize,
    split_quads,
)

from pomata.indicators import price_average

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- derived, not chosen; the rationale is the shared method in CORRECTNESS.md, the numbers are this
# indicator's. To add an indicator, set its three facts here; the property tier below is then the same shape as every
# other indicator's.
#   1. warmup  W = 0   (windowless and elementwise: every row is defined from row 0 -- each bar uses only its own OHLC)
#   2. memory  the oracle shares pomata's per-row recomputation, so the property holds from row 0 (M = 0); with W = 0
#              there is no warm-up to outlast, so a case is simply a series of bars -- every row is output
#   3. domain  coherent_ohlc(): coherent positive-finite OHLC bars; SERIES_MAX bars span several total sizes
# price_average is a windowless equal-weighted mean of the four OHLC prices, homogeneous of degree 1 (so it keeps the
# scale-homogeneity and large-magnitude tiers, unlike a scale-invariant ratio). It has no window parameter, so
# ``_cases`` draws only the series (no window to couple, hence no ``window`` in the unpacked pair). Repetitions N are
# the shared CI profile (tests/conftest.py); override per-test only if its parameter space is larger.
# ----------------------------------------------------------------------------------------------------------------------
SERIES_MAX = 50


@st.composite
def _cases[T](draw: st.DrawFn, bars: st.SearchStrategy[T], min_size: int = 1) -> list[T]:
    """
    A series of bars sized from the facts above. price_average is windowless (W = 0), so -- unlike the windowed
    indicators' ``(series, window)`` pair -- a case is just the series: every row is output, never warm-up.
    """
    # NOTE: windowless -- returns the bare series (no window to couple length to); the W + D coupling of the windowed
    # ``_cases`` is vacuous here because W = 0 and every drawn row is already a defined output.
    return draw(st.lists(bars, min_size=min_size, max_size=SERIES_MAX))


def apply_price_average(
    open: Sequence[float | None],
    high: Sequence[float | None],
    low: Sequence[float | None],
    close: Sequence[float | None],
) -> list[float | None]:
    """
    Materialize ``price_average`` over a four-column ``Float64`` frame from the aligned input lists.
    """
    return materialize(
        {OPEN: open, HIGH: high, LOW: low, CLOSE: close},
        price_average(pl.col(OPEN), pl.col(HIGH), pl.col(LOW), pl.col(CLOSE)),
    )


class TestPriceAverageContract:
    """
    Type, shape, and lazy/eager guarantees.
    """

    def test_returns_expr(self) -> None:
        """
        Verifies that the factory returns a ``pl.Expr`` without touching a frame.
        """
        assert isinstance(price_average(pl.col(OPEN), pl.col(HIGH), pl.col(LOW), pl.col(CLOSE)), pl.Expr)

    def test_preserves_length_and_dtype(self) -> None:
        """
        Verifies that the output has one value per input row and is ``Float64``.
        """
        frame = pl.DataFrame(
            {
                OPEN: pl.Series(OPEN, [10.0, 11.0, 12.0, 11.5, 13.0], dtype=pl.Float64),
                HIGH: pl.Series(HIGH, [11.0, 12.0, 13.0, 12.5, 14.0], dtype=pl.Float64),
                LOW: pl.Series(LOW, [9.0, 10.0, 11.0, 11.0, 12.0], dtype=pl.Float64),
                CLOSE: pl.Series(CLOSE, [10.0, 11.5, 12.5, 11.5, 13.5], dtype=pl.Float64),
            }
        )
        result = frame.select(price_average(pl.col(OPEN), pl.col(HIGH), pl.col(LOW), pl.col(CLOSE)).alias("y"))
        assert result.height == frame.height
        assert result.schema["y"] == pl.Float64

    def test_lazy_eager_parity(self) -> None:
        """
        Verifies that eager and lazy application produce identical materialized output.
        """
        frame = pl.DataFrame(
            {
                OPEN: pl.Series(OPEN, [10.0, 11.0, 12.0, 11.5, 13.0], dtype=pl.Float64),
                HIGH: pl.Series(HIGH, [11.0, 12.0, 13.0, 12.5, 14.0], dtype=pl.Float64),
                LOW: pl.Series(LOW, [9.0, 10.0, 11.0, 11.0, 12.0], dtype=pl.Float64),
                CLOSE: pl.Series(CLOSE, [10.0, 11.5, 12.5, 11.5, 13.5], dtype=pl.Float64),
            }
        )
        expr = price_average(pl.col(OPEN), pl.col(HIGH), pl.col(LOW), pl.col(CLOSE)).alias("y")
        result_eager = frame.select(expr)
        result_lazy = frame.lazy().select(expr).collect()
        assert_frame_equal(result_eager, result_lazy)

    def test_over_is_identity(self) -> None:
        """
        Verifies that ``.over`` is optional for this elementwise transform: partitioning by group is identical to the
        un-partitioned call (no cross-bar state can leak across group boundaries).
        """
        frame = pl.DataFrame(
            {
                "ticker": ["a", "a", "a", "b", "b", "b"],
                OPEN: pl.Series(OPEN, [10.0, 11.0, 12.0, 11.5, 13.0, 14.0], dtype=pl.Float64),
                HIGH: pl.Series(HIGH, [11.0, 12.0, 13.0, 12.5, 14.0, 15.0], dtype=pl.Float64),
                LOW: pl.Series(LOW, [9.0, 10.0, 11.0, 11.0, 12.0, 13.0], dtype=pl.Float64),
                CLOSE: pl.Series(CLOSE, [10.0, 11.5, 12.5, 11.5, 13.5, 14.5], dtype=pl.Float64),
            }
        )
        plain = frame.select(price_average(pl.col(OPEN), pl.col(HIGH), pl.col(LOW), pl.col(CLOSE)).alias("y"))[
            "y"
        ].to_list()
        grouped = frame.select(
            price_average(pl.col(OPEN), pl.col(HIGH), pl.col(LOW), pl.col(CLOSE)).over("ticker").alias("y")
        )["y"].to_list()
        assert_matches(plain, grouped)
        assert_matches(plain, [10.0, 11.125, 12.125, 11.625, 13.125, 14.125])


class TestPriceAverageEdge:
    """
    Boundaries and null / NaN handling.
    """

    def test_empty(self) -> None:
        """
        Verifies that an empty input yields an empty output.
        """
        assert_matches(apply_price_average([], [], [], []), [])

    def test_all_null(self) -> None:
        """
        Verifies that an all-null input yields an all-null output (the sum propagates ``null`` on every row).
        """
        assert_matches(
            apply_price_average([None, None, None], [None, None, None], [None, None, None], [None, None, None]),
            [None, None, None],
        )

    def test_single_row(self) -> None:
        """
        Verifies that a one-row series resolves to the single representative price (no window, no warm-up).
        """
        assert_matches(apply_price_average([10.0], [11.0], [9.0], [10.0]), [10.0])

    def test_constant_series(self) -> None:
        """
        Verifies that when every input price equals the same constant the result is that constant on every row.
        """
        assert_matches(
            apply_price_average([5.0, 5.0, 5.0], [5.0, 5.0, 5.0], [5.0, 5.0, 5.0], [5.0, 5.0, 5.0]), [5.0, 5.0, 5.0]
        )

    def test_null_propagates(self) -> None:
        """
        Verifies that a ``null`` in any input makes that row ``null`` (the sum propagates ``null``).
        """
        assert_matches(
            apply_price_average([10.0, None, 12.0], [11.0, 12.0, 13.0], [9.0, 10.0, 11.0], [10.0, 11.5, 12.5]),
            [10.0, None, 12.125],
        )

    def test_nan_propagates(self) -> None:
        """
        Verifies that a ``NaN`` in any input makes that row ``NaN`` (the sum propagates ``NaN``).
        """
        assert_matches(
            apply_price_average([10.0, float("nan"), 12.0], [11.0, 12.0, 13.0], [9.0, 10.0, 11.0], [10.0, 11.5, 12.5]),
            [10.0, math.nan, 12.125],
        )

    def test_null_takes_precedence_over_nan(self) -> None:
        """
        Verifies that a row carrying both a ``null`` (in ``open``) and a ``NaN`` (in ``high``) yields
        ``null`` — ``null`` takes precedence over ``NaN``.
        """
        assert_matches(apply_price_average([10.0, None], [11.0, float("nan")], [9.0, 10.0], [10.0, 11.5]), [10.0, None])


class TestPriceAverageCorrectness:
    """
    Against the naive reference oracle and frozen golden-master values.
    """

    def test_matches_reference(self) -> None:
        """
        Verifies agreement with the naive closed-form reference over a representative OHLC series.
        """
        open = [10.0, 11.0, 12.0, 11.5, 13.0, 14.0, 13.5, 15.0]
        high = [11.0, 12.0, 13.0, 12.5, 14.0, 15.0, 14.5, 16.0]
        low = [9.0, 10.0, 11.0, 11.0, 12.0, 13.0, 12.5, 14.0]
        close = [10.0, 11.5, 12.5, 11.5, 13.5, 14.5, 13.0, 15.5]
        assert_matches(
            apply_price_average(open, high, low, close),
            price_average_reference(open, high, low, close),
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference over a five-bar OHLC series.
        """
        assert_matches(
            apply_price_average(
                [10.0, 11.0, 12.0, 11.5, 13.0],
                [11.0, 12.0, 13.0, 12.5, 14.0],
                [9.0, 10.0, 11.0, 11.0, 12.0],
                [10.0, 11.5, 12.5, 11.5, 13.5],
            ),
            [10.0, 11.125, 12.125, 11.625, 13.125],
        )


class TestPriceAverageProperties:
    """
    Invariants that must hold for all inputs (property-based).
    """

    # NOTE: exact transform -- implementation and oracle compute identical arithmetic, residual is zero, so a fixed
    # reference band applies here (not input_scale-sized like the sum-based degree-1 kernels).
    @given(
        case=_cases(coherent_ohlc(), min_size=0),
    )
    def test_matches_reference_for_any_input(
        self,
        case: list[tuple[float, float, float, float]],
    ) -> None:
        """
        Verifies that, for any aligned input series, the implementation matches the naive reference.
        """
        rows = case
        open, high, low, close = split_quads(rows)
        assert_matches(
            apply_price_average(open, high, low, close),
            price_average_reference(open, high, low, close),
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    @given(
        case=_cases(coherent_ohlc()),
        exponent=st.sampled_from([-4, -3, -2, -1, 1, 2, 3, 4]),
    )
    def test_scale_homogeneity(
        self,
        case: list[tuple[float, float, float, float]],
        exponent: int,
    ) -> None:
        """
        Verifies that the transform is homogeneous of degree 1: ``f(k * prices) == k * f(prices)``. ``k`` is a power of
        two so the rescaling is lossless and cannot introduce a floating-point artifact.
        """
        k = 2.0**exponent
        rows = case
        open, high, low, close = split_quads(rows)
        result_base = apply_price_average(open, high, low, close)
        open_scaled = [value * k for value in open]
        high_scaled = [value * k for value in high]
        low_scaled = [value * k for value in low]
        close_scaled = [value * k for value in close]
        result_scaled = apply_price_average(open_scaled, high_scaled, low_scaled, close_scaled)
        assert_scale_homogeneous(result_scaled, result_base, k=k, degree=1)

    @given(
        case=_cases(coherent_ohlc_with_missing()),
    )
    def test_matches_reference_under_missing_data(
        self,
        case: list[tuple[float | None, float | None, float | None, float | None]],
    ) -> None:
        """
        Verifies that, for inputs freely mixing null / NaN / finite, the implementation matches the naive reference.
        """
        rows = case
        open, high, low, close = split_quads(rows)
        assert_matches(
            apply_price_average(open, high, low, close),
            price_average_reference(open, high, low, close),
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    @given(
        case=_cases(coherent_ohlc()),
        scale=st.sampled_from([1e-6, 1e6, 1e9]),
    )
    def test_matches_reference_at_large_magnitude(
        self,
        case: list[tuple[float, float, float, float]],
        scale: float,
    ) -> None:
        """
        Verifies that at extreme positive magnitudes the implementation stays finite where the reference is and agrees.
        """
        rows = case
        open = [row[0] * scale for row in rows]
        high = [row[1] * scale for row in rows]
        low = [row[2] * scale for row in rows]
        close = [row[3] * scale for row in rows]
        assert_matches(
            apply_price_average(open, high, low, close),
            price_average_reference(open, high, low, close),
            rel_tol=RELATIVE_TOLERANCE_SCALE,
            abs_tol=ABSOLUTE_TOLERANCE_STREAMING,
        )
