"""
Tests for ``pomata.indicators.absolute_price_oscillator`` — the Absolute Price Oscillator (fast EMA minus slow EMA).

``absolute_price_oscillator`` is single-input and built from two EMAs, so tests use the shared ``apply_expr`` helper
to materialize the factory over a one-column ``Float64`` frame; ``assert_matches`` and the naive
``absolute_price_oscillator_reference`` oracle are shared across the suite.

The ladder is the canonical one: contract (type / shape / lazy-eager / ``.over`` independence), edge (warm-up / equal
windows / single-row / null / NaN), correctness (vs the closed-form reference and a frozen golden master), and
properties (reference agreement incl. missing data, scale-homogeneity, and large-magnitude stability). Categories are
split into classes; cross-cutting categories use markers (see ``tests/README.md``).
"""

import math

import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st
from polars.testing import assert_frame_equal
from tests.indicators.oracles import absolute_price_oscillator_reference
from tests.support import (
    ABSOLUTE_TOLERANCE_EXACT,
    COLUMN_X,
    GROUP_KEY,
    RELATIVE_TOLERANCE_PROPERTY,
    RELATIVE_TOLERANCE_REFERENCE,
    RELATIVE_TOLERANCE_SCALE,
    STREAMING_TOLERANCE_FACTOR,
    apply_expr,
    assert_matches,
    assert_scale_homogeneous,
    input_scale,
    missing_data_floats,
)

from pomata.indicators import absolute_price_oscillator

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- derived, not chosen; the rationale is the shared method in CORRECTNESS.md, the numbers are this
# indicator's. To add an indicator, set its three facts here; the property tier below is then the same shape as every
# other indicator's.
#   1. warmup  W = max(window_fast, window_slow) - 1   (values are null until the slower EMA leaves its warm-up; the
#              contract requires window_fast <= window_slow, so the slow EMA is the one that warms up last)
#   2. memory  the oracle shares pomata's recursive EMA seeding, so the property holds from the first defined row
#              (M = 0); each example carries D in [window_slow, 2 * window_slow] defined values -- one slow window of
#              output, never all warm-up
#   3. domain  finite values, the safe magnitude widened per test below; windows span 1 .. WINDOW_MAX
# APO is homogeneous of degree 1 (a difference of two EMAs, each linear in the price), so it carries a degree-1
# scale-homogeneity property and magnitude-relative tolerances: the streaming EMA difference and its two-pass oracle
# diverge by about ``input_scale * machine_eps`` on degenerate inputs, sized by STREAMING_TOLERANCE_FACTOR. Repetitions
# N are the shared CI profile (tests/conftest.py); override per-test only if its parameter space is larger.
# ----------------------------------------------------------------------------------------------------------------------
WINDOW_MAX = 15


@st.composite
def _cases[T](draw: st.DrawFn, values: st.SearchStrategy[T]) -> tuple[list[T], int, int]:
    """
    A (series, window_fast, window_slow) triple sized from the facts above honoring the contract
    ``1 <= window_fast <= window_slow``: ``window_slow`` is drawn first and ``window_fast`` is then bounded above by it
    (so the now-rejected reversed regime is never sampled), and length = warm-up (driven by ``window_slow``) + a slow
    window of defined values, so every example has output to check (never an all-warm-up series, the waste a window
    decoupled from the length would cause).
    """
    window_slow = draw(st.integers(min_value=1, max_value=WINDOW_MAX))
    window_fast = draw(st.integers(min_value=1, max_value=window_slow))
    defined = draw(st.integers(min_value=window_slow, max_value=2 * window_slow))
    length = (window_slow - 1) + defined
    return draw(st.lists(values, min_size=length, max_size=length)), window_fast, window_slow


class TestAbsolutePriceOscillatorContract:
    """
    Type, shape, and lazy/eager guarantees.
    """

    def test_returns_expr(self) -> None:
        """
        Verifies that the factory returns a ``pl.Expr`` without touching a frame.
        """
        assert isinstance(absolute_price_oscillator(pl.col(COLUMN_X), window_fast=2, window_slow=3), pl.Expr)

    def test_preserves_length_and_dtype(self) -> None:
        """
        Verifies that the output has one value per input row and is ``Float64``.
        """
        frame = pl.DataFrame({COLUMN_X: pl.Series(COLUMN_X, [1.0, 2.0, 3.0, 4.0, 5.0])})
        result = frame.select(absolute_price_oscillator(pl.col(COLUMN_X), window_fast=2, window_slow=3).alias("y"))
        assert result.height == frame.height
        assert result.schema["y"] == pl.Float64

    def test_lazy_eager_parity(self) -> None:
        """
        Verifies that eager and lazy application produce identical materialized output.
        """
        frame = pl.DataFrame({COLUMN_X: pl.Series(COLUMN_X, [1.0, 2.0, 3.0, 4.0, 5.0])})
        expr = absolute_price_oscillator(pl.col(COLUMN_X), window_fast=2, window_slow=3).alias("y")
        result_eager = frame.select(expr)
        result_lazy = frame.lazy().select(expr).collect()
        assert_frame_equal(result_eager, result_lazy)

    def test_over_partitions_independently(self) -> None:
        """
        Verifies that under ``.over`` each EMA resets per group and never spans group boundaries.
        """
        frame = pl.DataFrame(
            {GROUP_KEY: ["a"] * 4 + ["b"] * 4, COLUMN_X: [10.0, 11.0, 12.0, 11.0, 20.0, 22.0, 24.0, 22.0]}
        )
        expr = absolute_price_oscillator(pl.col(COLUMN_X), window_fast=2, window_slow=3).over(GROUP_KEY).round(4)
        result = frame.select(expr.alias("y"))["y"].to_list()
        assert_matches(result, [None, None, 0.5, 0.1667, None, None, 1.0, 0.3333])


class TestAbsolutePriceOscillatorEdge:
    """
    Boundaries, warm-up, and null / NaN handling.
    """

    def test_windows_below_one_raises(self) -> None:
        """
        Verifies that a window ``< 1`` raises ``ValueError``.
        """
        with pytest.raises(ValueError, match="window_fast must be >= 1"):
            absolute_price_oscillator(pl.col(COLUMN_X), window_fast=0, window_slow=3)
        with pytest.raises(ValueError, match="window_slow must be >= 1"):
            absolute_price_oscillator(pl.col(COLUMN_X), window_fast=2, window_slow=0)

    def test_fast_above_slow_raises(self) -> None:
        """
        Verifies that ``window_fast > window_slow`` raises ``ValueError`` (the fast leg must be the shorter one), while
        the equal-window case is accepted.
        """
        with pytest.raises(ValueError, match="window_fast must be <= window_slow"):
            absolute_price_oscillator(pl.col(COLUMN_X), window_fast=5, window_slow=3)
        assert isinstance(absolute_price_oscillator(pl.col(COLUMN_X), window_fast=3, window_slow=3), pl.Expr)

    def test_warmup_null_count(self) -> None:
        """
        Verifies that values are null until the slow EMA leaves its warm-up (the first ``window_slow - 1`` rows).
        """
        result = apply_expr(
            [1.0, 2.0, 3.0, 4.0, 5.0], absolute_price_oscillator(pl.col(COLUMN_X), window_fast=2, window_slow=3)
        )
        assert result[:2] == [None, None]
        assert result[2] is not None

    def test_equal_windows_are_zero(self) -> None:
        """
        Verifies that equal fast and slow windows cancel to ``0`` once warmed up (the two EMAs are identical).
        """
        result = apply_expr(
            [10.0, 11.0, 12.0], absolute_price_oscillator(pl.col(COLUMN_X), window_fast=2, window_slow=2)
        )
        assert_matches(result, [None, 0.0, 0.0])

    def test_single_row(self) -> None:
        """
        Verifies behavior on a one-element series: the slow EMA never warms up, so the result is all warm-up.
        """
        assert_matches(
            apply_expr([42.0], absolute_price_oscillator(pl.col(COLUMN_X), window_fast=2, window_slow=3)), [None]
        )

    def test_window_exceeds_length(self) -> None:
        """
        Verifies that when the longest window exceeds the series length the whole output is null (no slow-EMA value).
        """
        assert_matches(
            apply_expr([1.0, 2.0], absolute_price_oscillator(pl.col(COLUMN_X), window_fast=2, window_slow=3)),
            [None, None],
        )

    def test_empty(self) -> None:
        """
        Verifies behavior on an empty series.
        """
        assert_matches(apply_expr([], absolute_price_oscillator(pl.col(COLUMN_X), window_fast=2, window_slow=3)), [])

    def test_all_null(self) -> None:
        """
        Verifies that an all-null series yields all null.
        """
        assert_matches(
            apply_expr([None, None, None], absolute_price_oscillator(pl.col(COLUMN_X), window_fast=2, window_slow=3)),
            [None, None, None],
        )

    def test_null_propagates(self) -> None:
        """
        Verifies that a ``null`` contaminates the recursive EMA state and yields ``null`` for subsequent rows.
        """
        result = apply_expr(
            [10.0, 11.0, None, 13.0, 15.0], absolute_price_oscillator(pl.col(COLUMN_X), window_fast=2, window_slow=3)
        )
        assert result[2] is None

    def test_nan_propagates(self) -> None:
        """
        Verifies that a ``NaN`` propagates through both EMAs, yielding ``NaN``.
        """
        result = apply_expr(
            [10.0, 11.0, 12.0, math.nan, 15.0],
            absolute_price_oscillator(pl.col(COLUMN_X), window_fast=2, window_slow=3),
        )
        assert result[-1] is not None
        assert math.isnan(result[-1])


class TestAbsolutePriceOscillatorCorrectness:
    """
    Against the naive reference oracle and frozen golden-master values.
    """

    def test_matches_reference(self) -> None:
        """
        Verifies agreement with the naive closed-form reference across several window pairs.
        """
        values = [10.0, 11.0, 12.0, 11.0, 13.0, 14.0, 13.0, 15.0, 14.0, 16.0]
        for window_fast, window_slow in ((2, 3), (3, 5), (1, 4), (4, 4)):
            result = apply_expr(
                values, absolute_price_oscillator(pl.col(COLUMN_X), window_fast=window_fast, window_slow=window_slow)
            )
            assert_matches(
                result,
                absolute_price_oscillator_reference(values, window_fast, window_slow),
                rel_tol=RELATIVE_TOLERANCE_REFERENCE,
                abs_tol=ABSOLUTE_TOLERANCE_EXACT,
            )

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference: absolute_price_oscillator(fast=2, slow=3) over the sample series.
        """
        result = apply_expr(
            [10.0, 11.0, 12.0, 11.0, 13.0, 14.0, 13.0, 15.0],
            absolute_price_oscillator(pl.col(COLUMN_X), window_fast=2, window_slow=3).round(4),
        )
        assert_matches(result, [None, None, 0.5, 0.1667, 0.3889, 0.463, 0.1543, 0.3848])


class TestAbsolutePriceOscillatorProperties:
    """
    Invariants that must hold for all inputs (property-based).
    """

    @given(
        case=_cases(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)),
    )
    def test_matches_reference_for_any_input(
        self,
        case: tuple[list[float], int, int],
    ) -> None:
        """
        Verifies that, for any series and window pair, the implementation matches the naive reference.
        """
        values, window_fast, window_slow = case
        assert_matches(
            apply_expr(
                values, absolute_price_oscillator(pl.col(COLUMN_X), window_fast=window_fast, window_slow=window_slow)
            ),
            absolute_price_oscillator_reference(values, window_fast, window_slow),
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=input_scale(values) * STREAMING_TOLERANCE_FACTOR,
        )

    @given(
        case=_cases(st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False)),
        exponent=st.sampled_from([-4, -3, -2, -1, 1, 2, 3, 4]),
    )
    def test_scale_homogeneity(
        self,
        case: tuple[list[float], int, int],
        exponent: int,
    ) -> None:
        """
        Verifies that APO is homogeneous of degree 1:
        ``absolute_price_oscillator(k * x) == k * absolute_price_oscillator(x)`` (both EMAs are linear). ``k`` is a
        power of two so the rescaling is lossless and cannot introduce a sub-ULP drift.
        """
        k = 2.0**exponent
        values, window_fast, window_slow = case
        scaled_values = [value * k for value in values]
        result_base = apply_expr(
            values, absolute_price_oscillator(pl.col(COLUMN_X), window_fast=window_fast, window_slow=window_slow)
        )
        result_scaled = apply_expr(
            scaled_values, absolute_price_oscillator(pl.col(COLUMN_X), window_fast=window_fast, window_slow=window_slow)
        )
        assert_scale_homogeneous(result_scaled, result_base, k=k, degree=1)

    @given(
        case=_cases(missing_data_floats()),
    )
    def test_matches_reference_under_missing_data(
        self,
        case: tuple[list[float | None], int, int],
    ) -> None:
        """
        Verifies that, for inputs freely mixing null / NaN / finite, the implementation matches the naive reference.
        """
        values, window_fast, window_slow = case
        assert_matches(
            apply_expr(
                values, absolute_price_oscillator(pl.col(COLUMN_X), window_fast=window_fast, window_slow=window_slow)
            ),
            absolute_price_oscillator_reference(values, window_fast, window_slow),
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=input_scale(values) * STREAMING_TOLERANCE_FACTOR,
        )

    @given(
        case=_cases(st.floats(min_value=1e-3, max_value=1.0, allow_nan=False, allow_infinity=False)),
        scale=st.sampled_from([1e-6, 1e6, 1e9]),
    )
    def test_matches_reference_at_large_magnitude(
        self,
        case: tuple[list[float], int, int],
        scale: float,
    ) -> None:
        """
        Verifies that at extreme positive magnitudes the implementation stays finite where the reference is and agrees.
        """
        values, window_fast, window_slow = case
        scaled_values = [value * scale for value in values]
        assert_matches(
            apply_expr(
                scaled_values,
                absolute_price_oscillator(pl.col(COLUMN_X), window_fast=window_fast, window_slow=window_slow),
            ),
            absolute_price_oscillator_reference(scaled_values, window_fast, window_slow),
            rel_tol=RELATIVE_TOLERANCE_SCALE,
            abs_tol=input_scale(scaled_values) * STREAMING_TOLERANCE_FACTOR,
        )
