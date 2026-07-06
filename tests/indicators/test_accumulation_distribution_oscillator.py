"""
Tests for ``pomata.indicators.accumulation_distribution_oscillator`` — the Chaikin Oscillator.

The indicator is multi-input (high, low, close, volume) and single-output, so tests use a local
``apply_accumulation_distribution_oscillator`` helper to materialize the factory over a four-column ``Float64``
frame; ``assert_matches`` and the naive
``accumulation_distribution_oscillator_reference`` oracle (which composes ``accumulation_distribution_reference`` and
``ema_reference``) are shared across the suite. The oscillator is homogeneous of degree 1 (the AD line scales with
volume while the multiplier is price-invariant), so it carries scale-homogeneity and large-magnitude properties. Tests
use small spans (fast 2, slow 3) to keep the warm-up short.

The ladder is the canonical one: contract, edge (warm-up / null / NaN), correctness (vs the closed-form reference and a
frozen golden master), and properties (reference agreement incl. missing data, degree-1 scale-homogeneity,
large-magnitude stability). Categories are split into classes; cross-cutting categories use markers (see
``tests/README.md``).
"""

import math
from collections.abc import Sequence

import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st
from tests.indicators.oracles import accumulation_distribution_oscillator_reference
from tests.support import (
    ABSOLUTE_TOLERANCE_SCALE,
    ABSOLUTE_TOLERANCE_STREAMING,
    CLOSE,
    GROUP_KEY,
    HIGH,
    LOW,
    RELATIVE_TOLERANCE_PROPERTY,
    RELATIVE_TOLERANCE_SCALE,
    VOLUME,
    assert_matches,
    assert_scale_homogeneous,
    coherent_hlcv,
    coherent_hlcv_with_missing,
    count_leading_nulls,
    materialize,
    split_quads,
)

from pomata.indicators import accumulation_distribution_oscillator

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- derived, not chosen; the rationale is the shared method in CORRECTNESS.md, the numbers are this
# indicator's. To add an indicator, set its three facts here; the property tier below is then the same shape as every
# other indicator's.
#   1. warmup  W = window_slow - 1   (the oscillator inherits the slow EMA's warm-up over the accumulation/distribution
#              line). The property tier holds the spans fixed at WINDOW_FAST / WINDOW_SLOW, so W is the constant below.
#   2. memory  the oracle shares pomata's recursive EMA seeding, so the property holds from the first defined row
#              (M = 0); each example carries D in [1, SERIES_MAX] defined bars on top of W -- one window of output,
#              never an all-warm-up series
#   3. domain  coherent_hlcv(): coherent (high >= low, low <= close <= high) positive-finite OHLCV bars -- the
#              money-flow multiplier divides by ``high - low`` and is unbounded on impossible bars, outside the
#              indicator's domain; the missing-data tier draws from coherent_hlcv_with_missing. SERIES_MAX bars span
#              several total sizes
# The oscillator is homogeneous of degree 1 (the multiplier is price-invariant while the line scales with volume), so it
# carries a degree-1 scale-homogeneity property and a large-magnitude tier. Repetitions N are the shared CI profile
# (tests/conftest.py); override per-test only if its parameter space is larger.
# ----------------------------------------------------------------------------------------------------------------------
WINDOW_FAST = 2
WINDOW_SLOW = 3
WARMUP = WINDOW_SLOW - 1
SERIES_MAX = 50


@st.composite
def _cases[T](draw: st.DrawFn, bars: st.SearchStrategy[T]) -> list[T]:
    """
    A series sized from the facts above: the property tier holds the spans fixed, so -- unlike the windowed indicators'
    ``(series, window)`` pair -- a case is just the series, floored at ``WARMUP + 1`` bars so the oscillator always has
    at least one defined output (never an all-warm-up series).
    """
    defined = draw(st.integers(min_value=1, max_value=SERIES_MAX))
    length = WARMUP + defined
    return draw(st.lists(bars, min_size=length, max_size=length))


def apply_accumulation_distribution_oscillator(
    high: Sequence[float | None],
    low: Sequence[float | None],
    close: Sequence[float | None],
    volume: Sequence[float | None],
    window_fast: int = 2,
    window_slow: int = 3,
) -> list[float | None]:
    """
    Materialize ``accumulation_distribution_oscillator`` over a four-column ``Float64`` frame built from the inputs.
    """
    return materialize(
        {HIGH: high, LOW: low, CLOSE: close, VOLUME: volume},
        accumulation_distribution_oscillator(
            pl.col(HIGH),
            pl.col(LOW),
            pl.col(CLOSE),
            pl.col(VOLUME),
            window_fast=window_fast,
            window_slow=window_slow,
        ),
    )


class TestAccumulationDistributionOscillatorContract:
    """
    Type, shape, and lazy/eager guarantees.
    """

    def test_over_partitions_independently(self) -> None:
        """
        Verifies that under ``.over`` the running sum and the EMAs reset per group: the partitioned line equals the
        per-group calls.
        """
        frame = pl.DataFrame(
            {
                GROUP_KEY: ["a"] * 4 + ["b"] * 4,
                HIGH: [10.2, 10.5, 10.7, 10.3, 20.2, 20.5, 20.7, 20.3],
                LOW: [9.8, 10.0, 10.2, 9.9, 19.8, 20.0, 20.2, 19.9],
                CLOSE: [10.0, 10.3, 10.5, 10.1, 20.0, 20.3, 20.5, 20.1],
                VOLUME: [100.0, 150.0, 120.0, 200.0, 100.0, 150.0, 120.0, 200.0],
            }
        )
        expr = accumulation_distribution_oscillator(
            pl.col(HIGH), pl.col(LOW), pl.col(CLOSE), pl.col(VOLUME), window_fast=2, window_slow=3
        ).over(GROUP_KEY)
        grouped = frame.select(expr.alias("y"))["y"].to_list()
        group_a = apply_accumulation_distribution_oscillator(
            [10.2, 10.5, 10.7, 10.3], [9.8, 10.0, 10.2, 9.9], [10.0, 10.3, 10.5, 10.1], [100.0, 150.0, 120.0, 200.0]
        )
        group_b = apply_accumulation_distribution_oscillator(
            [20.2, 20.5, 20.7, 20.3], [19.8, 20.0, 20.2, 19.9], [20.0, 20.3, 20.5, 20.1], [100.0, 150.0, 120.0, 200.0]
        )
        assert_matches(grouped, group_a + group_b)


class TestAccumulationDistributionOscillatorEdge:
    """
    Boundaries, warm-up, and null / NaN handling.
    """

    def test_window_below_one_raises(self) -> None:
        """
        Verifies that either window ``< 1`` raises ``ValueError``.
        """
        with pytest.raises(ValueError, match="window_fast must be >= 1"):
            accumulation_distribution_oscillator(
                pl.col(HIGH), pl.col(LOW), pl.col(CLOSE), pl.col(VOLUME), window_fast=0, window_slow=10
            )
        with pytest.raises(ValueError, match="window_slow must be >= 1"):
            accumulation_distribution_oscillator(
                pl.col(HIGH), pl.col(LOW), pl.col(CLOSE), pl.col(VOLUME), window_fast=3, window_slow=0
            )

    def test_fast_above_slow_raises(self) -> None:
        """
        Verifies that ``window_fast > window_slow`` raises ``ValueError`` (a reversed pair flips the oscillator's sign
        and warms up over ``window_fast - 1`` rows rather than the documented ``window_slow - 1``).
        """
        with pytest.raises(ValueError, match="windows must be ordered window_fast <= window_slow"):
            accumulation_distribution_oscillator(
                pl.col(HIGH), pl.col(LOW), pl.col(CLOSE), pl.col(VOLUME), window_fast=4, window_slow=2
            )
        with pytest.raises(ValueError, match="windows must be ordered window_fast <= window_slow"):
            accumulation_distribution_oscillator(
                pl.col(HIGH), pl.col(LOW), pl.col(CLOSE), pl.col(VOLUME), window_fast=11, window_slow=10
            )

    def test_warmup_null_count(self) -> None:
        """
        Verifies that the line is null for the first ``window_slow - 1`` rows (the slow EMA warm-up).
        """
        result = apply_accumulation_distribution_oscillator(
            [10.2, 10.5, 10.7, 10.3], [9.8, 10.0, 10.2, 9.9], [10.0, 10.3, 10.5, 10.1], [100.0, 150.0, 120.0, 200.0]
        )
        assert result[:2] == [None, None]
        assert result[2] is not None

    def test_single_row(self) -> None:
        """
        Verifies behavior on a one-element series: the slow EMA warm-up never completes, so the result is null.
        """
        assert_matches(apply_accumulation_distribution_oscillator([10.0], [8.0], [9.0], [100.0]), [None])

    def test_window_equals_length(self) -> None:
        """
        Verifies the single defined value when the slow window equals the series length.
        """
        high = [10.2, 10.5, 10.7]
        low = [9.8, 10.0, 10.2]
        close = [10.0, 10.3, 10.5]
        volume = [100.0, 150.0, 120.0]
        assert_matches(
            apply_accumulation_distribution_oscillator(high, low, close, volume),
            accumulation_distribution_oscillator_reference(high, low, close, volume, 2, 3),
        )

    def test_window_exceeds_length(self) -> None:
        """
        Verifies that when the longer (slow) window exceeds the series length the result is all-null (the slow EMA
        warm-up never completes).
        """
        assert_matches(
            apply_accumulation_distribution_oscillator(
                [10.2, 10.5, 10.7], [9.8, 10.0, 10.2], [10.0, 10.3, 10.5], [100.0, 150.0, 120.0], 2, 5
            ),
            [None, None, None],
        )

    def test_all_null(self) -> None:
        """
        Verifies that an all-null OHLCV frame yields an all-null result (the line and its EMAs never seed).
        """
        assert_matches(
            apply_accumulation_distribution_oscillator([None] * 4, [None] * 4, [None] * 4, [None] * 4),
            [None, None, None, None],
        )

    def test_null_bridged(self) -> None:
        """
        Verifies that an interior ``null`` is bridged: the recursion carries its state across the gap.
        """
        high = [10.2, 10.5, 10.7, 10.3, 10.8]
        low = [9.8, 10.0, 10.2, 9.9, 10.3]
        close = [10.0, 10.3, 10.5, 10.1, 10.6]
        volume = [100.0, 150.0, None, 200.0, 200.0]
        assert_matches(
            apply_accumulation_distribution_oscillator(high, low, close, volume),
            accumulation_distribution_oscillator_reference(high, low, close, volume, 2, 3),
        )

    def test_nan_latches(self) -> None:
        """
        Verifies that a NaN propagates (matching the naive reference).
        """
        high = [10.2, 10.5, 10.7, 10.3, 10.8]
        low = [9.8, 10.0, 10.2, 9.9, 10.3]
        close = [10.0, 10.3, 10.5, 10.1, 10.6]
        volume = [100.0, 150.0, 150.0, 200.0, math.nan]
        assert_matches(
            apply_accumulation_distribution_oscillator(high, low, close, volume),
            accumulation_distribution_oscillator_reference(high, low, close, volume, 2, 3),
        )


class TestAccumulationDistributionOscillatorCorrectness:
    """
    Against the naive reference oracle and frozen golden-master values.
    """

    def test_matches_reference(self) -> None:
        """
        Verifies agreement with the naive closed-form reference across several span combinations.
        """
        high = [10.2, 10.5, 10.7, 10.3, 10.8, 11.0, 10.6, 11.2]
        low = [9.8, 10.0, 10.2, 9.9, 10.3, 10.5, 10.1, 10.7]
        close = [10.0, 10.3, 10.5, 10.1, 10.6, 10.8, 10.4, 11.0]
        volume = [100.0, 150.0, 120.0, 200.0, 180.0, 160.0, 140.0, 210.0]
        for window_fast, window_slow in ((2, 3), (3, 5), (1, 4)):
            assert_matches(
                apply_accumulation_distribution_oscillator(high, low, close, volume, window_fast, window_slow),
                accumulation_distribution_oscillator_reference(high, low, close, volume, window_fast, window_slow),
                rel_tol=RELATIVE_TOLERANCE_PROPERTY,
                abs_tol=ABSOLUTE_TOLERANCE_SCALE,
            )

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference: adosc(fast=2, slow=3) over the sample series.
        """
        result = apply_accumulation_distribution_oscillator(
            [10.2, 10.5, 10.7, 10.3, 10.8],
            [9.8, 10.0, 10.2, 9.9, 10.3],
            [10.0, 10.3, 10.5, 10.1, 10.6],
            [100.0, 150.0, 120.0, 200.0, 180.0],
        )
        assert_matches(
            [None if v is None else round(v, 4) for v in result],
            [None, None, 13.0, 8.6667, 11.0556],
        )


class TestAccumulationDistributionOscillatorProperties:
    """
    Invariants that must hold for all inputs (property-based).
    """

    @given(case=_cases(coherent_hlcv()))
    def test_matches_reference_for_any_input(
        self,
        case: list[tuple[float, float, float, float]],
    ) -> None:
        """
        Verifies that, for any coherent OHLC series, the implementation matches the naive reference. Bars are drawn
        coherent (``low <= close <= high``) because the money-flow multiplier divides by ``high - low`` and is unbounded
        on impossible bars — outside the indicator's domain (see ``coherent_hlcv``).
        """
        rows = case
        high, low, close, volume = split_quads(rows)
        assert_matches(
            apply_accumulation_distribution_oscillator(high, low, close, volume),
            accumulation_distribution_oscillator_reference(high, low, close, volume, 2, 3),
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=ABSOLUTE_TOLERANCE_SCALE,
        )

    @given(case=_cases(coherent_hlcv_with_missing()))
    def test_matches_reference_under_missing_data(
        self,
        case: list[tuple[float | None, float | None, float | None, float | None]],
    ) -> None:
        """
        Verifies that, for positive inputs freely mixing null / NaN, the implementation matches the naive reference.
        """
        rows = case
        high, low, close, volume = split_quads(rows)
        assert_matches(
            apply_accumulation_distribution_oscillator(high, low, close, volume),
            accumulation_distribution_oscillator_reference(high, low, close, volume, 2, 3),
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=ABSOLUTE_TOLERANCE_SCALE,
        )

    @given(
        case=_cases(coherent_hlcv()),
        exponent=st.sampled_from([-4, -3, -2, -1, 1, 2, 3, 4]),
    )
    def test_scale_homogeneity(
        self,
        case: list[tuple[float, float, float, float]],
        exponent: int,
    ) -> None:
        """
        Verifies that ``accumulation_distribution_oscillator`` is homogeneous of degree 1: scaling every input value
        by a constant ``k`` scales the output by the same ``k`` -- ``accumulation_distribution_oscillator(k * x) ==
        k * accumulation_distribution_oscillator(x)``. ``k`` is a power of two, so the rescale is exact and adds no
        floating-point error.
        """
        k = 2.0**exponent
        rows = case
        high, low, close, volume = split_quads(rows)
        base = apply_accumulation_distribution_oscillator(high, low, close, volume)
        scaled = apply_accumulation_distribution_oscillator(
            [value * k for value in high],
            [value * k for value in low],
            [value * k for value in close],
            [value * k for value in volume],
        )
        assert_scale_homogeneous(scaled, base, k=k, degree=1)

    @given(case=_cases(coherent_hlcv()))
    def test_warmup_null_count_property(
        self,
        case: list[tuple[float, float, float, float]],
    ) -> None:
        """
        Verifies that the leading-null run is exactly ``min(WARMUP, len(values))``.
        """
        rows = case
        high, low, close, volume = split_quads(rows)
        result = apply_accumulation_distribution_oscillator(high, low, close, volume)
        leading_nulls = count_leading_nulls(result)
        # NOTE: ``_cases`` floors the length at ``WARMUP + 1``, so ``min`` always resolves to ``WARMUP``; the form is
        # kept to state the exact warm-up rule (the leading-null run is never clamped by a too-short series here).
        assert leading_nulls == min(WARMUP, len(rows))

    @given(
        case=_cases(coherent_hlcv()),
        # The bar prices reach 1e3, so the scale tops out at 1e6 (price * scale <= 1e9): the oscillator is a difference
        # of two EMAs of a money-flow accumulation, and past ~1e9 that subtraction loses its few significant digits to
        # catastrophic cancellation, so the one-pass and two-pass forms diverge on the residual — a precision limit, not
        # a bug. 1e-9 still exercises the tiny end.
        scale=st.sampled_from([1e-9, 1e6]),
    )
    def test_matches_reference_at_large_magnitude(
        self,
        case: list[tuple[float, float, float, float]],
        scale: float,
    ) -> None:
        """
        Verifies that at extreme magnitudes the implementation stays finite where the reference is and agrees.
        """
        rows = case
        high = [row[0] * scale for row in rows]
        low = [row[1] * scale for row in rows]
        close = [row[2] * scale for row in rows]
        volume = [row[3] * scale for row in rows]
        assert_matches(
            apply_accumulation_distribution_oscillator(high, low, close, volume),
            accumulation_distribution_oscillator_reference(high, low, close, volume, 2, 3),
            rel_tol=RELATIVE_TOLERANCE_SCALE,
            abs_tol=ABSOLUTE_TOLERANCE_STREAMING,
        )
