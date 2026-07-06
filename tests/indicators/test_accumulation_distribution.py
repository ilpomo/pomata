"""
Tests for ``pomata.indicators.accumulation_distribution`` — the Accumulation/Distribution Line.

``accumulation_distribution`` is multi-input (high, low, close, volume), so tests build the four-column frame inline
rather than using the single-input ``apply_expr`` helper; ``assert_matches`` and the naive
``accumulation_distribution_reference`` oracle are shared.
"""

import math
from collections.abc import Sequence

import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st
from tests.indicators.oracles import accumulation_distribution_reference
from tests.support import (
    CLOSE,
    EXACT_TOLERANCE_FACTOR,
    GROUP_KEY,
    HIGH,
    LOW,
    RELATIVE_TOLERANCE_PROPERTY,
    RELATIVE_TOLERANCE_REFERENCE,
    RELATIVE_TOLERANCE_SCALE,
    VOLUME,
    assert_matches,
    assert_scale_homogeneous,
    coherent_hlcv,
    coherent_hlcv_with_missing,
    input_scale,
    materialize,
    split_quads,
)

from pomata.indicators import accumulation_distribution

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- derived, not chosen; the rationale is the shared method in CORRECTNESS.md, the numbers are this
# indicator's. To add an indicator, set its three facts here; the property tier below is then the same shape as every
# other indicator's.
#   1. warmup  W = 0   (windowless cumulative line: the first row already carries the first bar's Money Flow Volume and
#              every later row is the running sum from there, so every row is defined)
#   2. memory  the oracle shares pomata's running cumulative sum, so the property holds from row 0 (M = 0); with W = 0
#              there is no warm-up to outlast, so a case is simply a series of bars -- every row is output
#   3. domain  coherent_hlcv(): coherent (high >= low, low <= close <= high) positive-finite bars (doji bars included)
#              for the finite tiers; the missing-data tier draws coherent_hlcv_with_missing. SERIES_MAX bars span
#              several total sizes
# The line is homogeneous of degree 1 in volume (the multiplier is price-invariant), so it carries a degree-1
# volume-scale-homogeneity property and a large-magnitude tier. accumulation_distribution has no window parameter, so
# the case builders draw only the series length (no window to couple). Repetitions N are the shared CI profile
# (tests/conftest.py); override per-test only if its parameter space is larger.
# ----------------------------------------------------------------------------------------------------------------------
SERIES_MAX = 60


@st.composite
def _bar_cases[T](draw: st.DrawFn, bars: st.SearchStrategy[T]) -> list[T]:
    """
    A series of bars sized from the facts above. accumulation_distribution is windowless (W = 0), so a case is just the
    series -- every row is output, never warm-up.
    """
    return draw(st.lists(bars, min_size=1, max_size=SERIES_MAX))


def apply_accumulation_distribution(
    high_values: Sequence[float | None],
    low_values: Sequence[float | None],
    close_values: Sequence[float | None],
    volume_values: Sequence[float | None],
) -> list[float | None]:
    """
    Materialize ``accumulation_distribution`` over a four-column ``Float64`` frame built from the four input lists.
    """
    return materialize(
        {HIGH: high_values, LOW: low_values, CLOSE: close_values, VOLUME: volume_values},
        accumulation_distribution(pl.col(HIGH), pl.col(LOW), pl.col(CLOSE), pl.col(VOLUME)),
    )


class TestAccumulationDistributionContract:
    """
    Type, shape, and lazy/eager guarantees.
    """

    def test_over_partitions_independently(self) -> None:
        """
        Verifies that under ``.over`` the cumulative sum restarts per group and never spans group boundaries.
        """
        high_values = [10.0, 11.0, 12.0]
        low_values = [8.0, 9.0, 10.0]
        close_values = [9.0, 10.5, 10.0]
        volume_values = [100.0, 200.0, 300.0]
        frame = pl.DataFrame(
            {
                GROUP_KEY: ["a", "a", "a", "b", "b", "b"],
                HIGH: high_values + high_values,
                LOW: low_values + low_values,
                CLOSE: close_values + close_values,
                VOLUME: volume_values + volume_values,
            }
        )
        expr = accumulation_distribution(pl.col(HIGH), pl.col(LOW), pl.col(CLOSE), pl.col(VOLUME))
        result_over = frame.select(expr.over(GROUP_KEY).alias("y"))["y"].to_list()
        result_group = apply_accumulation_distribution(high_values, low_values, close_values, volume_values)
        assert_matches(result_over, result_group + result_group)


class TestAccumulationDistributionEdge:
    """
    Boundaries, doji bars, and null / NaN handling.
    """

    def test_unequal_lengths_raise_in_reference(self) -> None:
        """
        Verifies that the reference oracle rejects inputs of differing length (a guard the impl does not need).
        """
        with pytest.raises(ValueError, match="high, low, close and volume must have equal length"):
            accumulation_distribution_reference([1.0, 2.0], [1.0], [1.0, 2.0], [1.0, 2.0])

    def test_no_warmup(self) -> None:
        """
        Verifies that there is no warm-up: the first row already carries the first bar's Money Flow Volume.
        """
        result = apply_accumulation_distribution([10.0, 11.0], [8.0, 9.0], [9.0, 10.5], [100.0, 200.0])
        assert result[0] is not None

    def test_single_row(self) -> None:
        """
        Verifies behavior on a one-element series.
        """
        assert_matches(apply_accumulation_distribution([10.0], [8.0], [9.0], [100.0]), [0.0])
        assert_matches(apply_accumulation_distribution([12.0], [8.0], [11.0], [100.0]), [50.0])

    def test_all_null(self) -> None:
        """
        Verifies that an all-null input yields an all-null output.
        """
        assert_matches(
            apply_accumulation_distribution(
                [None, None, None], [None, None, None], [None, None, None], [None, None, None]
            ),
            [None, None, None],
        )

    def test_close_at_high_is_plus_volume(self) -> None:
        """
        Verifies that a bar closing at its high contributes ``+volume`` (multiplier ``+1``).
        """
        assert_matches(apply_accumulation_distribution([10.0], [8.0], [10.0], [250.0]), [250.0])

    def test_close_at_low_is_minus_volume(self) -> None:
        """
        Verifies that a bar closing at its low contributes ``-volume`` (multiplier ``-1``).
        """
        assert_matches(apply_accumulation_distribution([10.0], [8.0], [8.0], [250.0]), [-250.0])

    def test_doji_bar_contributes_zero(self) -> None:
        """
        Verifies that a doji bar (``high == low``) contributes ``0`` rather than an undefined ``0 / 0``.
        """
        assert_matches(
            apply_accumulation_distribution([5.0, 5.0, 5.0], [5.0, 5.0, 5.0], [5.0, 5.0, 5.0], [100.0, 200.0, 300.0]),
            [0.0, 0.0, 0.0],
        )

    def test_doji_bar_ignores_close(self) -> None:
        """
        Verifies that on a doji bar the multiplier is ``0`` regardless of ``close`` (even a ``null`` ``close``).
        """
        assert_matches(
            apply_accumulation_distribution([5.0, 5.0], [5.0, 5.0], [99.0, None], [100.0, 200.0]), [0.0, 0.0]
        )

    def test_doji_bar_with_null_volume_is_null(self) -> None:
        """
        Verifies that a doji bar with ``null`` volume yields ``null`` (volume still multiplies the zero multiplier),
        while a following finite bar resumes the cumulative sum.
        """
        assert_matches(apply_accumulation_distribution([5.0, 6.0], [5.0, 4.0], [5.0, 5.0], [None, 100.0]), [None, 0.0])
        assert apply_accumulation_distribution([5.0], [5.0], [5.0], [None]) == [None]

    def test_null_in_high_propagates(self) -> None:
        """
        Verifies that a ``null`` high yields ``null`` at that bar while the running total is carried across it.
        """
        assert_matches(
            apply_accumulation_distribution(
                [10.0, None, 12.0, 13.0], [8.0, 9.0, 10.0, 11.0], [10.0, 10.5, 10.0, 13.0], [100.0, 200.0, 300.0, 400.0]
            ),
            accumulation_distribution_reference(
                [10.0, None, 12.0, 13.0], [8.0, 9.0, 10.0, 11.0], [10.0, 10.5, 10.0, 13.0], [100.0, 200.0, 300.0, 400.0]
            ),
        )

    def test_nan_high_and_low_poisons(self) -> None:
        """
        Verifies that a ``high == low == NaN`` bar does **not** take the doji branch (``NaN - NaN`` is ``NaN``, never
        ``== 0``), so the ``NaN`` reaches the cumulative sum and poisons the line rather than contributing ``0``. A
        ``null`` volume still voids that bar to ``null`` (``NaN * null`` is ``null``).
        """
        assert_matches(apply_accumulation_distribution([math.nan], [math.nan], [5.0], [100.0]), [math.nan])
        assert_matches(apply_accumulation_distribution([math.nan], [math.nan], [5.0], [math.nan]), [math.nan])
        assert apply_accumulation_distribution([math.nan], [math.nan], [5.0], [None]) == [None]
        assert_matches(
            apply_accumulation_distribution([math.nan], [math.nan], [5.0], [100.0]),
            accumulation_distribution_reference([math.nan], [math.nan], [5.0], [100.0]),
        )
        assert_matches(
            apply_accumulation_distribution([math.nan], [math.nan], [5.0], [math.nan]),
            accumulation_distribution_reference([math.nan], [math.nan], [5.0], [math.nan]),
        )

    def test_null_in_volume_propagates(self) -> None:
        """
        Verifies that a ``null`` volume yields ``null`` at that bar while the running total is carried across it.
        """
        assert_matches(
            apply_accumulation_distribution(
                [10.0, 11.0, 12.0, 13.0], [8.0, 9.0, 10.0, 11.0], [9.0, 10.0, 10.0, 13.0], [100.0, None, 300.0, 400.0]
            ),
            accumulation_distribution_reference(
                [10.0, 11.0, 12.0, 13.0], [8.0, 9.0, 10.0, 11.0], [9.0, 10.0, 10.0, 13.0], [100.0, None, 300.0, 400.0]
            ),
        )

    def test_interior_null_carries_running_total(self) -> None:
        """
        Verifies that an interior ``null`` yields ``null`` at its position while the cumulative sum bridges the gap.
        """
        assert_matches(
            apply_accumulation_distribution(
                [10.0, 11.0, 12.0, 13.0], [8.0, 9.0, 10.0, 11.0], [9.0, None, 10.0, 13.0], [100.0, 200.0, 300.0, 400.0]
            ),
            [0.0, None, -300.0, 100.0],
        )

    def test_nan_latches(self) -> None:
        """
        Verifies that a ``NaN`` reaching the cumulative sum latches and every later non-null row is ``NaN``.
        """
        assert_matches(
            apply_accumulation_distribution(
                [10.0, 11.0, 12.0, 13.0],
                [8.0, 9.0, 10.0, 11.0],
                [9.0, math.nan, 10.0, 13.0],
                [100.0, 200.0, 300.0, 400.0],
            ),
            [0.0, math.nan, math.nan, math.nan],
        )

    def test_nan_in_volume_latches(self) -> None:
        """
        Verifies that a ``NaN`` volume contributes ``NaN`` and latches the cumulative sum.
        """
        assert_matches(
            apply_accumulation_distribution(
                [10.0, 11.0, 12.0], [8.0, 9.0, 10.0], [9.0, 10.5, 10.0], [100.0, math.nan, 300.0]
            ),
            [0.0, math.nan, math.nan],
        )


class TestAccumulationDistributionCorrectness:
    """
    Against the naive reference oracle and frozen golden-master values.
    """

    def test_matches_reference(self) -> None:
        """
        Verifies agreement with the naive closed-form reference on a representative OHLCV series.
        """
        high_values = [10.0, 11.5, 12.0, 13.2, 14.0, 13.0, 15.0, 16.0]
        low_values = [8.0, 9.0, 10.5, 11.0, 12.0, 11.5, 13.0, 14.5]
        close_values = [9.0, 11.0, 10.7, 13.0, 12.5, 12.0, 14.8, 15.0]
        volume_values = [100.0, 250.0, 300.0, 175.0, 500.0, 220.0, 410.0, 330.0]
        assert_matches(
            apply_accumulation_distribution(high_values, low_values, close_values, volume_values),
            accumulation_distribution_reference(high_values, low_values, close_values, volume_values),
        )

    def test_matches_reference_with_doji_and_gaps(self) -> None:
        """
        Verifies agreement with the reference on a series containing doji bars, nulls, and a NaN.
        """
        high_values = [10.0, 12.0, 12.0, None, 14.0, 13.0]
        low_values = [8.0, 12.0, 10.0, 11.0, 12.0, 13.0]
        close_values = [9.0, 12.0, 11.5, 11.5, math.nan, 13.0]
        volume_values = [100.0, 200.0, 300.0, 400.0, 500.0, 600.0]
        assert_matches(
            apply_accumulation_distribution(high_values, low_values, close_values, volume_values),
            accumulation_distribution_reference(high_values, low_values, close_values, volume_values),
        )

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference: AD over high [10..14], low [8..12], close [9, 10.5, 10, 13, 12.5],
        volume [100, 200, 300, 400, 500] == [0, 100, -200, 200, -50].
        """
        assert_matches(
            apply_accumulation_distribution(
                [10.0, 11.0, 12.0, 13.0, 14.0],
                [8.0, 9.0, 10.0, 11.0, 12.0],
                [9.0, 10.5, 10.0, 13.0, 12.5],
                [100.0, 200.0, 300.0, 400.0, 500.0],
            ),
            [0.0, 100.0, -200.0, 200.0, -50.0],
        )


class TestAccumulationDistributionProperties:
    """
    Invariants that must hold for all inputs (property-based).
    """

    @given(case=_bar_cases(coherent_hlcv()))
    def test_matches_reference_for_any_input(
        self,
        case: list[tuple[float, float, float, float]],
    ) -> None:
        """
        Verifies that, for any coherent OHLCV series (including the doji bars coherent draws yield), the implementation
        matches the naive reference; the null / NaN sprinkle is covered by the missing-data tier.
        """
        high_values, low_values, close_values, volume_values = split_quads(case)
        assert_matches(
            apply_accumulation_distribution(high_values, low_values, close_values, volume_values),
            accumulation_distribution_reference(high_values, low_values, close_values, volume_values),
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
            abs_tol=input_scale(volume_values) * EXACT_TOLERANCE_FACTOR,
        )

    @given(case=_bar_cases(coherent_hlcv()))
    def test_is_running_difference_of_consecutive_terms(
        self,
        case: list[tuple[float, float, float, float]],
    ) -> None:
        """
        Verifies that consecutive line differences equal each bar's Money Flow Volume (the line is a cumulative sum).
        """
        high_values, low_values, close_values, volume_values = split_quads(case)
        line = apply_accumulation_distribution(high_values, low_values, close_values, volume_values)
        # the line is a cumulative sum, so differencing two large partial sums loses precision proportional to its
        # magnitude (catastrophic cancellation) -- size the floor to the line, not a flat absolute constant.
        tolerance = input_scale(line) * EXACT_TOLERANCE_FACTOR
        for index, value in enumerate(line):
            assert value is not None
            previous = 0.0 if index == 0 else line[index - 1]
            assert previous is not None
            high = high_values[index]
            low = low_values[index]
            close = close_values[index]
            volume = volume_values[index]
            multiplier = 0.0 if high == low else ((close - low) - (high - close)) / (high - low)
            expected_step = multiplier * volume
            assert math.isclose(
                value - previous,
                expected_step,
                rel_tol=RELATIVE_TOLERANCE_PROPERTY,
                abs_tol=tolerance,
            )

    @given(
        case=_bar_cases(coherent_hlcv()),
        exponent=st.sampled_from([-4, -3, -2, -1, 1, 2, 3, 4]),
    )
    def test_volume_scale_homogeneity(
        self,
        case: list[tuple[float, float, float, float]],
        exponent: int,
    ) -> None:
        """
        Verifies that ``accumulation_distribution`` is homogeneous of degree 1 in volume: scaling the volume by a
        constant ``k`` scales the output by the same ``k``, while the prices are untouched. ``k`` is a power of two,
        so the rescale is exact and adds no floating-point error.
        """
        c = 2.0**exponent
        high_values, low_values, close_values, volume_values = split_quads(case)
        scaled_volume_values = [volume * c for volume in volume_values]
        result_base = apply_accumulation_distribution(high_values, low_values, close_values, volume_values)
        result_scaled = apply_accumulation_distribution(high_values, low_values, close_values, scaled_volume_values)
        assert_scale_homogeneous(result_scaled, result_base, k=c, degree=1)

    @given(case=_bar_cases(coherent_hlcv_with_missing()))
    def test_matches_reference_under_missing_data(
        self,
        case: list[tuple[float | None, float | None, float | None, float | None]],
    ) -> None:
        """
        Verifies that, for inputs freely mixing null / NaN / finite, the implementation matches the naive reference.
        """
        high, low, close, volume = split_quads(case)
        assert_matches(
            apply_accumulation_distribution(high, low, close, volume),
            accumulation_distribution_reference(high, low, close, volume),
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=input_scale(volume) * EXACT_TOLERANCE_FACTOR,
        )

    @given(
        case=_bar_cases(coherent_hlcv()),
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
        high_base, low_base, close_base, volume_base = split_quads(case)
        high = [value * scale for value in high_base]
        low = [value * scale for value in low_base]
        close = [value * scale for value in close_base]
        volume = [value * scale for value in volume_base]
        assert_matches(
            apply_accumulation_distribution(high, low, close, volume),
            accumulation_distribution_reference(high, low, close, volume),
            rel_tol=RELATIVE_TOLERANCE_SCALE,
            abs_tol=input_scale(volume) * EXACT_TOLERANCE_FACTOR,
        )
