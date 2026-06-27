"""
Tests for ``pomata.indicators.money_flow_index`` — the Money Flow Index (MFI).

The MFI is multi-input (``high``, ``low``, ``close``, ``volume``), so this module cannot reuse the single-input
``apply_expr`` helper and instead defines a local ``apply_money_flow_index`` that builds the four-column ``Float64``
frame inline. The shared ``assert_matches`` comparator and the naive ``money_flow_index_reference`` oracle are reused
unchanged.

Categories are split into classes; cross-cutting categories elsewhere use markers (see ``tests/README.md``).
"""

import math
from collections.abc import Sequence

import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st
from polars.testing import assert_frame_equal
from tests.indicators.oracles import money_flow_index_reference
from tests.support import (
    ABSOLUTE_TOLERANCE_REFERENCE,
    BOUND_MARGIN,
    CLOSE,
    GROUP_KEY,
    HIGH,
    LOW,
    RELATIVE_TOLERANCE_PROPERTY,
    VOLUME,
    WINDOW_MAX,
    assert_matches,
    assert_scale_homogeneous,
    coherent_hlcv,
    coherent_hlcv_with_missing,
    count_leading_nulls,
    materialize,
    split_quads,
)

from pomata.indicators import money_flow_index

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- derived, not chosen; the rationale is the shared method in CORRECTNESS.md, the numbers are this
# indicator's. To add an indicator, set its three facts here; the property tier below is then the same shape as every
# other indicator's.
#   1. warmup  W(window) = window   (a full ``window`` of typical-price *changes*, hence ``window + 1`` bars, is needed
#              before the first value is defined, so the first defined row is at index ``window``, not ``window - 1``)
#   2. memory  the oracle shares pomata's seeding, so the property holds from the first defined row (M = 0); each
#              example carries D in [window, 2 * window] defined bars -- a window of output, never all warm-up
#   3. domain  coherent_hlcv(): coherent (high >= low, low <= close <= high) positive-finite bars; a flat window gives
#              the ``0 / 0`` money ratio (NaN, its own pinned edge-case test covers it; the scale tiers match NaN to
#              NaN); the missing-data tier draws coherent_hlcv_with_missing. Windows span 1 .. WINDOW_MAX
# The MFI is a scale-INVARIANT bounded ratio (O(1) in ``[0, 100]``, in both price and volume), so the scale tiers use an
# ABSOLUTE tolerance, never ``input_scale``-sized, and the large-magnitude tier is vacuous (the common factor cancels)
# and absent. Repetitions N are the shared CI profile (tests/conftest.py); override per-test only if its parameter space
# is larger.
# ----------------------------------------------------------------------------------------------------------------------


@st.composite
def _cases[T](draw: st.DrawFn, bars: st.SearchStrategy[T]) -> tuple[list[T], int]:
    """
    A (series, window) pair sized from the facts above: ``window`` over its regimes, length = warm-up + a window of
    defined bars, so every example has output to check (never an all-warm-up series).
    """
    window = draw(st.integers(min_value=1, max_value=WINDOW_MAX))
    defined = draw(st.integers(min_value=window, max_value=2 * window))
    length = window + defined
    return draw(st.lists(bars, min_size=length, max_size=length)), window


def apply_money_flow_index(
    high: Sequence[float | None],
    low: Sequence[float | None],
    close: Sequence[float | None],
    volume: Sequence[float | None],
    window: int,
) -> list[float | None]:
    """
    Materialize ``money_flow_index`` over a four-column ``Float64`` frame built from the four input lists.

    Args:
        high: The per-bar high observations (may contain ``None`` and ``float('nan')``).
        low: The per-bar low observations (may contain ``None`` and ``float('nan')``); same length as ``high``.
        close: The per-bar close observations (may contain ``None`` and ``float('nan')``); same length as ``high``.
        volume: The per-bar volume observations (may contain ``None`` and ``float('nan')``); same length as ``high``.
        window: Number of observations in the moving window. Must be ``>= 1``.

    Returns:
        The materialized MFI as a Python list of the same length as the inputs, with ``None`` for ``null`` entries.
    """
    return materialize(
        {HIGH: high, LOW: low, CLOSE: close, VOLUME: volume},
        money_flow_index(pl.col(HIGH), pl.col(LOW), pl.col(CLOSE), pl.col(VOLUME), window),
    )


class TestMoneyFlowIndexContract:
    """
    Type, shape, and lazy/eager guarantees.
    """

    def test_returns_expr(self) -> None:
        """
        Verifies that the factory returns a ``pl.Expr`` without touching a frame.
        """
        assert isinstance(money_flow_index(pl.col(HIGH), pl.col(LOW), pl.col(CLOSE), pl.col(VOLUME), 3), pl.Expr)

    def test_preserves_length_and_dtype(self) -> None:
        """
        Verifies that the output has one value per input row and is ``Float64``.
        """
        frame = pl.DataFrame(
            {
                HIGH: pl.Series(HIGH, [10.0, 11.0, 12.0, 11.0, 13.0]),
                LOW: pl.Series(LOW, [8.0, 9.0, 10.0, 9.0, 11.0]),
                CLOSE: pl.Series(CLOSE, [9.0, 10.0, 11.0, 10.0, 12.0]),
                VOLUME: pl.Series(VOLUME, [100.0, 150.0, 120.0, 130.0, 110.0]),
            }
        )
        result = frame.select(money_flow_index(pl.col(HIGH), pl.col(LOW), pl.col(CLOSE), pl.col(VOLUME), 3).alias("y"))
        assert result.height == frame.height
        assert result.schema["y"] == pl.Float64

    def test_lazy_eager_parity(self) -> None:
        """
        Verifies that eager and lazy application produce identical materialized output.
        """
        frame = pl.DataFrame(
            {
                HIGH: pl.Series(HIGH, [10.0, 11.0, 12.0, 11.0, 13.0, 14.0]),
                LOW: pl.Series(LOW, [8.0, 9.0, 10.0, 9.0, 11.0, 12.0]),
                CLOSE: pl.Series(CLOSE, [9.0, 10.0, 11.0, 10.0, 12.0, 13.0]),
                VOLUME: pl.Series(VOLUME, [100.0, 150.0, 120.0, 130.0, 110.0, 160.0]),
            }
        )
        expr = money_flow_index(pl.col(HIGH), pl.col(LOW), pl.col(CLOSE), pl.col(VOLUME), 3).alias("y")
        result_eager = frame.select(expr)
        result_lazy = frame.lazy().select(expr).collect()
        assert_frame_equal(result_eager, result_lazy)

    def test_over_partitions_independently(self) -> None:
        """
        Verifies that under ``.over`` the difference and rolling sums reset per group and never span group boundaries.
        """
        high_a = [10.0, 11.0, 12.0, 11.0, 13.0]
        low_a = [8.0, 9.0, 10.0, 9.0, 11.0]
        close_a = [9.0, 10.0, 11.0, 10.0, 12.0]
        volume_a = [100.0, 150.0, 120.0, 130.0, 110.0]
        high_b = [20.0, 21.0, 22.0, 21.0, 23.0]
        low_b = [18.0, 19.0, 20.0, 19.0, 21.0]
        close_b = [19.0, 20.0, 21.0, 20.0, 22.0]
        volume_b = [200.0, 250.0, 220.0, 230.0, 210.0]
        frame = pl.DataFrame(
            {
                GROUP_KEY: ["a"] * 5 + ["b"] * 5,
                HIGH: high_a + high_b,
                LOW: low_a + low_b,
                CLOSE: close_a + close_b,
                VOLUME: volume_a + volume_b,
            }
        )
        result_over = frame.select(
            money_flow_index(pl.col(HIGH), pl.col(LOW), pl.col(CLOSE), pl.col(VOLUME), 2).over(GROUP_KEY).alias("y")
        )["y"].to_list()
        result_a = apply_money_flow_index(high_a, low_a, close_a, volume_a, 2)
        result_b = apply_money_flow_index(high_b, low_b, close_b, volume_b, 2)
        assert_matches(result_over, result_a + result_b)


class TestMoneyFlowIndexEdge:
    """
    Boundaries, warm-up, and null / NaN handling.
    """

    def test_window_below_one_raises(self) -> None:
        """
        Verifies that ``window < 1`` raises ``ValueError``.
        """
        with pytest.raises(ValueError, match="window must be >= 1"):
            money_flow_index(pl.col(HIGH), pl.col(LOW), pl.col(CLOSE), pl.col(VOLUME), 0)

    def test_empty(self) -> None:
        """
        Verifies that an empty input yields an empty output.
        """
        assert_matches(apply_money_flow_index([], [], [], [], 3), [])

    def test_warmup_null_count(self) -> None:
        """
        Verifies that the first ``window`` rows are null (a full window of changes is needed) and the next is defined.
        """
        result = apply_money_flow_index(
            [10.0, 11.0, 12.0, 11.0, 13.0, 14.0],
            [8.0, 9.0, 10.0, 9.0, 11.0, 12.0],
            [9.0, 10.0, 11.0, 10.0, 12.0, 13.0],
            [100.0, 150.0, 120.0, 130.0, 110.0, 160.0],
            3,
        )
        assert result[:3] == [None, None, None]
        assert result[3] is not None

    def test_window_one(self) -> None:
        """
        Verifies that with ``window == 1`` the warm-up is a single row and each bar is either fully up or fully down.
        """
        assert_matches(
            apply_money_flow_index(
                [10.0, 11.0, 12.0, 11.0, 13.0],
                [8.0, 9.0, 10.0, 9.0, 11.0],
                [9.0, 10.0, 11.0, 10.0, 12.0],
                [100.0, 150.0, 120.0, 130.0, 110.0],
                1,
            ),
            [None, 100.0, 100.0, 0.0, 100.0],
        )

    def test_single_row(self) -> None:
        """
        Verifies behavior on a one-element series: with no predecessor every window is unfilled.
        """
        assert_matches(apply_money_flow_index([10.0], [8.0], [9.0], [100.0], 1), [None])

    def test_window_equals_length(self) -> None:
        """
        Verifies that when ``window`` equals the series length the whole output is null (no full window of changes).
        """
        assert_matches(
            apply_money_flow_index([10.0, 11.0, 12.0], [8.0, 9.0, 10.0], [9.0, 10.0, 11.0], [100.0, 150.0, 120.0], 3),
            [None, None, None],
        )

    def test_window_exceeds_length(self) -> None:
        """
        Verifies that a window longer than the series yields an all-null result (no full window of changes accrues).
        """
        assert_matches(
            apply_money_flow_index([10.0, 11.0, 12.0], [8.0, 9.0, 10.0], [9.0, 10.0, 11.0], [100.0, 150.0, 120.0], 5),
            [None, None, None],
        )

    def test_all_up_saturates_at_one_hundred(self) -> None:
        """
        Verifies that a strictly rising typical price (zero negative flow) saturates the MFI at ``100``.
        """
        assert_matches(
            apply_money_flow_index(
                [10.0, 11.0, 12.0, 13.0, 14.0],
                [8.0, 9.0, 10.0, 11.0, 12.0],
                [9.0, 10.0, 11.0, 12.0, 13.0],
                [100.0, 150.0, 120.0, 130.0, 110.0],
                3,
            ),
            [None, None, None, 100.0, 100.0],
        )

    def test_all_down_saturates_at_zero(self) -> None:
        """
        Verifies that a strictly falling typical price (zero positive flow) saturates the MFI at ``0``.
        """
        assert_matches(
            apply_money_flow_index(
                [14.0, 13.0, 12.0, 11.0, 10.0],
                [12.0, 11.0, 10.0, 9.0, 8.0],
                [13.0, 12.0, 11.0, 10.0, 9.0],
                [100.0, 150.0, 120.0, 130.0, 110.0],
                3,
            ),
            [None, None, None, 0.0, 0.0],
        )

    def test_constant_typical_is_nan(self) -> None:
        """
        Verifies that a window whose typical price never moves leaves the money ratio at ``0 / 0`` and yields ``NaN``.
        """
        assert_matches(
            apply_money_flow_index([10.0] * 5, [8.0] * 5, [9.0] * 5, [100.0] * 5, 3),
            [None, None, None, math.nan, math.nan],
        )

    def test_huge_then_dust_flow_stays_bounded(self) -> None:
        """
        Verifies the conditioning-limit bound contract: when the money flow spans a vast dynamic range within a window
        (a ``1e6``-scale typical price with unit volume, then a dust ``0.1``-scale price with ``1e-9`` volume), the
        streaming positive / negative flow sums are residual-dominated and the raw ratio escapes ``[0, 100]`` (it
        reached the order of ``1e4`` on this input). The clip keeps every value inside ``[0, 100]``: past a sane
        dynamic range the value degrades but never escapes the documented bound (see ``CORRECTNESS.md``).
        """
        typical = [
            1e6,
            1.1e6,
            0.9e6,
            1.05e6,
            0.95e6,
            1e6,
            0.1,
            0.11,
            0.09,
            0.12,
            0.1,
            0.1,
            0.11,
            0.1,
            0.09,
            0.1,
            0.11,
            0.1,
        ]
        volume = [1.0, 1.1, 0.9, 1.0, 1.0, 1.0, 1e-9, 2e-9, 1e-9, 3e-9, 1e-9, 2e-9, 1e-9, 1e-9, 2e-9, 1e-9, 1e-9, 1e-9]
        result = apply_money_flow_index(typical, typical, typical, volume, 4)
        finite = [value for value in result if value is not None and not math.isnan(value)]
        assert finite
        assert all(0.0 <= value <= 100.0 for value in finite)

    def test_flat_tail_after_movement_is_nan(self) -> None:
        """
        Verifies that a window which goes flat *after* large movement yields ``NaN``, the realistic degenerate the
        all-flat-from-row-0 case above cannot reach.

        Polars' rolling sum subtracts on exit, so once the large head money flows leave the window the positive and
        negative running totals retain a sub-ULP residual instead of the exact zero a fresh sum gives; the unguarded
        ratio would then read a saturated ``100.0`` on a market that has gone flat. The exact all-zero-change detection
        pins it to ``NaN`` and to full agreement with the oracle.
        """
        flat = [505132.00716615457, 15188.555761361386, 164208.86117163018, 356680.25052010204] + [
            679933.0366393882
        ] * 17
        volume = [
            6.746136702216398,
            3.02323049755392,
            4.84189237444706,
            8.637905388412495,
            5.251431067040438,
            3.867220889222082,
            6.740822776141895,
            8.783112777381728,
            7.2749252676535425,
            9.822690113975282,
            4.990696483661105,
            1.130210325158168,
            1.3420527757945298,
            9.74637597183996,
            4.286237946769904,
            8.004804172700894,
            9.735629456515243,
            5.934211531687549,
            7.060284229333455,
            6.05871898893224,
            7.565414383394959,
        ]
        result = apply_money_flow_index(flat, flat, flat, volume, 12)
        assert_matches(result, money_flow_index_reference(flat, flat, flat, volume, 12))
        assert all(value is not None and math.isnan(value) for value in result[16:])

    def test_null_in_price_propagates(self) -> None:
        """
        Verifies that an interior ``null`` in a price column voids the typical price at that row and the next change.
        """
        high_values = [10.0, 11.0, 12.0, 11.0, 13.0]
        low_values = [8.0, 9.0, 10.0, 9.0, 11.0]
        close_values = [9.0, 10.0, None, 10.0, 12.0]
        volume_values = [100.0, 150.0, 120.0, 130.0, 110.0]
        assert_matches(
            apply_money_flow_index(high_values, low_values, close_values, volume_values, 2),
            money_flow_index_reference(high_values, low_values, close_values, volume_values, 2),
        )

    def test_null_in_volume_voids_only_its_row(self) -> None:
        """
        Verifies that a ``null`` volume voids only that row's money flow while the typical-price difference survives.
        """
        high_values = [10.0, 11.0, 12.0, 11.0, 13.0]
        low_values = [8.0, 9.0, 10.0, 9.0, 11.0]
        close_values = [9.0, 10.0, 11.0, 10.0, 12.0]
        volume_values = [100.0, 150.0, None, 130.0, 110.0]
        assert_matches(
            apply_money_flow_index(high_values, low_values, close_values, volume_values, 2),
            money_flow_index_reference(high_values, low_values, close_values, volume_values, 2),
        )

    def test_nan_in_price_propagates(self) -> None:
        """
        Verifies that a ``NaN`` in a price column contaminates the affected money flow and yields ``NaN``.
        """
        high_values = [10.0, 11.0, math.nan, 11.0, 13.0]
        low_values = [8.0, 9.0, 10.0, 9.0, 11.0]
        close_values = [9.0, 10.0, 11.0, 10.0, 12.0]
        volume_values = [100.0, 150.0, 120.0, 130.0, 110.0]
        assert_matches(
            apply_money_flow_index(high_values, low_values, close_values, volume_values, 2),
            money_flow_index_reference(high_values, low_values, close_values, volume_values, 2),
        )

    def test_nan_in_volume_propagates(self) -> None:
        """
        Verifies that a ``NaN`` volume contaminates that row's money flow and yields ``NaN`` for windows reaching it.
        """
        high_values = [10.0, 11.0, 12.0, 11.0, 13.0]
        low_values = [8.0, 9.0, 10.0, 9.0, 11.0]
        close_values = [9.0, 10.0, 11.0, 10.0, 12.0]
        volume_values = [100.0, 150.0, math.nan, 130.0, 110.0]
        assert_matches(
            apply_money_flow_index(high_values, low_values, close_values, volume_values, 2),
            money_flow_index_reference(high_values, low_values, close_values, volume_values, 2),
        )

    def test_nan_typical_price_poisons_successor_change(self) -> None:
        """
        Verifies that a ``NaN`` typical price poisons both its own change and the next one into ``NaN``.

        The bar after a ``NaN`` typical price has a ``NaN`` typical change, undefined in sign; it is routed to ``NaN``
        in both the positive and the negative flow rather than classified as a fully-positive bar at its finite money
        flow, so every window reaching either change yields ``NaN``. The series falls 19 -> (NaN bar) -> 11 -> 10, so
        with ``window == 1`` the own change (index 1) and the successor change (index 2) are both ``NaN`` while the
        final clean down-change (index 3) gives ``0``.
        """
        high_values = [20.0, math.nan, 12.0, 11.0]
        low_values = [18.0, 9.0, 10.0, 9.0]
        close_values = [19.0, 10.0, 11.0, 10.0]
        volume_values = [100.0, 100.0, 100.0, 100.0]
        assert_matches(
            apply_money_flow_index(high_values, low_values, close_values, volume_values, 1),
            [None, math.nan, math.nan, 0.0],
        )

    def test_all_null_is_all_null(self) -> None:
        """
        Verifies that an all-null input yields an all-null output.
        """
        assert_matches(
            apply_money_flow_index([None] * 4, [None] * 4, [None] * 4, [None] * 4, 2),
            [None, None, None, None],
        )

    def test_all_nan_is_all_null_then_nan(self) -> None:
        """
        Verifies the all-NaN input: the warm-up row stays null and the remaining defined rows are ``NaN``.
        """
        assert_matches(
            apply_money_flow_index([math.nan] * 4, [math.nan] * 4, [math.nan] * 4, [math.nan] * 4, 2),
            money_flow_index_reference([math.nan] * 4, [math.nan] * 4, [math.nan] * 4, [math.nan] * 4, 2),
        )

    def test_leading_null_defers_warmup(self) -> None:
        """
        Verifies that a leading ``null`` defers the first usable change and extends the null run.
        """
        high_values = [None, 11.0, 12.0, 13.0, 14.0, 15.0]
        low_values = [None, 9.0, 10.0, 11.0, 12.0, 13.0]
        close_values = [None, 10.0, 11.0, 12.0, 13.0, 14.0]
        volume_values = [None, 150.0, 120.0, 130.0, 110.0, 160.0]
        assert_matches(
            apply_money_flow_index(high_values, low_values, close_values, volume_values, 2),
            money_flow_index_reference(high_values, low_values, close_values, volume_values, 2),
        )


class TestMoneyFlowIndexCorrectness:
    """
    Against the naive reference oracle and frozen golden-master values.
    """

    def test_matches_reference(self) -> None:
        """
        Verifies agreement with the naive closed-form reference across several windows.
        """
        high_values = [11.0, 12.0, 13.0, 12.0, 14.0, 15.0, 14.0, 16.0]
        low_values = [9.0, 10.0, 11.0, 10.0, 12.0, 13.0, 12.0, 14.0]
        close_values = [10.0, 11.0, 12.0, 11.0, 13.0, 14.0, 13.0, 15.0]
        volume_values = [100.0, 150.0, 120.0, 130.0, 110.0, 160.0, 140.0, 170.0]
        for window in (1, 2, 3, 4, 5):
            assert_matches(
                apply_money_flow_index(high_values, low_values, close_values, volume_values, window),
                money_flow_index_reference(high_values, low_values, close_values, volume_values, window),
            )

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference: MFI(window=3) over the documented high/low/close/volume bars.
        """
        assert_matches(
            apply_money_flow_index(
                [10.0, 11.0, 12.0, 11.0, 13.0, 14.0, 13.0, 15.0],
                [8.0, 9.0, 10.0, 9.0, 11.0, 12.0, 11.0, 13.0],
                [9.0, 10.0, 11.0, 10.0, 12.0, 13.0, 12.0, 14.0],
                [100.0, 150.0, 120.0, 130.0, 110.0, 160.0, 140.0, 170.0],
                3,
            ),
            [
                None,
                None,
                None,
                68.44660194174757,
                67.00507614213197,
                72.34042553191489,
                66.92913385826772,
                72.63843648208469,
            ],
        )


class TestMoneyFlowIndexProperties:
    """
    Invariants that must hold for all inputs (property-based).
    """

    @given(case=_cases(coherent_hlcv()))
    def test_matches_reference_for_any_input(
        self,
        case: tuple[list[tuple[float, float, float, float]], int],
    ) -> None:
        """
        Verifies that, for any high/low/close/volume series and window, the implementation matches the naive reference.

        Bars are drawn so high >= low and volume is strictly positive (the in-spec regime); the reference and the
        implementation are compared with a small tolerance to absorb float-summation order differences.
        """
        rows, window = case
        high_values, low_values, close_values, volume_values = split_quads(rows)
        assert_matches(
            apply_money_flow_index(high_values, low_values, close_values, volume_values, window),
            money_flow_index_reference(high_values, low_values, close_values, volume_values, window),
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    @given(case=_cases(coherent_hlcv()))
    def test_bounded_in_zero_one_hundred(
        self,
        case: tuple[list[tuple[float, float, float, float]], int],
    ) -> None:
        """
        Verifies that on clean, in-spec data every defined MFI value lies within ``[0, 100]``.
        """
        rows, window = case
        high_values, low_values, close_values, volume_values = split_quads(rows)
        result = apply_money_flow_index(high_values, low_values, close_values, volume_values, window)
        for value in result:
            if value is None or math.isnan(value):
                continue
            assert -BOUND_MARGIN <= value <= 100.0 + BOUND_MARGIN

    @given(
        case=_cases(coherent_hlcv()),
        exponent=st.sampled_from([-4, -3, -2, -1, 1, 2, 3, 4]),
    )
    def test_price_scale_invariance(
        self,
        case: tuple[list[tuple[float, float, float, float]], int],
        exponent: int,
    ) -> None:
        """
        Verifies that a positive rescaling of all prices leaves the MFI unchanged (the up/down ratio is scale-free).
        """
        k = 2.0**exponent
        rows, window = case
        high_values, low_values, close_values, volume_values = split_quads(rows)
        result_base = apply_money_flow_index(high_values, low_values, close_values, volume_values, window)
        result_scaled = apply_money_flow_index(
            [value * k for value in high_values],
            [value * k for value in low_values],
            [value * k for value in close_values],
            volume_values,
            window,
        )
        assert_scale_homogeneous(result_scaled, result_base, k=k, degree=0)

    @given(
        case=_cases(coherent_hlcv()),
        exponent=st.sampled_from([-4, -3, -2, -1, 1, 2, 3, 4]),
    )
    def test_volume_scale_invariance(
        self,
        case: tuple[list[tuple[float, float, float, float]], int],
        exponent: int,
    ) -> None:
        """
        Verifies that a positive global rescaling of volume leaves the MFI unchanged (both flows scale together).
        """
        c = 2.0**exponent
        rows, window = case
        high_values, low_values, close_values, volume_values = split_quads(rows)
        result_base = apply_money_flow_index(high_values, low_values, close_values, volume_values, window)
        result_scaled = apply_money_flow_index(
            high_values, low_values, close_values, [value * c for value in volume_values], window
        )
        assert_scale_homogeneous(result_scaled, result_base, k=c, degree=0)

    @given(case=_cases(coherent_hlcv()))
    def test_warmup_null_count_property(
        self,
        case: tuple[list[tuple[float, float, float, float]], int],
    ) -> None:
        """
        Verifies that on clean data the leading-null run is exactly ``min(window, len(values))``.
        """
        rows, window = case
        high_values, low_values, close_values, volume_values = split_quads(rows)
        result = apply_money_flow_index(high_values, low_values, close_values, volume_values, window)
        leading_nulls = count_leading_nulls(result)
        # NOTE: ``_cases`` couples length > window, so ``min`` always resolves to ``window``; the form is kept to state
        # the exact warm-up rule (the leading-null run is never clamped by a too-short series here).
        assert leading_nulls == min(window, len(rows))

    @given(case=_cases(coherent_hlcv_with_missing()))
    def test_matches_reference_under_missing_data(
        self,
        case: tuple[list[tuple[float | None, float | None, float | None, float | None]], int],
    ) -> None:
        """
        Verifies that, for inputs freely mixing null / NaN / finite, the implementation matches the naive reference.
        """
        rows, window = case
        high = [high_value for high_value, _, _, _ in rows]
        low = [low_value for _, low_value, _, _ in rows]
        close = [close_value for _, _, close_value, _ in rows]
        volume = [volume_value for _, _, _, volume_value in rows]
        assert_matches(
            apply_money_flow_index(high, low, close, volume, window),
            money_flow_index_reference(high, low, close, volume, window),
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )
