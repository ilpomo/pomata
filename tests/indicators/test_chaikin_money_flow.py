"""
Tests for ``pomata.indicators.chaikin_money_flow`` — the Chaikin Money Flow (CMF).

The CMF is multi-input (``high``, ``low``, ``close``, ``volume``), so this module cannot reuse the single-input
``apply_expr`` helper and instead defines a local ``apply_chaikin_money_flow`` that builds the four-column ``Float64``
frame inline.
The shared ``assert_matches`` comparator and the naive ``chaikin_money_flow_reference`` oracle are reused unchanged.

Categories are split into classes; cross-cutting categories elsewhere use markers (see ``tests/README.md``).
"""

import math
from collections.abc import Sequence

import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st
from tests.indicators.oracles import chaikin_money_flow_reference
from tests.support import (
    ABSOLUTE_TOLERANCE_REFERENCE,
    BOUND_MARGIN,
    CLOSE,
    GROUP_KEY,
    HIGH,
    LOW,
    RELATIVE_TOLERANCE_PROPERTY,
    RELATIVE_TOLERANCE_REFERENCE,
    VOLUME,
    WINDOW_MAX,
    assert_matches,
    assert_scale_homogeneous,
    coherent_hlcv_with_missing,
    count_leading_nulls,
    materialize,
    split_quads,
)

from pomata.indicators import chaikin_money_flow

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- derived, not chosen; the rationale is the shared method in CORRECTNESS.md, the numbers are this
# indicator's. To add an indicator, set its three facts here; the property tier below is then the same shape as every
# other indicator's.
#   1. warmup  W(window) = window - 1   (the value is defined only once a full window of bars is available)
#   2. memory  the oracle shares pomata's windowed seeding, so the property holds from the first defined row (M = 0);
#              each example carries D in [window, 2 * window] defined bars -- one window of output, never all warm-up
#   3. domain  well_formed_bar(): a per-bar low, a strictly-positive high-low spread, a close inside ``[low, high]``,
#              and a strictly-positive volume, so neither the per-bar range nor the windowed total volume collapses and
#              the comparison stays off the ``0 / 0`` boundary (the genuine zero-range bar and zero-volume window have
#              their own pinned edge-case tests); the missing-data tier draws from coherent_hlcv_with_missing. Windows
#              span 1 .. WINDOW_MAX
# CMF is a scale-INVARIANT bounded ratio (O(1) in ``[-1, +1]``): its value is independent of both the common price scale
# and the common volume scale, so its scale tier uses an ABSOLUTE tolerance, never ``input_scale``-sized, and the
# large-magnitude tier is vacuous (the common factor cancels in the ratio) and absent. Repetitions N are the shared CI
# profile (tests/conftest.py); override per-test only if its parameter space is larger.
# ----------------------------------------------------------------------------------------------------------------------


@st.composite
def _cases[T](draw: st.DrawFn, bars: st.SearchStrategy[T]) -> tuple[list[T], int]:
    """
    A (series, window) pair sized from the facts above: ``window`` over its regimes, length = warm-up + a window of
    defined bars, so every example has output to check (never an all-warm-up series, the waste a ``window`` decoupled
    from the length would cause).
    """
    window = draw(st.integers(min_value=1, max_value=WINDOW_MAX))
    defined = draw(st.integers(min_value=window, max_value=2 * window))
    length = (window - 1) + defined
    return draw(st.lists(bars, min_size=length, max_size=length)), window


@st.composite
def well_formed_bar(draw: st.DrawFn, *, min_spread: float = 0.5) -> tuple[float, float, float, float]:
    """
    Draw one well-formed ``(high, low, close, volume)`` bar (``low <= close <= high``) with a strictly-positive spread
    and a strictly-positive volume.

    Mirrors the per-bar domain the property tier drew with ``random`` before the sizing sweep -- a base ``low`` in
    ``[1, 100]``, a high-low spread (``>= min_spread``, so the per-bar range never collapses to zero), a ``close``
    placed inside ``[low, high]``, and a strictly-positive volume in ``[1, 1e3]`` (so the windowed total volume never
    hits the ``0 / 0`` boundary, which the edge tests pin directly). Drawing each component independently lets
    Hypothesis shrink toward a minimal counterexample.

    Args:
        draw: The Hypothesis draw callable supplied to a ``@st.composite`` strategy.
        min_spread: Minimum high-low spread, keeping the per-bar range strictly positive.

    Returns:
        A ``(high, low, close, volume)`` tuple satisfying ``low <= close <= high`` with positive range and volume.
    """
    low = draw(st.floats(min_value=1.0, max_value=100.0, allow_nan=False, allow_infinity=False))
    spread = draw(st.floats(min_value=min_spread, max_value=50.0, allow_nan=False, allow_infinity=False))
    fraction = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
    volume = draw(st.floats(min_value=1.0, max_value=1e3, allow_nan=False, allow_infinity=False))
    return low + spread, low, low + fraction * spread, volume


def apply_chaikin_money_flow(
    high: Sequence[float | None],
    low: Sequence[float | None],
    close: Sequence[float | None],
    volume: Sequence[float | None],
    window: int,
) -> list[float | None]:
    """
    Materialize ``chaikin_money_flow`` over a four-column ``Float64`` frame built from the four input lists.

    Args:
        high: The per-bar high observations (may contain ``None`` and ``float('nan')``).
        low: The per-bar low observations (may contain ``None`` and ``float('nan')``); same length as ``high``.
        close: The per-bar close observations (may contain ``None`` and ``float('nan')``); same length as ``high``.
        volume: The per-bar volume observations (may contain ``None`` and ``float('nan')``); same length as ``high``.
        window: Number of observations in the moving window. Must be ``>= 1``.

    Returns:
        The materialized CMF as a Python list of the same length as the inputs, with ``None`` for ``null`` entries.
    """
    return materialize(
        {HIGH: high, LOW: low, CLOSE: close, VOLUME: volume},
        chaikin_money_flow(pl.col(HIGH), pl.col(LOW), pl.col(CLOSE), pl.col(VOLUME), window),
    )


class TestChaikinMoneyFlowContract:
    """
    Type, shape, and lazy/eager guarantees.
    """

    def test_over_partitions_independently(self) -> None:
        """
        Verifies that under ``.over`` both rolling sums reset per group and never span group boundaries.
        """
        frame = pl.DataFrame(
            {
                GROUP_KEY: ["a", "a", "a", "b", "b", "b"],
                HIGH: [10.0, 12.0, 11.0, 20.0, 22.0, 21.0],
                LOW: [8.0, 9.0, 9.0, 18.0, 19.0, 19.0],
                CLOSE: [9.0, 11.0, 10.0, 19.0, 21.0, 20.0],
                VOLUME: [100.0, 200.0, 150.0, 100.0, 200.0, 150.0],
            }
        )
        result = frame.select(
            chaikin_money_flow(pl.col(HIGH), pl.col(LOW), pl.col(CLOSE), pl.col(VOLUME), 2).over(GROUP_KEY).alias("y")
        )["y"].to_list()
        result_a = apply_chaikin_money_flow(
            [10.0, 12.0, 11.0], [8.0, 9.0, 9.0], [9.0, 11.0, 10.0], [100.0, 200.0, 150.0], 2
        )
        result_b = apply_chaikin_money_flow(
            [20.0, 22.0, 21.0], [18.0, 19.0, 19.0], [19.0, 21.0, 20.0], [100.0, 200.0, 150.0], 2
        )
        assert_matches(result, result_a + result_b)


class TestChaikinMoneyFlowEdge:
    """
    Boundaries, warm-up, and null / NaN handling.
    """

    def test_window_below_one_raises(self) -> None:
        """
        Verifies that ``window < 1`` raises ``ValueError``.
        """
        with pytest.raises(ValueError, match="window must be >= 1"):
            chaikin_money_flow(pl.col(HIGH), pl.col(LOW), pl.col(CLOSE), pl.col(VOLUME), 0)

    def test_warmup_null_count(self) -> None:
        """
        Verifies that the first ``window - 1`` rows are null and the first full window is defined.
        """
        result = apply_chaikin_money_flow(
            [10.0, 12.0, 11.0, 13.0, 14.0],
            [8.0, 9.0, 9.0, 10.0, 11.0],
            [9.0, 11.0, 10.0, 12.0, 13.0],
            [100.0, 200.0, 150.0, 300.0, 250.0],
            3,
        )
        assert result[:2] == [None, None]
        assert result[2] is not None

    def test_single_row(self) -> None:
        """
        Verifies behavior on a one-element series.
        """
        assert_matches(apply_chaikin_money_flow([10.0], [8.0], [10.0], [100.0], 1), [1.0])
        assert_matches(apply_chaikin_money_flow([10.0], [8.0], [10.0], [100.0], 3), [None])

    def test_window_equals_length(self) -> None:
        """
        Verifies the single defined value when ``window`` equals the series length.
        """
        high_values = [10.0, 12.0, 11.0]
        low_values = [8.0, 9.0, 9.0]
        close_values = [9.0, 11.0, 10.0]
        volume_values = [100.0, 200.0, 150.0]
        assert_matches(
            apply_chaikin_money_flow(high_values, low_values, close_values, volume_values, 3),
            chaikin_money_flow_reference(high_values, low_values, close_values, volume_values, 3),
        )

    def test_window_exceeds_length(self) -> None:
        """
        Verifies that a window longer than the series yields an all-null result (the warm-up never completes).
        """
        assert_matches(
            apply_chaikin_money_flow([10.0, 12.0, 11.0], [8.0, 9.0, 9.0], [9.0, 11.0, 10.0], [100.0, 200.0, 150.0], 5),
            [None, None, None],
        )

    def test_all_null(self) -> None:
        """
        Verifies that an all-null input yields all ``null``.
        """
        assert_matches(
            apply_chaikin_money_flow([None, None, None], [None, None, None], [None, None, None], [None, None, None], 2),
            [None, None, None],
        )

    def test_all_nan(self) -> None:
        """
        Verifies that an all-NaN input yields ``null`` during warm-up and ``NaN`` thereafter.
        """
        assert_matches(
            apply_chaikin_money_flow([math.nan] * 3, [math.nan] * 3, [math.nan] * 3, [math.nan] * 3, 2),
            [None, math.nan, math.nan],
        )

    def test_close_at_high_is_plus_one(self) -> None:
        """
        Verifies that a close printed at the high gives a multiplier of ``+1`` and hence a CMF of ``+1``.
        """
        assert_matches(
            apply_chaikin_money_flow([10.0, 12.0, 11.0], [8.0, 9.0, 9.0], [10.0, 12.0, 11.0], [100.0, 200.0, 150.0], 2),
            [None, 1.0, 1.0],
        )

    def test_close_at_low_is_minus_one(self) -> None:
        """
        Verifies that a close printed at the low gives a multiplier of ``-1`` and hence a CMF of ``-1``.
        """
        assert_matches(
            apply_chaikin_money_flow([10.0, 12.0, 11.0], [8.0, 9.0, 9.0], [8.0, 9.0, 9.0], [100.0, 200.0, 150.0], 2),
            [None, -1.0, -1.0],
        )

    def test_close_at_midpoint_is_zero(self) -> None:
        """
        Verifies that a close at the bar midpoint gives a multiplier of ``0`` and hence a CMF of ``0``.
        """
        assert_matches(
            apply_chaikin_money_flow([10.0, 10.0, 10.0], [8.0, 8.0, 8.0], [9.0, 9.0, 9.0], [100.0, 200.0, 150.0], 2),
            [None, 0.0, 0.0],
        )

    def test_zero_range_bar_contributes_zero_numerator(self) -> None:
        """
        Verifies that a zero-range bar (``high == low``) sets its multiplier to ``0`` while its volume still counts.

        On the second row ``high == low``, so its money-flow volume is ``0`` but its volume of ``200`` enters the
        denominator, matching the naive reference.
        """
        high_values = [10.0, 12.0, 13.0]
        low_values = [8.0, 12.0, 10.0]
        close_values = [9.0, 12.0, 12.0]
        volume_values = [100.0, 200.0, 150.0]
        assert_matches(
            apply_chaikin_money_flow(high_values, low_values, close_values, volume_values, 2),
            chaikin_money_flow_reference(high_values, low_values, close_values, volume_values, 2),
        )

    def test_zero_total_volume_is_nan(self) -> None:
        """
        Verifies that a window whose total volume is zero yields ``NaN`` (IEEE-754 ``0 / 0``).
        """
        assert_matches(
            apply_chaikin_money_flow([10.0, 12.0, 11.0], [8.0, 9.0, 9.0], [9.0, 11.0, 10.0], [0.0, 0.0, 0.0], 2),
            [None, math.nan, math.nan],
        )

    def test_zero_volume_after_large_volume_is_nan(self) -> None:
        """
        Verifies that an all-zero-volume window still yields ``NaN`` after large volumes have slid out of the window.

        Polars' rolling sum subtracts on exit, so once the leading ``1e16`` volume leaves the window the running total
        retains a sub-ULP residual instead of an exact zero; dividing by that residual would fake a finite (or infinite)
        reading at the final all-zero-volume window. The exact all-zero detection pins it to ``NaN`` as documented.
        """
        assert_matches(
            apply_chaikin_money_flow(
                [12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0],
                [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
                [11.0, 12.5, 13.5, 14.0, 15.5, 16.0, 17.5],
                [1e16, 0.1, 0.2, 0.3, 0.0, 0.0, 0.0],
                3,
            ),
            [None, None, 0.0, 0.25, 0.2, 0.0, math.nan],
        )

    def test_zero_volume_with_null_price_is_null(self) -> None:
        """
        Verifies that ``null`` keeps precedence over the zero-volume ``NaN``: a window that is both all-zero-volume and
        carries a ``null`` price input stays ``null`` (its money-flow-volume sum is null), not the guard's ``NaN``.
        """
        assert_matches(
            apply_chaikin_money_flow([None, 10.0, 11.0], [9.0, 9.0, 9.0], [9.5, 9.5, 9.5], [0.0, 0.0, 0.0], 2),
            [None, None, math.nan],
        )

    def test_null_in_high_propagates(self) -> None:
        """
        Verifies that a ``null`` in a ``high`` bar yields ``null`` for every window that contains it.
        """
        high_values = [10.0, None, 11.0, 13.0]
        low_values = [8.0, 9.0, 9.0, 10.0]
        close_values = [9.0, 11.0, 10.0, 12.0]
        volume_values = [100.0, 200.0, 150.0, 300.0]
        assert_matches(
            apply_chaikin_money_flow(high_values, low_values, close_values, volume_values, 2),
            chaikin_money_flow_reference(high_values, low_values, close_values, volume_values, 2),
        )

    def test_null_in_volume_propagates(self) -> None:
        """
        Verifies that a ``null`` in a ``volume`` bar yields ``null`` for every window that contains it.
        """
        high_values = [10.0, 12.0, 11.0, 13.0]
        low_values = [8.0, 9.0, 9.0, 10.0]
        close_values = [9.0, 11.0, 10.0, 12.0]
        volume_values = [100.0, None, 150.0, 300.0]
        assert_matches(
            apply_chaikin_money_flow(high_values, low_values, close_values, volume_values, 2),
            chaikin_money_flow_reference(high_values, low_values, close_values, volume_values, 2),
        )

    def test_nan_in_close_propagates(self) -> None:
        """
        Verifies that a ``NaN`` in a ``close`` bar yields ``NaN`` for every window that contains it (no ``null``).
        """
        high_values = [10.0, 12.0, 11.0, 13.0]
        low_values = [8.0, 9.0, 9.0, 10.0]
        close_values = [9.0, math.nan, 10.0, 12.0]
        volume_values = [100.0, 200.0, 150.0, 300.0]
        assert_matches(
            apply_chaikin_money_flow(high_values, low_values, close_values, volume_values, 2),
            chaikin_money_flow_reference(high_values, low_values, close_values, volume_values, 2),
        )

    def test_null_takes_precedence_over_nan(self) -> None:
        """
        Verifies that a window containing both a ``null`` and a ``NaN`` yields ``null`` (``null`` precedence).
        """
        high_values = [10.0, 12.0, 11.0]
        low_values = [8.0, 9.0, 9.0]
        close_values = [math.nan, 11.0, 10.0]
        volume_values = [100.0, None, 150.0]
        assert_matches(
            apply_chaikin_money_flow(high_values, low_values, close_values, volume_values, 3),
            chaikin_money_flow_reference(high_values, low_values, close_values, volume_values, 3),
        )


class TestChaikinMoneyFlowCorrectness:
    """
    Against the naive reference oracle and frozen golden-master values.
    """

    def test_matches_reference(self) -> None:
        """
        Verifies agreement with the naive closed-form reference across several windows.
        """
        high_values = [10.0, 12.0, 11.0, 13.0, 14.0, 15.0, 13.0, 16.0]
        low_values = [8.0, 9.0, 9.0, 10.0, 11.0, 12.0, 11.0, 13.0]
        close_values = [9.0, 11.0, 10.0, 12.0, 13.0, 14.0, 12.0, 15.0]
        volume_values = [100.0, 200.0, 150.0, 300.0, 250.0, 400.0, 350.0, 500.0]
        for window in (1, 2, 3, 4, 5):
            assert_matches(
                apply_chaikin_money_flow(high_values, low_values, close_values, volume_values, window),
                chaikin_money_flow_reference(high_values, low_values, close_values, volume_values, window),
            )

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference: CMF(window=3) over high [10, 12, 11, 13, 14], low [8, 9, 9, 10, 11],
        close [9, 11, 10, 12, 13], volume [100, 200, 150, 300, 250].
        """
        assert_matches(
            apply_chaikin_money_flow(
                [10.0, 12.0, 11.0, 13.0, 14.0],
                [8.0, 9.0, 9.0, 10.0, 11.0],
                [9.0, 11.0, 10.0, 12.0, 13.0],
                [100.0, 200.0, 150.0, 300.0, 250.0],
                3,
            ),
            [None, None, 0.14814814814814814, 0.2564102564102564, 0.26190476190476186],
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )


class TestChaikinMoneyFlowProperties:
    """
    Invariants that must hold for all inputs (property-based).
    """

    @given(case=_cases(well_formed_bar()))
    def test_matches_reference_for_any_input(
        self,
        case: tuple[list[tuple[float, float, float, float]], int],
    ) -> None:
        """
        Verifies that, for any well-formed OHLCV panel with positive volume and window, the implementation matches
        the naive reference. Volume is drawn strictly positive to avoid the ``0 / 0`` boundary, which is exercised
        directly in the edge tests.
        """
        rows, window = case
        high_values, low_values, close_values, volume_values = split_quads(rows)
        assert_matches(
            apply_chaikin_money_flow(high_values, low_values, close_values, volume_values, window),
            chaikin_money_flow_reference(high_values, low_values, close_values, volume_values, window),
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    @given(case=_cases(well_formed_bar()))
    def test_bounded_in_unit_interval(
        self,
        case: tuple[list[tuple[float, float, float, float]], int],
    ) -> None:
        """
        Verifies that with well-formed bars (``low <= close <= high``) and positive volume the CMF lies in
        ``[-1, +1]``, since it is a volume-weighted average of multipliers that are each in ``[-1, +1]``.
        """
        rows, window = case
        high_values, low_values, close_values, volume_values = split_quads(rows)
        result = apply_chaikin_money_flow(high_values, low_values, close_values, volume_values, window)
        for value in result:
            if value is None:
                continue
            assert -1.0 - BOUND_MARGIN <= value <= 1.0 + BOUND_MARGIN

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
            apply_chaikin_money_flow(high, low, close, volume, window),
            chaikin_money_flow_reference(high, low, close, volume, window),
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    @given(
        case=_cases(well_formed_bar()),
        exponent=st.sampled_from([-4, -3, -2, -1, 1, 2, 3, 4]),
    )
    def test_price_scale_invariance(
        self,
        case: tuple[list[tuple[float, float, float, float]], int],
        exponent: int,
    ) -> None:
        """
        Verifies that ``chaikin_money_flow`` is invariant to a price rescale: scaling the price inputs by a constant
        ``k`` leaves the output unchanged, while the volume is untouched. ``k`` is a power of two, so the rescale is
        exact and adds no floating-point error.
        """
        k = 2.0**exponent
        rows, window = case
        high_values, low_values, close_values, volume_values = split_quads(rows)
        result_base = apply_chaikin_money_flow(high_values, low_values, close_values, volume_values, window)
        result_scaled = apply_chaikin_money_flow(
            [value * k for value in high_values],
            [value * k for value in low_values],
            [value * k for value in close_values],
            volume_values,
            window,
        )
        assert_scale_homogeneous(result_scaled, result_base, k=k, degree=0)

    @given(
        case=_cases(well_formed_bar()),
        exponent=st.sampled_from([-4, -3, -2, -1, 1, 2, 3, 4]),
    )
    def test_volume_scale_invariance(
        self,
        case: tuple[list[tuple[float, float, float, float]], int],
        exponent: int,
    ) -> None:
        """
        Verifies that ``chaikin_money_flow`` is invariant to a volume rescale: scaling the volume by a constant
        ``k`` leaves the output unchanged, while the prices are untouched. ``k`` is a power of two, so the rescale
        is exact and adds no floating-point error.
        """
        c = 2.0**exponent
        rows, window = case
        high_values, low_values, close_values, volume_values = split_quads(rows)
        result_base = apply_chaikin_money_flow(high_values, low_values, close_values, volume_values, window)
        result_scaled = apply_chaikin_money_flow(
            high_values, low_values, close_values, [value * c for value in volume_values], window
        )
        assert_scale_homogeneous(result_scaled, result_base, k=c, degree=0)

    @given(case=_cases(well_formed_bar()))
    def test_warmup_null_count_property(
        self,
        case: tuple[list[tuple[float, float, float, float]], int],
    ) -> None:
        """
        Verifies that the leading-null run is exactly ``min(window - 1, len(values))``.
        """
        rows, window = case
        high_values, low_values, close_values, volume_values = split_quads(rows)
        result = apply_chaikin_money_flow(high_values, low_values, close_values, volume_values, window)
        leading_nulls = count_leading_nulls(result)
        # NOTE: ``_cases`` couples length >= window, so ``min`` always resolves to ``window - 1``; the form is kept to
        # state the exact warm-up rule (the leading-null run is never clamped by a too-short series here).
        assert leading_nulls == min(window - 1, len(rows))
