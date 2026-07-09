"""
Tests for ``pomata.indicators.williams_r`` — Williams %R (Williams Percent Range).

``williams_r`` is multi-input (``high``, ``low``, ``close``), so this module cannot reuse the single-input
``apply_expr`` helper and instead defines a local ``apply_williams_r`` that builds the three-column ``Float64`` frame
inline. The shared ``assert_matches`` comparator and the naive ``williams_r_reference`` oracle are reused unchanged.

The property tests draw correlated bars from the ``well_formed_bar`` Hypothesis strategy — a per-bar low, a
strictly-positive high-low spread, and a close placed as a fraction of that spread — so every generated bar is
well-formed (``low <= close <= high``) and Hypothesis can shrink a failing example to a minimal counterexample.

Categories are split into classes; cross-cutting categories elsewhere use markers (see ``tests/README.md``).
"""

import math
from collections.abc import Sequence

import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st
from tests.indicators.oracles import williams_r_reference
from tests.support import (
    ABSOLUTE_TOLERANCE_EXACT,
    ABSOLUTE_TOLERANCE_REFERENCE,
    ABSOLUTE_TOLERANCE_SCALE,
    BOUND_MARGIN,
    CLOSE,
    GROUP_KEY,
    HIGH,
    LOW,
    RELATIVE_TOLERANCE_PROPERTY,
    RELATIVE_TOLERANCE_SCALE,
    WINDOW_MAX,
    assert_matches,
    assert_scale_homogeneous,
    coherent_hlc_with_missing,
    count_leading_nulls,
    materialize,
    split_triples,
)

from pomata.indicators import williams_r

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- derived, not chosen; the rationale is the shared method in CORRECTNESS.md, the numbers are this
# indicator's. To add an indicator, set its three facts here; the property tier below is then the same shape as every
# other indicator's.
#   1. warmup  W(window) = window - 1   (the rolling extremes emit only once ``window`` non-null bars have been seen)
#   2. memory  the oracle shares pomata's seeding, so the property holds from the first defined row (M = 0); each
#              example carries D in [window, 2 * window] defined bars -- one window of output, never all warm-up
#   3. domain  well_formed_bar(): a per-bar low, a strictly-positive high-low spread, and a close inside
#              ``[low, high]``, so the windowed range never collapses and the comparison stays off the ``0 / 0``
#              boundary (the genuine degenerate has its own pinned edge-case test); the missing-data tier draws from
#              coherent_hlc_with_missing. Windows span 1 .. WINDOW_MAX
# %R is a scale-INVARIANT bounded ratio (O(1) in ``[-100, 0]``), so the scale tier uses an ABSOLUTE tolerance, never
# ``input_scale``-sized, and the large-magnitude tier is vacuous (the common factor cancels) and absent.
# Repetitions N are the shared CI profile (tests/conftest.py); override per-test only if its parameter space is larger.
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


def apply_williams_r(
    high: Sequence[float | None],
    low: Sequence[float | None],
    close: Sequence[float | None],
    window: int,
) -> list[float | None]:
    """
    Materialize ``williams_r`` over a three-column ``Float64`` frame built from the aligned input lists.

    Args:
        high: The per-bar high observations (may contain ``None`` and ``float('nan')``).
        low: The per-bar low observations (may contain ``None`` and ``float('nan')``); same length as ``high``.
        close: The per-bar close observations (may contain ``None`` and ``float('nan')``); same length as ``high``.
        window: Number of observations in the moving window. Must be ``>= 1``.

    Returns:
        The materialized Williams %R as a Python list of the same length as the inputs, with ``None`` for ``null``
        entries.
    """
    return materialize(
        {HIGH: high, LOW: low, CLOSE: close}, williams_r(pl.col(HIGH), pl.col(LOW), pl.col(CLOSE), window)
    )


@st.composite
def well_formed_bar(draw: st.DrawFn, *, min_spread: float = 0.5) -> tuple[float, float, float]:
    """
    Draw one well-formed ``(high, low, close)`` bar (``low <= close <= high``) with a strictly-positive spread.

    The bar is built from a base ``low``, a strictly-positive high-low spread (``>= min_spread``, so the windowed range
    never collapses to zero), and a fraction in ``[0, 1]`` placing the ``close`` inside ``[low, high]``. Drawing each
    component independently lets Hypothesis shrink toward a minimal counterexample instead of an opaque seed.

    Args:
        draw: The Hypothesis draw callable supplied to a ``@st.composite`` strategy.
        min_spread: Minimum high-low spread, keeping the windowed range strictly positive.

    Returns:
        A ``(high, low, close)`` tuple satisfying ``low <= close <= high``.
    """
    low = draw(st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False))
    spread = draw(st.floats(min_value=min_spread, max_value=50.0, allow_nan=False, allow_infinity=False))
    fraction = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
    return low + spread, low, low + fraction * spread


class TestWilliamsRContract:
    """
    Type, shape, and lazy/eager guarantees.
    """

    def test_over_partitions_independently(self) -> None:
        """
        Verifies that under ``.over`` the window resets per group and never spans group boundaries.

        Group ``b`` is group ``a`` shifted up by a common ``+10``; %R is invariant to a common additive shift, so both
        groups produce identical per-row values, which is exactly what the reference oracle yields per group.
        """
        high_a = [10.0, 12.0, 11.0, 13.0]
        low_a = [8.0, 9.0, 10.0, 11.0]
        close_a = [9.0, 11.0, 10.5, 12.0]
        high_b = [value + 10.0 for value in high_a]
        low_b = [value + 10.0 for value in low_a]
        close_b = [value + 10.0 for value in close_a]
        frame = pl.DataFrame(
            {
                GROUP_KEY: ["a"] * 4 + ["b"] * 4,
                HIGH: high_a + high_b,
                LOW: low_a + low_b,
                CLOSE: close_a + close_b,
            }
        )
        result_over = frame.select(williams_r(pl.col(HIGH), pl.col(LOW), pl.col(CLOSE), 2).over(GROUP_KEY).alias("y"))[
            "y"
        ].to_list()
        expected = williams_r_reference(high_a, low_a, close_a, 2) + williams_r_reference(high_b, low_b, close_b, 2)
        assert_matches(result_over, expected)


class TestWilliamsREdge:
    """
    Boundaries, warm-up, null / NaN handling, and division-by-zero denominators.
    """

    def test_window_below_one_raises(self) -> None:
        """
        Verifies that ``window < 1`` raises ``ValueError``.
        """
        with pytest.raises(ValueError, match="window must be >= 1"):
            williams_r(pl.col(HIGH), pl.col(LOW), pl.col(CLOSE), 0)

    def test_single_row(self) -> None:
        """
        Verifies behavior on a one-element series.
        """
        assert_matches(apply_williams_r([10.0], [8.0], [9.0], 1), [-50.0])
        assert_matches(apply_williams_r([10.0], [8.0], [9.0], 2), [None])

    def test_all_null_series(self) -> None:
        """
        Verifies that an all-null input yields an all-null output.
        """
        assert_matches(
            apply_williams_r([None, None, None], [None, None, None], [None, None, None], 2),
            [None, None, None],
        )

    def test_null_in_high_propagates(self) -> None:
        """
        Verifies that a ``null`` in the high window yields ``null``.
        """
        high_values = [10.0, None, 11.0, 13.0]
        low_values = [8.0, 9.0, 10.0, 11.0]
        close_values = [9.0, 11.0, 10.5, 12.0]
        assert_matches(
            apply_williams_r(high_values, low_values, close_values, 2),
            williams_r_reference(high_values, low_values, close_values, 2),
        )

    def test_null_in_low_propagates(self) -> None:
        """
        Verifies that a ``null`` in the low window yields ``null``.
        """
        high_values = [10.0, 12.0, 11.0, 13.0]
        low_values = [8.0, None, 10.0, 11.0]
        close_values = [9.0, 11.0, 10.5, 12.0]
        assert_matches(
            apply_williams_r(high_values, low_values, close_values, 2),
            williams_r_reference(high_values, low_values, close_values, 2),
        )

    def test_null_in_close_propagates(self) -> None:
        """
        Verifies that a ``null`` in the current ``close`` yields ``null`` at that row.
        """
        high_values = [10.0, 12.0, 11.0, 13.0]
        low_values = [8.0, 9.0, 10.0, 11.0]
        close_values = [9.0, None, 10.5, 12.0]
        assert_matches(
            apply_williams_r(high_values, low_values, close_values, 2),
            williams_r_reference(high_values, low_values, close_values, 2),
        )

    def test_null_takes_precedence_over_nan(self) -> None:
        """
        Verifies that a window containing both a ``null`` and a ``NaN`` yields ``null`` (``null`` precedence).
        """
        high_values = [10.0, None, 11.0, 13.0]
        low_values = [8.0, math.nan, 10.0, 11.0]
        close_values = [9.0, 11.0, 10.5, 12.0]
        assert_matches(
            apply_williams_r(high_values, low_values, close_values, 2),
            williams_r_reference(high_values, low_values, close_values, 2),
        )

    def test_nan_in_high_propagates(self) -> None:
        """
        Verifies that a ``NaN`` in the high window yields ``NaN`` (no ``null`` present).
        """
        high_values = [10.0, math.nan, 11.0, 13.0]
        low_values = [8.0, 9.0, 10.0, 11.0]
        close_values = [9.0, 11.0, 10.5, 12.0]
        assert_matches(
            apply_williams_r(high_values, low_values, close_values, 2),
            williams_r_reference(high_values, low_values, close_values, 2),
        )

    def test_nan_in_close_propagates(self) -> None:
        """
        Verifies that a ``NaN`` in the current ``close`` yields ``NaN`` at that row (no ``null`` present).
        """
        high_values = [10.0, 12.0, 11.0, 13.0]
        low_values = [8.0, 9.0, 10.0, 11.0]
        close_values = [9.0, math.nan, 10.5, 12.0]
        assert_matches(
            apply_williams_r(high_values, low_values, close_values, 2),
            williams_r_reference(high_values, low_values, close_values, 2),
        )

    def test_warmup_null_count(self) -> None:
        """
        Verifies that the first ``window - 1`` rows are null and the first full window is defined.
        """
        result = apply_williams_r(
            [10.0, 12.0, 11.0, 13.0, 15.0],
            [8.0, 9.0, 10.0, 11.0, 12.0],
            [9.0, 11.0, 10.5, 12.0, 14.0],
            3,
        )
        assert result[:2] == [None, None]
        assert result[2] is not None

    def test_window_exceeds_length(self) -> None:
        """
        Verifies that a ``window`` larger than the series length yields an all-null output.
        """
        assert_matches(apply_williams_r([10.0, 12.0], [8.0, 9.0], [9.0, 11.0], 5), [None, None])

    def test_window_equals_length(self) -> None:
        """
        Verifies the single defined value when ``window`` equals the series length.
        """
        assert_matches(
            apply_williams_r([10.0, 12.0, 11.0], [8.0, 9.0, 10.0], [9.0, 11.0, 10.5], 3),
            williams_r_reference([10.0, 12.0, 11.0], [8.0, 9.0, 10.0], [9.0, 11.0, 10.5], 3),
        )

    def test_window_one_is_single_bar(self) -> None:
        """
        Verifies that ``window == 1`` collapses to the single bar: ``-100 * (high - close) / (high - low)``.
        """
        assert_matches(
            apply_williams_r([10.0, 12.0], [8.0, 9.0], [9.0, 11.0], 1),
            williams_r_reference([10.0, 12.0], [8.0, 9.0], [9.0, 11.0], 1),
        )

    def test_close_at_high_is_zero(self) -> None:
        """
        Verifies that a close at the windowed highest high gives ``%R == 0`` (top of the range, overbought).
        """
        result = apply_williams_r([10.0, 12.0, 11.0], [8.0, 9.0, 10.0], [10.0, 12.0, 12.0], 2)
        assert result[1] is not None
        assert result[2] is not None
        assert math.isclose(result[1], 0.0, abs_tol=ABSOLUTE_TOLERANCE_EXACT)
        assert math.isclose(result[2], 0.0, abs_tol=ABSOLUTE_TOLERANCE_EXACT)

    def test_close_at_low_is_minus_hundred(self) -> None:
        """
        Verifies that a close at the windowed lowest low gives ``%R == -100`` (bottom of the range, oversold).
        """
        result = apply_williams_r([10.0, 12.0, 11.0], [8.0, 9.0, 10.0], [8.0, 8.0, 9.0], 2)
        assert result[1] is not None
        assert result[2] is not None
        assert math.isclose(result[1], -100.0, abs_tol=ABSOLUTE_TOLERANCE_REFERENCE)
        assert math.isclose(result[2], -100.0, abs_tol=ABSOLUTE_TOLERANCE_REFERENCE)

    def test_constant_range_zero_over_zero_is_nan(self) -> None:
        """
        Verifies that a flat window (highest high == lowest low) with the close on that level is ``0 / 0 == NaN``.
        """
        assert_matches(
            apply_williams_r([5.0, 5.0, 5.0], [5.0, 5.0, 5.0], [5.0, 5.0, 5.0], 2),
            [None, math.nan, math.nan],
        )

    def test_constant_range_nonzero_numerator_is_inf(self) -> None:
        """
        Verifies that a flat window with the close off that level is a non-zero numerator over zero, i.e. ``+/-inf``.
        """
        result = apply_williams_r([5.0, 5.0], [5.0, 5.0], [3.0, 3.0], 2)
        assert result[0] is None
        assert result[1] is not None
        assert math.isinf(result[1])
        assert result[1] < 0.0

    def test_all_zero_series_is_nan(self) -> None:
        """
        Verifies that an all-zero high/low/close series collapses the range to zero and yields ``NaN`` (``0 / 0``).
        """
        assert_matches(
            apply_williams_r([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 2),
            [None, math.nan, math.nan],
        )

    def test_interior_null_propagates(self) -> None:
        """
        Verifies the interior-null case: a ``null`` taints exactly the windows that contain it.
        """
        high_values = [10.0, 12.0, None, 13.0, 15.0, 14.0]
        low_values = [8.0, 9.0, 10.0, 11.0, 12.0, 13.0]
        close_values = [9.0, 11.0, 10.5, 12.0, 14.0, 13.5]
        assert_matches(
            apply_williams_r(high_values, low_values, close_values, 2),
            williams_r_reference(high_values, low_values, close_values, 2),
        )

    def test_leading_and_trailing_nulls(self) -> None:
        """
        Verifies that leading and trailing nulls are handled exactly as the reference oracle prescribes.
        """
        high_values = [None, 12.0, 11.0, 13.0, None]
        low_values = [None, 9.0, 10.0, 11.0, None]
        close_values = [None, 11.0, 10.5, 12.0, None]
        assert_matches(
            apply_williams_r(high_values, low_values, close_values, 2),
            williams_r_reference(high_values, low_values, close_values, 2),
        )

    def test_all_nan(self) -> None:
        """
        Verifies that an all-NaN input yields ``NaN`` after the warm-up.
        """
        assert_matches(
            apply_williams_r([math.nan] * 3, [math.nan] * 3, [math.nan] * 3, 2),
            [None, math.nan, math.nan],
        )


class TestWilliamsRCorrectness:
    """
    Against the naive reference oracle and frozen golden-master values.
    """

    def test_matches_reference(self) -> None:
        """
        Verifies agreement with the naive closed-form reference across several windows.
        """
        high_values = [13.0, 11.0, 14.0, 11.0, 15.0, 19.0, 12.0, 16.0]
        low_values = [9.0, 8.0, 10.0, 7.0, 11.0, 13.0, 8.0, 12.0]
        close_values = [11.0, 9.0, 12.0, 9.0, 13.0, 17.0, 10.0, 14.0]
        for window in (1, 2, 3, 4, 5):
            assert_matches(
                apply_williams_r(high_values, low_values, close_values, window),
                williams_r_reference(high_values, low_values, close_values, window),
            )

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference: %R(window=3) over high [10, 12, 11, 13, 15, 14], low [8, 9, 10, 11, 12, 13],
        close [9, 11, 10.5, 12, 14, 13.5] == [None, None, -37.5, -25.0, -20.0, -37.5].
        """
        assert_matches(
            apply_williams_r(
                [10.0, 12.0, 11.0, 13.0, 15.0, 14.0],
                [8.0, 9.0, 10.0, 11.0, 12.0, 13.0],
                [9.0, 11.0, 10.5, 12.0, 14.0, 13.5],
                3,
            ),
            [None, None, -37.5, -25.0, -20.0, -37.5],
        )


class TestWilliamsRProperties:
    """
    Invariants that must hold for all inputs (property-based).
    """

    @given(case=_cases(well_formed_bar()))
    def test_matches_reference_for_any_input(self, case: tuple[list[tuple[float, float, float]], int]) -> None:
        """
        Verifies that, for any well-formed OHLC series and window, the implementation matches the naive reference.

        The ``well_formed_bar`` strategy draws a strictly-positive high-low spread, so the windowed range never
        collapses, keeping the comparison off the ``0 / 0`` boundary ``assert_matches`` would otherwise special-case.
        """
        rows, window = case
        high_values, low_values, close_values = split_triples(rows)
        assert_matches(
            apply_williams_r(high_values, low_values, close_values, window),
            williams_r_reference(high_values, low_values, close_values, window),
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    @given(case=_cases(coherent_hlc_with_missing()))
    def test_matches_reference_under_missing_data(
        self,
        case: tuple[list[tuple[float | None, float | None, float | None]], int],
    ) -> None:
        """
        Verifies that, for inputs freely mixing null / NaN / finite, the implementation matches the naive reference.
        """
        rows, window = case
        high, low, close = split_triples(rows)
        assert_matches(
            apply_williams_r(high, low, close, window),
            williams_r_reference(high, low, close, window),
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    @given(
        case=_cases(well_formed_bar()),
        exponent=st.sampled_from([-4, -3, -2, -1, 1, 2, 3, 4]),
    )
    def test_scale_invariance(
        self,
        case: tuple[list[tuple[float, float, float]], int],
        exponent: int,
    ) -> None:
        """
        Verifies that ``williams_r`` is scale-invariant: scaling every input value by a constant ``k`` leaves the
        output unchanged -- ``williams_r(k * x) == williams_r(x)``. ``k`` is a power of two, so the rescale is exact
        and adds no floating-point error.
        """
        k = 2.0**exponent
        rows, window = case
        high_values, low_values, close_values = split_triples(rows)
        result_base = apply_williams_r(high_values, low_values, close_values, window)
        result_scaled = apply_williams_r(
            [value * k for value in high_values],
            [value * k for value in low_values],
            [value * k for value in close_values],
            window,
        )
        assert_scale_homogeneous(result_scaled, result_base, k=k, degree=0)

    @given(
        case=_cases(well_formed_bar()),
        shift=st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False),
    )
    def test_additive_shift_invariance(
        self,
        case: tuple[list[tuple[float, float, float]], int],
        shift: float,
    ) -> None:
        """
        Verifies that ``williams_r`` is invariant to a common additive shift: adding the same constant to every
        input value leaves the output unchanged, because the shift cancels.
        """
        rows, window = case
        high_values, low_values, close_values = split_triples(rows)
        result_base = apply_williams_r(high_values, low_values, close_values, window)
        result_shifted = apply_williams_r(
            [value + shift for value in high_values],
            [value + shift for value in low_values],
            [value + shift for value in close_values],
            window,
        )
        assert_matches(result_shifted, result_base, rel_tol=RELATIVE_TOLERANCE_SCALE, abs_tol=ABSOLUTE_TOLERANCE_SCALE)

    @given(case=_cases(well_formed_bar()))
    def test_bounded(self, case: tuple[list[tuple[float, float, float]], int]) -> None:
        """
        Verifies that for well-formed bars (``low <= close <= high``) every defined %R lies within ``[-100, 0]``.
        """
        rows, window = case
        high_values, low_values, close_values = split_triples(rows)
        result = apply_williams_r(high_values, low_values, close_values, window)
        for value in result:
            if value is None:
                continue
            assert -100.0 - BOUND_MARGIN <= value <= BOUND_MARGIN

    @given(case=_cases(well_formed_bar()))
    def test_warmup_null_count_property(
        self,
        case: tuple[list[tuple[float, float, float]], int],
    ) -> None:
        """
        Verifies that the leading-null run is exactly ``min(window - 1, len(values))``.
        """
        rows, window = case
        high_values, low_values, close_values = split_triples(rows)
        result = apply_williams_r(high_values, low_values, close_values, window)
        leading_nulls = count_leading_nulls(result)
        # NOTE: ``_cases`` couples length >= window, so ``min`` always resolves to ``window - 1``; the form is kept to
        # state the exact warm-up rule (the leading-null run is never clamped by a too-short series here).
        assert leading_nulls == min(window - 1, len(high_values))
