"""
Tests for ``pomata.indicators.cci`` — the Commodity Channel Index.

``cci`` is multi-input (high, low, close), so these tests build the three-column frame inline rather than reusing the
single-input ``apply_expr`` helper; the ``assert_matches`` comparator and the naive ``cci_reference`` oracle are shared
with the rest of the suite. Categories are split into classes; cross-cutting categories elsewhere use markers (see
``tests/README.md``).
"""

import math
from collections.abc import Sequence

import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st
from tests.indicators.oracles import cci_reference
from tests.support import (
    ABSOLUTE_TOLERANCE_REFERENCE,
    CLOSE,
    GROUP_KEY,
    HIGH,
    LOW,
    RELATIVE_TOLERANCE_PROPERTY,
    RELATIVE_TOLERANCE_REFERENCE,
    WINDOW_MAX,
    assert_matches,
    assert_scale_homogeneous,
    coherent_hlc,
    coherent_hlc_with_missing,
    count_leading_nulls,
    materialize,
    split_triples,
)

from pomata.indicators import cci

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- derived, not chosen; the rationale is the shared method in CORRECTNESS.md, the numbers are this
# indicator's. To add an indicator, set its three facts here; the property tier below is then the same shape as every
# other indicator's.
#   1. warmup  W(window) = window - 1   (the index is the sma of the typical price, null until the window holds
#              ``window`` non-null typical prices)
#   2. memory  the oracle shares pomata's seeding, so the property holds from the first defined row (M = 0); each
#              example carries D in [window, 2 * window] defined bars -- one window of output, never all warm-up
#   3. domain  coherent_hlc(): coherent (high >= low, low <= close <= high) positive-finite bars; the strictly-positive
#              spread keeps the mean deviation off the ``0 / 0`` boundary (its own pinned edge-case test covers the flat
#              window). Windows span 1 .. WINDOW_MAX
# CCI is a scale-INVARIANT ratio (O(1) whatever the price magnitude), so the scale tier uses an ABSOLUTE tolerance,
# never ``input_scale``-sized, and the large-magnitude tier is vacuous (the common factor cancels) and absent.
# Repetitions N are the shared CI profile (tests/conftest.py); override per-test only if its parameter space is larger.
# ----------------------------------------------------------------------------------------------------------------------


@st.composite
def _cases[T](draw: st.DrawFn, bars: st.SearchStrategy[T], window_min: int = 1) -> tuple[list[T], int]:
    """
    A (series, window) pair sized from the facts above: ``window`` over its regimes, length = warm-up + a window of
    defined bars, so every example has output to check (never an all-warm-up series). ``window_min`` lets the scale tier
    exclude the ``window == 1`` degenerate, where every one-bar window is trivially flat (``0 / 0``).
    """
    window = draw(st.integers(min_value=window_min, max_value=WINDOW_MAX))
    defined = draw(st.integers(min_value=window, max_value=2 * window))
    length = (window - 1) + defined
    return draw(st.lists(bars, min_size=length, max_size=length)), window


def apply_cci(
    high: Sequence[float | None],
    low: Sequence[float | None],
    close: Sequence[float | None],
    window: int,
) -> list[float | None]:
    """
    Materialize the CCI expression over a three-column ``Float64`` frame built from the ``high``, ``low``, and ``close``
    lists.
    """
    return materialize({HIGH: high, LOW: low, CLOSE: close}, cci(pl.col(HIGH), pl.col(LOW), pl.col(CLOSE), window))


class TestCciContract:
    """
    Type, shape, and lazy/eager guarantees.
    """

    def test_over_partitions_independently(self) -> None:
        """
        Verifies that under ``.over`` the rolling mean and the shifts reset per group and never span group boundaries.

        Group ``b`` is group ``a`` rescaled by a positive factor, so by scale-invariance the two groups yield identical
        CCI values, confirming both partition independence and that no window leaks across the boundary.
        """
        frame = pl.DataFrame(
            {
                GROUP_KEY: ["a", "a", "a", "a", "b", "b", "b", "b"],
                HIGH: [10.0, 12.0, 11.0, 13.0, 20.0, 24.0, 22.0, 26.0],
                LOW: [8.0, 9.0, 9.0, 10.0, 16.0, 18.0, 18.0, 20.0],
                CLOSE: [9.0, 11.0, 10.0, 12.0, 18.0, 22.0, 20.0, 24.0],
            }
        )
        result = frame.select(cci(pl.col(HIGH), pl.col(LOW), pl.col(CLOSE), 3).over(GROUP_KEY).alias("y"))[
            "y"
        ].to_list()
        assert_matches(
            result,
            [None, None, 12.500000000000153, 100.0000000000001, None, None, 12.500000000000153, 100.0000000000001],
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )


class TestCciEdge:
    """
    Boundaries, warm-up, and null / NaN handling.
    """

    def test_window_below_one_raises(self) -> None:
        """
        Verifies that ``window < 1`` raises ``ValueError``.
        """
        with pytest.raises(ValueError, match="window must be >= 1"):
            cci(pl.col(HIGH), pl.col(LOW), pl.col(CLOSE), 0)

    def test_warmup_null_count(self) -> None:
        """
        Verifies that the first ``window - 1`` rows are ``null`` (warm-up) and the first full window is defined.
        """
        result = apply_cci(
            [10.0, 12.0, 11.0, 13.0, 15.0], [8.0, 9.0, 9.0, 10.0, 12.0], [9.0, 11.0, 10.0, 12.0, 14.0], 3
        )
        assert result[:2] == [None, None]
        assert result[2] is not None

    def test_window_one_is_nan(self) -> None:
        """
        Verifies that ``window == 1`` yields ``NaN`` everywhere: the typical price equals its own one-point mean, so the
        numerator and the mean deviation are both zero (the ``0 / 0`` denominator boundary).
        """
        assert_matches(
            apply_cci([10.0, 12.0, 11.0], [8.0, 9.0, 9.0], [9.0, 11.0, 10.0], 1),
            [math.nan, math.nan, math.nan],
        )

    def test_window_equals_length(self) -> None:
        """
        Verifies the single defined value when ``window`` equals the series length.
        """
        high_values = [10.0, 12.0, 11.0]
        low_values = [8.0, 9.0, 9.0]
        close_values = [9.0, 11.0, 10.0]
        assert_matches(
            apply_cci(high_values, low_values, close_values, 3),
            cci_reference(high_values, low_values, close_values, 3),
        )

    def test_window_exceeds_length(self) -> None:
        """
        Verifies that a window longer than the series leaves every position in warm-up (all ``null``).
        """
        assert_matches(apply_cci([10.0, 12.0], [8.0, 9.0], [9.0, 11.0], 5), [None, None])

    def test_single_row(self) -> None:
        """
        Verifies behavior on a one-element series.
        """
        assert_matches(apply_cci([10.0], [8.0], [9.0], 1), [math.nan])
        assert_matches(apply_cci([10.0], [8.0], [9.0], 3), [None])

    def test_constant_series_is_nan(self) -> None:
        """
        Verifies that a constant series gives a zero mean deviation, so every defined value is ``NaN`` (the ``0 / 0``
        denominator).
        """
        assert_matches(
            apply_cci([10.0] * 5, [8.0] * 5, [9.0] * 5, 3),
            [None, None, math.nan, math.nan, math.nan],
        )

    def test_all_zero_is_nan(self) -> None:
        """
        Verifies that an all-zero series gives a zero mean deviation with an exact-zero numerator, so every defined
        value is ``NaN`` (``0 / 0``) at the exact-zero denominator boundary.
        """
        assert_matches(
            apply_cci([0.0] * 5, [0.0] * 5, [0.0] * 5, 3),
            [None, None, math.nan, math.nan, math.nan],
        )

    def test_all_null(self) -> None:
        """
        Verifies that an all-null series yields all ``null``.
        """
        assert_matches(apply_cci([None, None, None], [None, None, None], [None, None, None], 2), [None, None, None])

    def test_all_nan(self) -> None:
        """
        Verifies that an all-NaN series yields ``null`` during warm-up then ``NaN``.
        """
        assert_matches(
            apply_cci([math.nan] * 3, [math.nan] * 3, [math.nan] * 3, 2),
            [None, math.nan, math.nan],
        )

    def test_null_in_window_is_null(self) -> None:
        """
        Verifies that a ``null`` in any leg taints exactly the windows (and shifts) that reach it.
        """
        high_values = [10.0, 12.0, 11.0, 13.0]
        low_values = [8.0, 9.0, 9.0, 10.0]
        close_values = [9.0, None, 10.0, 12.0]
        assert_matches(
            apply_cci(high_values, low_values, close_values, 2),
            cci_reference(high_values, low_values, close_values, 2),
        )
        assert_matches(apply_cci(high_values, low_values, close_values, 2), [None, None, None, 66.66666666666674])

    def test_interior_null_propagates(self) -> None:
        """
        Verifies that an interior ``null`` yields ``null`` for the windows it reaches and the result recovers after.
        """
        high_values = [10.0, 12.0, 11.0, 13.0, 15.0, 14.0]
        low_values = [8.0, 9.0, 9.0, 10.0, 12.0, 11.0]
        close_values = [9.0, 11.0, None, 12.0, 14.0, 12.0]
        assert_matches(
            apply_cci(high_values, low_values, close_values, 2),
            cci_reference(high_values, low_values, close_values, 2),
        )

    def test_nan_propagates(self) -> None:
        """
        Verifies that a ``NaN`` in any leg yields ``NaN`` for exactly the windows it reaches (no ``null`` present).
        """
        high_values = [10.0, 12.0, math.nan, 13.0, 15.0]
        low_values = [8.0, 9.0, 9.0, 10.0, 12.0]
        close_values = [9.0, 11.0, 10.0, 12.0, 14.0]
        assert_matches(
            apply_cci(high_values, low_values, close_values, 2),
            cci_reference(high_values, low_values, close_values, 2),
        )
        assert_matches(
            apply_cci(high_values, low_values, close_values, 2),
            [None, 66.66666666666674, math.nan, math.nan, 66.66666666666667],
        )

    def test_null_takes_precedence_over_nan(self) -> None:
        """
        Verifies that a window reaching both a ``null`` and a ``NaN`` yields ``null`` (``null`` takes precedence).
        """
        high_values = [10.0, 12.0, math.nan, 13.0]
        low_values = [8.0, None, 9.0, 10.0]
        close_values = [9.0, 11.0, 10.0, 12.0]
        assert_matches(
            apply_cci(high_values, low_values, close_values, 3),
            cci_reference(high_values, low_values, close_values, 3),
        )


class TestCciCorrectness:
    """
    Against the naive reference oracle and frozen golden-master values.
    """

    def test_matches_reference(self) -> None:
        """
        Verifies agreement with the naive closed-form reference across several windows.
        """
        high_values = [10.0, 12.0, 11.0, 13.0, 15.0, 14.0, 16.0, 18.0]
        low_values = [8.0, 9.0, 9.0, 10.0, 12.0, 11.0, 13.0, 14.0]
        close_values = [9.0, 11.0, 10.0, 12.0, 14.0, 12.0, 15.0, 17.0]
        for window in (1, 2, 3, 4, 5):
            assert_matches(
                apply_cci(high_values, low_values, close_values, window),
                cci_reference(high_values, low_values, close_values, window),
                rel_tol=RELATIVE_TOLERANCE_REFERENCE,
                abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
            )

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference: CCI(window=3) over high [10, 12, 11, 13, 15, 14, 16, 18], low
        [8, 9, 9, 10, 12, 11, 13, 14], close [9, 11, 10, 12, 14, 12, 15, 17].
        """
        assert_matches(
            apply_cci(
                [10.0, 12.0, 11.0, 13.0, 15.0, 14.0, 16.0, 18.0],
                [8.0, 9.0, 9.0, 10.0, 12.0, 11.0, 13.0, 14.0],
                [9.0, 11.0, 10.0, 12.0, 14.0, 12.0, 15.0, 17.0],
                3,
            ),
            [
                None,
                None,
                12.500000000000153,
                100.0000000000001,
                99.9999999999999,
                -20.00000000000008,
                90.90909090909088,
                89.47368421052632,
            ],
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )


class TestCciProperties:
    """
    Invariants that must hold for all inputs (property-based).
    """

    @given(case=_cases(coherent_hlc()))
    def test_matches_reference_for_any_input(
        self,
        case: tuple[list[tuple[float, float, float]], int],
    ) -> None:
        """
        Verifies that, for any coherent high/low/close series and window, the implementation matches the naive
        reference (a strictly-positive spread keeps the mean deviation off the ``0 / 0`` boundary, pinned in edge).
        """
        rows, window = case
        high_values, low_values, close_values = split_triples(rows)
        assert_matches(
            apply_cci(high_values, low_values, close_values, window),
            cci_reference(high_values, low_values, close_values, window),
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
            apply_cci(high, low, close, window),
            cci_reference(high, low, close, window),
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    @given(
        case=_cases(coherent_hlc(), window_min=2),
        exponent=st.sampled_from([-4, -3, -2, -1, 1, 2, 3, 4]),
    )
    def test_scale_invariance(
        self,
        case: tuple[list[tuple[float, float, float]], int],
        exponent: int,
    ) -> None:
        """
        Verifies that ``cci`` is scale-invariant: scaling every input value by a constant ``k`` leaves the output
        unchanged -- ``cci(k * x) == cci(x)``. ``k`` is a power of two, so the rescale is exact and adds no
        floating-point error.
        """
        k = 2.0**exponent
        rows, window = case
        high_values, low_values, close_values = split_triples(rows)
        result_base = apply_cci(high_values, low_values, close_values, window)
        result_scaled = apply_cci(
            [value * k for value in high_values],
            [value * k for value in low_values],
            [value * k for value in close_values],
            window,
        )
        assert_scale_homogeneous(result_scaled, result_base, k=k, degree=0)

    @given(case=_cases(coherent_hlc()))
    def test_warmup_null_count_property(
        self,
        case: tuple[list[tuple[float, float, float]], int],
    ) -> None:
        """
        Verifies that the leading-null run is exactly ``min(window - 1, len(values))``.
        """
        rows, window = case
        high_values, low_values, close_values = split_triples(rows)
        result = apply_cci(high_values, low_values, close_values, window)
        leading_nulls = count_leading_nulls(result)
        # NOTE: ``_cases`` couples length >= window, so ``min`` always resolves to ``window - 1``; the form is kept to
        # state the exact warm-up rule (the leading-null run is never clamped by a too-short series here).
        assert leading_nulls == min(window - 1, len(rows))
