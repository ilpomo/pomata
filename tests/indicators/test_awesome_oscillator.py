"""
Tests for ``pomata.indicators.awesome_oscillator`` — a fast minus a slow simple average of the bar median.

``awesome_oscillator`` is multi-input (``high`` / ``low``) and single-output; tests use a local
``apply_awesome_oscillator`` helper to materialize the factory over a two-column ``Float64`` frame. ``assert_matches``
and the naive ``awesome_oscillator_reference`` oracle (the certified ``price_median`` + ``sma`` composition) are shared.

The ladder is the canonical one: contract (type / shape / lazy-eager / ``.over`` independence), edge (warm-up / window
boundaries / relational validation / single-row / null / NaN / flat), correctness (vs the composed reference and a
frozen golden master), and properties (reference agreement incl. missing data, scale-homogeneity, large-magnitude
stability). Categories are split into classes; cross-cutting categories use markers (see ``tests/README.md``).
"""

import math
from collections.abc import Sequence

import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st
from tests.indicators.oracles import awesome_oscillator_reference
from tests.support import (
    EXACT_TOLERANCE_FACTOR,
    GROUP_KEY,
    HIGH,
    LOW,
    RELATIVE_TOLERANCE_PROPERTY,
    RELATIVE_TOLERANCE_SCALE,
    WINDOW_MAX,
    assert_matches,
    assert_scale_homogeneous,
    coherent_hl,
    coherent_hl_with_missing,
    input_scale,
    materialize,
    split_pairs,
)

from pomata.indicators import awesome_oscillator

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- derived, not chosen; the rationale is the shared method in CORRECTNESS.md, the numbers are this
# indicator's. To add an indicator, set its three facts here; the property tier below is then the same shape as every
# other indicator's.
#   1. warmup  W = window_slow - 1   (both averages must be defined before their difference is)
#   2. memory  the oracle is windowed like pomata, so the property holds from the first defined row (M = 0); each
#              example carries D in [window_slow, 2 * window_slow] defined bars -- output to check, never all warm-up
#   3. domain  coherent_hl(): coherent (high >= low) positive-finite bars; the oscillator is a difference of simple
#              averages of the median, so no subnormal-square floor is needed
# Windows span 1 .. WINDOW_MAX with window_fast <= window_slow. Repetitions N are the shared CI profile.
# ----------------------------------------------------------------------------------------------------------------------


@st.composite
def _cases[T](draw: st.DrawFn, bars: st.SearchStrategy[T]) -> tuple[list[T], int, int]:
    """
    A (series, window_fast, window_slow) triple sized from the facts above, with ``window_fast <= window_slow`` and the
    length covering the slow warm-up plus a window of defined output.
    """
    window_slow = draw(st.integers(min_value=1, max_value=WINDOW_MAX))
    window_fast = draw(st.integers(min_value=1, max_value=window_slow))
    defined = draw(st.integers(min_value=window_slow, max_value=2 * window_slow))
    length = (window_slow - 1) + defined
    return draw(st.lists(bars, min_size=length, max_size=length)), window_fast, window_slow


def apply_awesome_oscillator(
    high: Sequence[float | None],
    low: Sequence[float | None],
    window_fast: int,
    window_slow: int,
) -> list[float | None]:
    """
    Materialize ``awesome_oscillator`` over a two-column ``Float64`` frame built from the aligned high / low lists.
    """
    return materialize(
        {HIGH: high, LOW: low},
        awesome_oscillator(pl.col(HIGH), pl.col(LOW), window_fast=window_fast, window_slow=window_slow),
    )


class TestAwesomeOscillatorContract:
    """
    Type, shape, and lazy/eager guarantees.
    """

    def test_over_partitions_independently(self) -> None:
        """
        Verifies that under ``.over`` neither average spans group boundaries.
        """
        frame = pl.DataFrame(
            {
                GROUP_KEY: ["a", "a", "a", "b", "b", "b"],
                HIGH: [2.0, 4.0, 6.0, 12.0, 14.0, 16.0],
                LOW: [0.0, 2.0, 4.0, 10.0, 12.0, 14.0],
            }
        )
        expr = awesome_oscillator(pl.col(HIGH), pl.col(LOW), window_fast=2, window_slow=3).over(GROUP_KEY)
        result = frame.select(expr.alias("y"))["y"].to_list()
        # Each group warms up over window_slow - 1 = 2; group b must not inherit group a's averages.
        assert result[:2] == [None, None]
        assert result[3:5] == [None, None]
        assert result[2] is not None
        assert result[5] is not None


class TestAwesomeOscillatorEdge:
    """
    Boundaries, warm-up, relational validation, null / NaN, and the flat series.
    """

    def test_window_fast_below_one_raises(self) -> None:
        """
        Verifies that ``window_fast < 1`` raises ``ValueError``.
        """
        with pytest.raises(ValueError, match="window_fast must be >= 1"):
            awesome_oscillator(pl.col(HIGH), pl.col(LOW), window_fast=0, window_slow=3)

    def test_window_slow_below_one_raises(self) -> None:
        """
        Verifies that ``window_slow < 1`` raises ``ValueError``.
        """
        with pytest.raises(ValueError, match="window_slow must be >= 1"):
            awesome_oscillator(pl.col(HIGH), pl.col(LOW), window_fast=1, window_slow=0)

    def test_fast_exceeds_slow_raises(self) -> None:
        """
        Verifies that ``window_fast > window_slow`` raises ``ValueError`` (the fast leg must be the shorter one).
        """
        with pytest.raises(ValueError, match="windows must be ordered window_fast <= window_slow"):
            awesome_oscillator(pl.col(HIGH), pl.col(LOW), window_fast=5, window_slow=3)

    def test_equal_windows_is_zero(self) -> None:
        """
        Verifies that ``window_fast == window_slow`` gives an identically-zero oscillator where defined.
        """
        result = apply_awesome_oscillator([2.0, 4.0, 6.0, 8.0], [0.0, 2.0, 4.0, 6.0], 2, 2)
        assert_matches(result, [None, 0.0, 0.0, 0.0])

    def test_all_null(self) -> None:
        """
        Verifies that an all-null input yields an all-null output.
        """
        assert_matches(apply_awesome_oscillator([None, None, None], [None, None, None], 1, 2), [None, None, None])

    def test_warmup_null_count(self) -> None:
        """
        Verifies that the first ``window_slow - 1`` rows are null (warm-up) and the next is defined.
        """
        result = apply_awesome_oscillator([2.0, 4.0, 6.0, 8.0, 10.0], [0.0, 2.0, 4.0, 6.0, 8.0], 2, 3)
        assert result[:2] == [None, None]
        assert result[2] is not None

    def test_single_row(self) -> None:
        """
        Verifies a one-element series: ``window_fast == window_slow == 1`` gives ``0``, a larger window is warm-up.
        """
        assert_matches(apply_awesome_oscillator([2.0], [0.0], 1, 1), [0.0])
        assert_matches(apply_awesome_oscillator([2.0], [0.0], 1, 3), [None])

    def test_flat_series_is_zero(self) -> None:
        """
        Verifies the flat series: over a constant median both averages equal it, so the oscillator is ``0``.
        """
        flat = [5.0, 5.0, 5.0, 5.0, 5.0]
        assert_matches(apply_awesome_oscillator(flat, flat, 2, 3), [None, None, 0.0, 0.0, 0.0])

    def test_null_and_nan_follow_the_legs(self) -> None:
        """
        Verifies that a ``null`` / ``NaN`` in either input flows through the median and the two averages exactly as the
        composed reference says.
        """
        high = [2.0, None, 6.0, 8.0, math.nan, 12.0]
        low = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
        result = apply_awesome_oscillator(high, low, 1, 2)
        assert_matches(result, awesome_oscillator_reference(high, low, 1, 2))


class TestAwesomeOscillatorCorrectness:
    """
    Against the composed reference oracle and a frozen golden master.
    """

    def test_matches_reference(self) -> None:
        """
        Verifies agreement with the composed reference across several ``window_fast`` / ``window_slow`` pairs.
        """
        high = [2.0, 4.0, 6.0, 5.0, 7.0, 9.0, 8.0, 11.0]
        low = [0.0, 2.0, 4.0, 3.0, 5.0, 7.0, 6.0, 9.0]
        for window_fast, window_slow in ((1, 2), (2, 3), (2, 5), (3, 3)):
            assert_matches(
                apply_awesome_oscillator(high, low, window_fast, window_slow),
                awesome_oscillator_reference(high, low, window_fast, window_slow),
            )

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference: AO(window_fast=2, window_slow=3) over a +2 median ramp == a constant 1.
        """
        result = apply_awesome_oscillator([2.0, 4.0, 6.0, 8.0, 10.0], [0.0, 2.0, 4.0, 6.0, 8.0], 2, 3)
        assert_matches(result, [None, None, 1.0, 1.0, 1.0])


class TestAwesomeOscillatorProperties:
    """
    Invariants that must hold for all inputs (property-based).
    """

    @given(case=_cases(coherent_hl()))
    def test_matches_reference_for_any_input(
        self,
        case: tuple[list[tuple[float, float]], int, int],
    ) -> None:
        """
        Verifies that, for any coherent high/low series and windows, the output matches the composed reference.
        """
        rows, window_fast, window_slow = case
        high, low = split_pairs(rows)
        assert_matches(
            apply_awesome_oscillator(high, low, window_fast, window_slow),
            awesome_oscillator_reference(high, low, window_fast, window_slow),
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=input_scale(high) * EXACT_TOLERANCE_FACTOR,
        )

    @given(
        case=_cases(coherent_hl()),
        exponent=st.sampled_from([-4, -3, -2, -1, 1, 2, 3, 4]),
    )
    def test_scale_homogeneity(
        self,
        case: tuple[list[tuple[float, float]], int, int],
        exponent: int,
    ) -> None:
        """
        Verifies that, for positive ``k``, the oscillator is homogeneous of degree 1: ``AO(k * bars) == k * AO``. ``k``
        is a power of two so the rescaling is lossless.
        """
        k = 2.0**exponent
        rows, window_fast, window_slow = case
        high, low = split_pairs(rows)
        base = apply_awesome_oscillator(high, low, window_fast, window_slow)
        scaled = apply_awesome_oscillator([v * k for v in high], [v * k for v in low], window_fast, window_slow)
        assert_scale_homogeneous(scaled, base, k=k, degree=1)

    @given(case=_cases(coherent_hl_with_missing()))
    def test_matches_reference_under_missing_data(
        self,
        case: tuple[list[tuple[float | None, float | None]], int, int],
    ) -> None:
        """
        Verifies that, for inputs freely mixing null / NaN / finite, the output matches the composed reference.
        """
        rows, window_fast, window_slow = case
        high, low = split_pairs(rows)
        assert_matches(
            apply_awesome_oscillator(high, low, window_fast, window_slow),
            awesome_oscillator_reference(high, low, window_fast, window_slow),
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=input_scale(high) * EXACT_TOLERANCE_FACTOR,
        )

    @given(
        case=_cases(coherent_hl()),
        scale=st.sampled_from([1e-6, 1e6, 1e9]),
    )
    def test_matches_reference_at_large_magnitude(
        self,
        case: tuple[list[tuple[float, float]], int, int],
        scale: float,
    ) -> None:
        """
        Verifies that at extreme positive magnitudes the output stays finite where the reference is and agrees.
        """
        rows, window_fast, window_slow = case
        high = [row[0] * scale for row in rows]
        low = [row[1] * scale for row in rows]
        assert_matches(
            apply_awesome_oscillator(high, low, window_fast, window_slow),
            awesome_oscillator_reference(high, low, window_fast, window_slow),
            rel_tol=RELATIVE_TOLERANCE_SCALE,
            abs_tol=input_scale(high) * EXACT_TOLERANCE_FACTOR,
        )
