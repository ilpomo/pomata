"""
Tests for ``pomata.indicators.atr_normalized`` — the ATR as a percentage of the close (NATR).

``atr_normalized`` is multi-input (high, low, close) and single-output, so tests use a local ``apply_atr_normalized``
helper to materialize the factory over a three-column ``Float64`` frame; ``assert_matches`` and the naive
``atr_normalized_reference`` oracle (which composes ``atr_reference``) are shared across the suite. Inputs are positive
prices (so the close denominator is non-zero), and NATR is scale-invariant — so it carries a scale-invariance property
in place of the homogeneity / large-magnitude tests used for scale-dependent indicators.

The ladder is the canonical one: contract, edge (warm-up / null / NaN), correctness (vs the closed-form reference and a
frozen golden master), and properties (reference agreement incl. missing data, scale-invariance). Categories are split
into classes; cross-cutting categories use markers (see ``tests/README.md``).
"""

import math
from collections.abc import Sequence

import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st
from tests.indicators.oracles import atr_normalized_reference
from tests.support import (
    ABSOLUTE_TOLERANCE_REFERENCE,
    CLOSE,
    GROUP_KEY,
    HIGH,
    LOW,
    RELATIVE_TOLERANCE_PROPERTY,
    WINDOW_MAX,
    assert_matches,
    assert_scale_homogeneous,
    coherent_hlc,
    coherent_hlc_with_missing,
    materialize,
    split_triples,
)

from pomata.indicators import atr_normalized

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- derived, not chosen; the rationale is the shared method in CORRECTNESS.md, the numbers are this
# indicator's. To add an indicator, set its three facts here; the property tier below is then the same shape as every
# other indicator's.
#   1. warmup  W(window) = window - 1   (inherited from the underlying ATR, whose rma emits only once ``window``
#              non-null true ranges have accrued)
#   2. memory  the oracle shares pomata's recursive Wilder seeding, so the property holds from the first defined row
#              (M = 0); each example carries D in [window, 2 * window] defined bars -- one window of output, never all
#              warm-up
#   3. domain  coherent_hlc(): coherent (high >= low, low <= close <= high) positive-finite bars -- the positive close
#              keeps the denominator non-zero; windows span 1 .. WINDOW_MAX
# NATR is scale-INVARIANT (atr / close): its value is O(1) whatever the input magnitude, so its tolerance is ABSOLUTE
# (never input_scale-sized), and it carries a scale-INVARIANCE property in place of the homogeneity / large-magnitude
# tests of a scale-dependent indicator -- a large-magnitude test would be vacuous because the common scale cancels in
# the ratio. Repetitions N are the shared CI profile (tests/conftest.py); override per-test only if its parameter space
# is larger.
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


def apply_atr_normalized(
    high: Sequence[float | None],
    low: Sequence[float | None],
    close: Sequence[float | None],
    window: int,
) -> list[float | None]:
    """
    Materialize ``atr_normalized`` over a three-column ``Float64`` frame built from the aligned HLC lists.
    """
    return materialize(
        {HIGH: high, LOW: low, CLOSE: close}, atr_normalized(pl.col(HIGH), pl.col(LOW), pl.col(CLOSE), window)
    )


class TestAtrNormalizedContract:
    """
    Type, shape, and lazy/eager guarantees.
    """

    def test_over_partitions_independently(self) -> None:
        """
        Verifies that under ``.over`` the underlying ATR resets per group: the partitioned line equals the per-group
        calls.
        """
        frame = pl.DataFrame(
            {
                GROUP_KEY: ["a"] * 3 + ["b"] * 3,
                HIGH: [10.2, 10.5, 10.7, 20.2, 20.5, 20.7],
                LOW: [9.8, 10.0, 10.2, 19.8, 20.0, 20.2],
                CLOSE: [10.0, 10.3, 10.5, 20.0, 20.3, 20.5],
            }
        )
        expr = atr_normalized(pl.col(HIGH), pl.col(LOW), pl.col(CLOSE), 2).over(GROUP_KEY)
        grouped = frame.select(expr.alias("y"))["y"].to_list()
        group_a = apply_atr_normalized([10.2, 10.5, 10.7], [9.8, 10.0, 10.2], [10.0, 10.3, 10.5], 2)
        group_b = apply_atr_normalized([20.2, 20.5, 20.7], [19.8, 20.0, 20.2], [20.0, 20.3, 20.5], 2)
        assert_matches(grouped, group_a + group_b)


class TestAtrNormalizedEdge:
    """
    Boundaries, warm-up, and null / NaN handling.
    """

    def test_window_below_one_raises(self) -> None:
        """
        Verifies that ``window < 1`` raises ``ValueError``.
        """
        with pytest.raises(ValueError, match="window must be >= 1"):
            atr_normalized(pl.col(HIGH), pl.col(LOW), pl.col(CLOSE), 0)

    def test_warmup_null_count(self) -> None:
        """
        Verifies that the line is null for the first ``window - 1`` rows (inherited from the ATR).
        """
        result = apply_atr_normalized([10.2, 10.5, 10.7, 10.3], [9.8, 10.0, 10.2, 9.9], [10.0, 10.3, 10.5, 10.1], 2)
        assert result[0] is None
        assert result[1] is not None

    def test_single_row(self) -> None:
        """
        Verifies behavior on a one-element series: ``window == 1`` is defined, a larger window is all warm-up.
        """
        result_window_one = apply_atr_normalized([10.0], [8.0], [9.0], 1)
        assert result_window_one[0] is not None
        assert_matches(apply_atr_normalized([10.0], [8.0], [9.0], 3), [None])

    def test_window_exceeds_length(self) -> None:
        """
        Verifies that a window longer than the series yields an all-null result (the warm-up never completes).
        """
        assert_matches(
            apply_atr_normalized([10.2, 10.5, 10.7], [9.8, 10.0, 10.2], [10.0, 10.3, 10.5], 5), [None, None, None]
        )

    def test_all_null(self) -> None:
        """
        Verifies that an all-null OHLC frame yields an all-null result (the underlying ATR never seeds).
        """
        assert_matches(
            apply_atr_normalized([None, None, None], [None, None, None], [None, None, None], 2), [None, None, None]
        )

    def test_null_propagates(self) -> None:
        """
        Verifies that a null propagates (matching the naive reference).
        """
        high = [10.2, 10.5, 10.7, 10.3, 10.8]
        low = [9.8, 10.0, 10.2, 9.9, 10.3]
        close = [10.0, 10.3, None, 10.1, 10.6]
        assert_matches(
            apply_atr_normalized(high, low, close, 2),
            atr_normalized_reference(high, low, close, 2),
        )

    def test_nan_latches(self) -> None:
        """
        Verifies that a NaN propagates (matching the naive reference).
        """
        high = [10.2, 10.5, 10.7, 10.3, 10.8]
        low = [9.8, 10.0, 10.2, 9.9, 10.3]
        close = [10.0, 10.3, 10.4, 10.1, math.nan]
        assert_matches(
            apply_atr_normalized(high, low, close, 2),
            atr_normalized_reference(high, low, close, 2),
        )

    def test_zero_close_is_non_finite(self) -> None:
        """
        Verifies the documented zero-close behavior, which follows IEEE-754 division: a non-zero ATR over a zero close
        yields ``+inf`` (``100 * atr / 0``), while a zero ATR over a zero close yields ``NaN`` (``0 / 0``). The result
        is pinned directly because the naive reference does bare Python division and raises there, so it cannot
        represent this branch.
        """
        non_zero_atr = apply_atr_normalized([10.0, 12.0], [8.0, 9.0], [9.0, 0.0], 2)
        assert_matches(non_zero_atr, [None, math.inf])
        zero_atr = apply_atr_normalized([0.0, 0.0], [0.0, 0.0], [0.0, 0.0], 2)
        assert_matches(zero_atr, [None, math.nan])


class TestAtrNormalizedCorrectness:
    """
    Against the naive reference oracle and frozen golden-master values.
    """

    def test_matches_reference(self) -> None:
        """
        Verifies agreement with the naive closed-form reference across several windows.
        """
        high = [10.2, 10.5, 10.7, 10.3, 10.8, 11.0, 10.6, 11.2]
        low = [9.8, 10.0, 10.2, 9.9, 10.3, 10.5, 10.1, 10.7]
        close = [10.0, 10.3, 10.5, 10.1, 10.6, 10.8, 10.4, 11.0]
        for window in (1, 2, 3, 4):
            assert_matches(
                apply_atr_normalized(high, low, close, window), atr_normalized_reference(high, low, close, window)
            )

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference: atr_normalized(window=2) over the sample series.
        """
        result = apply_atr_normalized(
            [10.2, 10.5, 10.7, 10.3, 10.8], [9.8, 10.0, 10.2, 9.9, 10.3], [10.0, 10.3, 10.5, 10.1, 10.6], 2
        )
        assert_matches(
            [None if v is None else round(v, 4) for v in result],
            [None, 4.3689, 4.5238, 5.3218, 5.8373],
        )


class TestAtrNormalizedProperties:
    """
    Invariants that must hold for all inputs (property-based).
    """

    @given(case=_cases(coherent_hlc()))
    def test_matches_reference_for_any_input(
        self,
        case: tuple[list[tuple[float, float, float]], int],
    ) -> None:
        """
        Verifies that, for any positive HLC series and window, the implementation matches the naive reference.
        """
        rows, window = case
        high, low, close = split_triples(rows)
        assert_matches(
            apply_atr_normalized(high, low, close, window),
            atr_normalized_reference(high, low, close, window),
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    @given(
        case=_cases(coherent_hlc()),
        exponent=st.sampled_from([-4, -3, -2, -1, 1, 2, 3, 4]),
    )
    def test_scale_invariance(
        self,
        case: tuple[list[tuple[float, float, float]], int],
        exponent: int,
    ) -> None:
        """
        Verifies that NATR is scale-invariant: scaling all of high / low / close by a positive ``k`` leaves it
        unchanged (the ATR and the close scale together). ``k`` is a power of two so the rescaling is lossless and
        cannot perturb the ratio through floating-point rounding.
        """
        k = 2.0**exponent
        rows, window = case
        high, low, close = split_triples(rows)
        base = apply_atr_normalized(high, low, close, window)
        scaled = apply_atr_normalized(
            [value * k for value in high], [value * k for value in low], [value * k for value in close], window
        )
        assert_scale_homogeneous(scaled, base, k=k, degree=0)

    @given(case=_cases(coherent_hlc_with_missing()))
    def test_matches_reference_under_missing_data(
        self,
        case: tuple[list[tuple[float | None, float | None, float | None]], int],
    ) -> None:
        """
        Verifies that, for positive inputs freely mixing null / NaN, the implementation matches the naive reference.
        """
        rows, window = case
        high, low, close = split_triples(rows)
        assert_matches(
            apply_atr_normalized(high, low, close, window),
            atr_normalized_reference(high, low, close, window),
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )
