"""
Tests for ``pomata.indicators.aroon_oscillator`` — Aroon Up minus Aroon Down as a single line.

``aroon_oscillator`` is multi-input (high, low) and single-output, so tests use a local ``apply_aroon_oscillator``
helper to materialize the factory over a two-column ``Float64`` frame; ``assert_matches`` and the naive
``aroon_oscillator_reference`` oracle are shared across the suite. It is bounded in ``[-100, 100]`` and scale-invariant
(only the positions of the extremes matter) — so it carries scale-invariance and boundedness properties in place of the
homogeneity / large-magnitude tests used for scale-dependent indicators.

The ladder is the canonical one: contract, edge (warm-up / null / NaN / the up-minus-down identity), correctness (vs the
closed-form reference and a frozen golden master), and properties (reference agreement incl. missing data,
scale-invariance, boundedness). Categories are split into classes; cross-cutting categories use markers (see
``tests/README.md``).
"""

import math
from collections.abc import Sequence

import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st
from polars.testing import assert_frame_equal
from tests.indicators.oracles import aroon_oscillator_reference
from tests.support import (
    BOUND_MARGIN,
    GROUP_KEY,
    HIGH,
    LOW,
    assert_matches,
    assert_scale_homogeneous,
    coherent_hl,
    coherent_hl_with_missing,
    materialize,
    split_pairs,
)

from pomata.indicators import aroon, aroon_oscillator

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- derived, not chosen; the rationale is the shared method in CORRECTNESS.md, the numbers are this
# indicator's. To add an indicator, set its three facts here; the property tier below is then the same shape as every
# other indicator's.
#   1. warmup  W(window) = window   (the line is null for the first ``window`` rows, inherited from :func:`aroon`: a
#              full ``window + 1``-bar look-back must exist before the positions of the extremes are defined)
#   2. memory  the oracle shares pomata's seeding, so the property holds from the first defined row (M = 0); each
#              example carries D in [window, 2 * window] defined bars -- one window of output, never all warm-up
#   3. domain  the agreement / boundedness tiers draw ``coherent_hl`` (small positive bars so ties exercise the
#              most-recent-extreme rule the difference inherits); the missing-data tier draws
#              ``coherent_hl_with_missing``. Windows span 1 .. WINDOW_MAX
# The oscillator is a scale-INVARIANT bounded line (O(1) in ``[-100, 100]``, a difference of position-of-extreme
# percentages), so the scale tier uses an ABSOLUTE tolerance, never ``input_scale``-sized, and the large-magnitude tier
# is vacuous (the common factor cancels) and absent. Repetitions N are the shared CI profile (tests/conftest.py);
# override per-test only if its parameter space is larger.
# ----------------------------------------------------------------------------------------------------------------------
WINDOW_MAX = 15


@st.composite
def _cases[T](draw: st.DrawFn, bars: st.SearchStrategy[T]) -> tuple[list[T], int]:
    """
    A (series, window) pair sized from the facts above: ``window`` over its regimes, length = warm-up + a window of
    defined bars, so every example has output to check (never an all-warm-up series, the waste a ``window`` decoupled
    from the length would cause).
    """
    window = draw(st.integers(min_value=1, max_value=WINDOW_MAX))
    defined = draw(st.integers(min_value=window, max_value=2 * window))
    length = window + defined
    return draw(st.lists(bars, min_size=length, max_size=length)), window


def apply_aroon_oscillator(
    high: Sequence[float | None],
    low: Sequence[float | None],
    window: int,
) -> list[float | None]:
    """
    Materialize ``aroon_oscillator`` over a two-column ``Float64`` frame built from the aligned high / low lists.
    """
    return materialize({HIGH: high, LOW: low}, aroon_oscillator(pl.col(HIGH), pl.col(LOW), window))


class TestAroonOscillatorContract:
    """
    Type, shape, and lazy/eager guarantees.
    """

    def test_returns_expr(self) -> None:
        """
        Verifies that the factory returns a ``pl.Expr`` without touching a frame.
        """
        assert isinstance(aroon_oscillator(pl.col(HIGH), pl.col(LOW), 14), pl.Expr)

    def test_preserves_length_and_dtype(self) -> None:
        """
        Verifies that the output has one value per input row and is ``Float64``.
        """
        frame = pl.DataFrame({HIGH: [3.0, 2.0, 4.0, 5.0], LOW: [1.0, 0.0, 2.0, 3.0]})
        result = frame.select(aroon_oscillator(pl.col(HIGH), pl.col(LOW), 2).alias("y"))
        assert result.height == frame.height
        assert result.schema["y"] == pl.Float64

    def test_lazy_eager_parity(self) -> None:
        """
        Verifies that eager and lazy application produce identical materialized output.
        """
        frame = pl.DataFrame({HIGH: [3.0, 2.0, 4.0, 5.0], LOW: [1.0, 0.0, 2.0, 3.0]})
        expr = aroon_oscillator(pl.col(HIGH), pl.col(LOW), 2).alias("y")
        result_eager = frame.select(expr)
        result_lazy = frame.lazy().select(expr).collect()
        assert_frame_equal(result_eager, result_lazy)

    def test_over_partitions_independently(self) -> None:
        """
        Verifies that under ``.over`` the rolling extremes reset per group: the partitioned line equals the per-group
        calls.
        """
        frame = pl.DataFrame(
            {
                GROUP_KEY: ["a"] * 4 + ["b"] * 4,
                HIGH: [10.0, 11.0, 12.0, 11.0, 20.0, 22.0, 21.0, 23.0],
                LOW: [9.0, 10.0, 11.0, 10.0, 19.0, 21.0, 20.0, 22.0],
            }
        )
        expr = aroon_oscillator(pl.col(HIGH), pl.col(LOW), 2).over(GROUP_KEY)
        grouped = frame.select(expr.alias("y"))["y"].to_list()
        group_a = apply_aroon_oscillator([10.0, 11.0, 12.0, 11.0], [9.0, 10.0, 11.0, 10.0], 2)
        group_b = apply_aroon_oscillator([20.0, 22.0, 21.0, 23.0], [19.0, 21.0, 20.0, 22.0], 2)
        assert_matches(grouped, group_a + group_b)


class TestAroonOscillatorEdge:
    """
    Boundaries, warm-up, the up-minus-down identity, and null / NaN handling.
    """

    def test_window_below_one_raises(self) -> None:
        """
        Verifies that ``window < 1`` raises ``ValueError``.
        """
        with pytest.raises(ValueError, match="window must be >= 1"):
            aroon_oscillator(pl.col(HIGH), pl.col(LOW), 0)

    def test_warmup_null_count(self) -> None:
        """
        Verifies that the line is null for the first ``window`` rows and defined once a full look-back exists.
        """
        result = apply_aroon_oscillator([1.0, 2.0, 3.0, 4.0, 5.0], [0.0, 1.0, 2.0, 3.0, 4.0], 2)
        assert result[:2] == [None, None]
        assert result[2] is not None

    def test_window_exceeds_length(self) -> None:
        """
        Verifies that when ``window`` exceeds the series length the whole output is null (no full look-back exists).
        """
        assert_matches(apply_aroon_oscillator([1.0, 2.0, 3.0], [0.0, 1.0, 2.0], 5), [None, None, None])

    def test_empty(self) -> None:
        """
        Verifies behavior on an empty series.
        """
        assert_matches(apply_aroon_oscillator([], [], 2), [])

    def test_single_row(self) -> None:
        """
        Verifies behavior on a one-element series: the lone bar is always warm-up.
        """
        assert_matches(apply_aroon_oscillator([5.0], [4.0], 2), [None])

    def test_all_null(self) -> None:
        """
        Verifies that all-null inputs yield all null.
        """
        assert_matches(
            apply_aroon_oscillator([None, None, None, None], [None, None, None, None], 2), [None, None, None, None]
        )

    def test_equals_up_minus_down(self) -> None:
        """
        Verifies the defining identity: the oscillator equals the :func:`aroon` Up line minus the Down line.
        """
        high = [10.0, 11.0, 12.0, 11.0, 13.0, 12.0, 14.0, 13.0]
        low = [9.0, 10.0, 11.0, 10.0, 12.0, 11.0, 13.0, 12.0]
        oscillator = apply_aroon_oscillator(high, low, 3)
        frame = pl.DataFrame({HIGH: high, LOW: low})
        bands = frame.select(aroon(pl.col(HIGH), pl.col(LOW), 3).alias("a")).unnest("a")
        up = bands["up"].to_list()
        down = bands["down"].to_list()
        expected = [None if u is None or d is None else u - d for u, d in zip(up, down, strict=True)]
        assert_matches(oscillator, expected)

    def test_null_propagates(self) -> None:
        """
        Verifies that a ``null`` anywhere in the look-back yields ``null``.
        """
        high = [10.0, 11.0, None, 13.0, 14.0, 15.0]
        low = [9.0, 10.0, 11.0, 12.0, 13.0, 14.0]
        assert_matches(apply_aroon_oscillator(high, low, 2), aroon_oscillator_reference(high, low, 2))

    def test_nan_propagates(self) -> None:
        """
        Verifies that a ``NaN`` in the look-back yields ``NaN``.
        """
        high = [10.0, 11.0, 12.0, math.nan, 14.0, 15.0]
        low = [9.0, 10.0, 11.0, 12.0, 13.0, 14.0]
        assert_matches(apply_aroon_oscillator(high, low, 2), aroon_oscillator_reference(high, low, 2))


class TestAroonOscillatorCorrectness:
    """
    Against the naive reference oracle and frozen golden-master values.
    """

    def test_matches_reference(self) -> None:
        """
        Verifies agreement with the naive closed-form reference across several windows.
        """
        high = [10.0, 11.0, 12.0, 11.0, 13.0, 12.0, 14.0, 13.0, 15.0, 14.0]
        low = [9.0, 10.0, 11.0, 10.0, 12.0, 11.0, 13.0, 12.0, 14.0, 13.0]
        for window in (1, 2, 3, 5):
            assert_matches(apply_aroon_oscillator(high, low, window), aroon_oscillator_reference(high, low, window))

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference: aroon_oscillator(window=3) over the sample series.
        """
        high = [10.0, 11.0, 12.0, 11.0, 13.0, 12.0, 14.0, 13.0]
        low = [9.0, 10.0, 11.0, 10.0, 12.0, 11.0, 13.0, 12.0]
        result = apply_aroon_oscillator(high, low, 3)
        assert_matches(
            [None if v is None else round(v, 4) for v in result],
            [None, None, None, 66.6667, 33.3333, 33.3333, 100.0, 33.3333],
        )


class TestAroonOscillatorProperties:
    """
    Invariants that must hold for all inputs (property-based).
    """

    @given(case=_cases(coherent_hl()))
    def test_matches_reference_for_any_input(
        self,
        case: tuple[list[tuple[float, float]], int],
    ) -> None:
        """
        Verifies that, for any series and window (small integers force frequent ties), the line matches the reference.
        """
        rows, window = case
        high, low = split_pairs(rows)
        assert_matches(apply_aroon_oscillator(high, low, window), aroon_oscillator_reference(high, low, window))

    @given(
        case=_cases(coherent_hl()),
        exponent=st.sampled_from([-4, -3, -2, -1, 1, 2, 3, 4]),
    )
    def test_scale_invariance(
        self,
        case: tuple[list[tuple[float, float]], int],
        exponent: int,
    ) -> None:
        """
        Verifies that the oscillator is scale-invariant: scaling ``high`` and ``low`` by a positive ``k`` leaves it
        unchanged. ``k`` is a power of two so the rescaling is lossless: the indicator is an argmax/argmin, and an
        arbitrary factor can round two near-tied highs to the same value and flip which one wins, changing the result.
        """
        k = 2.0**exponent
        rows, window = case
        high, low = split_pairs(rows)
        base = apply_aroon_oscillator(high, low, window)
        scaled = apply_aroon_oscillator([value * k for value in high], [value * k for value in low], window)
        assert_scale_homogeneous(scaled, base, k=k, degree=0)

    @given(case=_cases(coherent_hl()))
    def test_bounded(
        self,
        case: tuple[list[tuple[float, float]], int],
    ) -> None:
        """
        Verifies that every defined value lies within ``[-100, 100]``.
        """
        rows, window = case
        high, low = split_pairs(rows)
        for value in apply_aroon_oscillator(high, low, window):
            if value is not None and not math.isnan(value):
                assert -100.0 - BOUND_MARGIN <= value <= 100.0 + BOUND_MARGIN

    @given(case=_cases(coherent_hl_with_missing()))
    def test_matches_reference_under_missing_data(
        self,
        case: tuple[list[tuple[float | None, float | None]], int],
    ) -> None:
        """
        Verifies that, for inputs freely mixing null / NaN / finite, the line matches the naive reference.
        """
        rows, window = case
        high, low = split_pairs(rows)
        assert_matches(apply_aroon_oscillator(high, low, window), aroon_oscillator_reference(high, low, window))
