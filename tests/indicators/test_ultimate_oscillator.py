"""
Tests for ``pomata.indicators.ultimate_oscillator`` — Larry Williams' three-window momentum oscillator.

``ultimate_oscillator`` is multi-input (high, low, close), so tests use a local ``apply_ultimate_oscillator`` helper to
materialize the factory over a three-column ``Float64`` frame; ``assert_matches`` and the naive
``ultimate_oscillator_reference`` oracle are shared across the suite. It is scale-invariant and bounded in ``[0, 100]``
for well-formed bars.

The ladder is the canonical one: contract, edge (window floors / warm-up / flat / null / NaN), correctness (vs the
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
from tests.indicators.oracles import ultimate_oscillator_reference
from tests.support import (
    ABSOLUTE_TOLERANCE_REFERENCE,
    BOUND_MARGIN,
    CLOSE,
    GROUP_KEY,
    HIGH,
    LOW,
    RELATIVE_TOLERANCE_REFERENCE,
    WINDOW_MAX,
    assert_matches,
    assert_scale_homogeneous,
    coherent_hlc,
    coherent_hlc_with_missing,
    materialize,
    split_triples,
)

from pomata.indicators import ultimate_oscillator

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- derived, not chosen; the rationale is the shared method in CORRECTNESS.md, the numbers are this
# indicator's. To add an indicator, set its three facts here; the property tier below is then the same shape as every
# other indicator's.
#   1. warmup  W(window_short, window_medium, window_long) = max(windows) - 1   (the rolling sums emit only once the
#              longest window holds a full set of true-range / buying-pressure terms)
#   2. memory  the oracle shares pomata's seeding, so the property holds from the first defined row (M = 0); each
#              example carries D in [W + 1, W + 1 + span] defined bars -- a window of output, never all warm-up
#   3. domain  coherent_hlc(): coherent (high >= low, low <= close <= high) positive-finite bars -- the oscillator is
#              only well-defined and bounded in ``[0, 100]`` on well-formed bars; windows span 1 .. WINDOW_MAX
# The oscillator is a scale-INVARIANT bounded ratio (O(1) in ``[0, 100]``), so the scale tier uses an ABSOLUTE
# tolerance, never ``input_scale``-sized, and the large-magnitude tier is vacuous (the common factor cancels) and
# absent. Repetitions N are the shared CI profile (tests/conftest.py); override per-test only if its parameter space is
# larger.
# ----------------------------------------------------------------------------------------------------------------------


@st.composite
def _cases[T](draw: st.DrawFn, bars: st.SearchStrategy[T]) -> tuple[list[T], int, int, int]:
    """
    A (series, window_short, window_medium, window_long) tuple sized from the facts above honoring the contract
    ``1 <= window_short <= window_medium <= window_long``: each bound is drawn no smaller than the previous one (so the
    now-rejected mis-ordered regime is never sampled), and length = warm-up + a window of defined bars, so every example
    has output to check (never an all-warm-up series, the waste a window decoupled from the length would cause).
    """
    window_short = draw(st.integers(min_value=1, max_value=WINDOW_MAX))
    window_medium = draw(st.integers(min_value=window_short, max_value=WINDOW_MAX))
    window_long = draw(st.integers(min_value=window_medium, max_value=WINDOW_MAX))
    warmup = window_long - 1
    defined = draw(st.integers(min_value=1, max_value=WINDOW_MAX))
    length = warmup + defined
    return draw(st.lists(bars, min_size=length, max_size=length)), window_short, window_medium, window_long


def apply_ultimate_oscillator(
    high: Sequence[float | None],
    low: Sequence[float | None],
    close: Sequence[float | None],
    window_short: int = 7,
    window_medium: int = 14,
    window_long: int = 28,
) -> list[float | None]:
    """
    Materialize ``ultimate_oscillator`` over a three-column ``Float64`` frame built from the aligned HLC lists.
    """
    return materialize(
        {HIGH: high, LOW: low, CLOSE: close},
        ultimate_oscillator(
            pl.col(HIGH),
            pl.col(LOW),
            pl.col(CLOSE),
            window_short=window_short,
            window_medium=window_medium,
            window_long=window_long,
        ),
    )


class TestUltimateOscillatorContract:
    """
    Type, shape, and lazy/eager guarantees.
    """

    def test_returns_expr(self) -> None:
        """
        Verifies that the factory returns a ``pl.Expr`` without touching a frame.
        """
        assert isinstance(
            ultimate_oscillator(
                pl.col(HIGH), pl.col(LOW), pl.col(CLOSE), window_short=7, window_medium=14, window_long=28
            ),
            pl.Expr,
        )

    def test_preserves_length_and_dtype(self) -> None:
        """
        Verifies that the output has one value per input row and is ``Float64``.
        """
        frame = pl.DataFrame({HIGH: [10.0, 11.0, 12.0], LOW: [9.0, 10.0, 11.0], CLOSE: [9.5, 10.5, 11.5]})
        result = frame.select(
            ultimate_oscillator(
                pl.col(HIGH), pl.col(LOW), pl.col(CLOSE), window_short=1, window_medium=2, window_long=2
            ).alias("y")
        )
        assert result.height == frame.height
        assert result.schema["y"] == pl.Float64

    def test_lazy_eager_parity(self) -> None:
        """
        Verifies that eager and lazy application produce identical materialized output.
        """
        frame = pl.DataFrame(
            {HIGH: [10.0, 11.0, 12.0, 11.5], LOW: [9.0, 10.0, 11.0, 10.5], CLOSE: [9.5, 10.5, 11.5, 11.0]}
        )
        expr = ultimate_oscillator(
            pl.col(HIGH), pl.col(LOW), pl.col(CLOSE), window_short=2, window_medium=2, window_long=3
        ).alias("y")
        result_eager = frame.select(expr)
        result_lazy = frame.lazy().select(expr).collect()
        assert_frame_equal(result_eager, result_lazy)

    def test_over_partitions_independently(self) -> None:
        """
        Verifies that under ``.over`` the windows reset per group and never span boundaries.
        """
        frame = pl.DataFrame(
            {
                GROUP_KEY: ["a"] * 5 + ["b"] * 5,
                HIGH: [10.0, 11.0, 12.0, 11.5, 13.0, 20.0, 21.0, 22.0, 21.5, 23.0],
                LOW: [9.0, 10.0, 11.0, 10.5, 12.0, 19.0, 20.0, 21.0, 20.5, 22.0],
                CLOSE: [9.5, 10.5, 11.5, 11.0, 12.5, 19.5, 20.5, 21.5, 21.0, 22.5],
            }
        )
        grouped = frame.select(
            ultimate_oscillator(
                pl.col(HIGH), pl.col(LOW), pl.col(CLOSE), window_short=2, window_medium=3, window_long=4
            )
            .over(GROUP_KEY)
            .alias("y")
        )["y"].to_list()
        group_a = apply_ultimate_oscillator(
            [10.0, 11.0, 12.0, 11.5, 13.0], [9.0, 10.0, 11.0, 10.5, 12.0], [9.5, 10.5, 11.5, 11.0, 12.5], 2, 3, 4
        )
        group_b = apply_ultimate_oscillator(
            [20.0, 21.0, 22.0, 21.5, 23.0], [19.0, 20.0, 21.0, 20.5, 22.0], [19.5, 20.5, 21.5, 21.0, 22.5], 2, 3, 4
        )
        assert_matches(grouped, group_a + group_b)


class TestUltimateOscillatorEdge:
    """
    Boundaries, warm-up, flat range, and null / NaN handling.
    """

    def test_window_short_below_one_raises(self) -> None:
        """
        Verifies that ``window_short < 1`` raises ``ValueError``.
        """
        with pytest.raises(ValueError, match="window_short must be >= 1"):
            ultimate_oscillator(
                pl.col(HIGH), pl.col(LOW), pl.col(CLOSE), window_short=0, window_medium=14, window_long=28
            )

    def test_window_medium_below_one_raises(self) -> None:
        """
        Verifies that ``window_medium < 1`` raises ``ValueError``.
        """
        with pytest.raises(ValueError, match="window_medium must be >= 1"):
            ultimate_oscillator(
                pl.col(HIGH), pl.col(LOW), pl.col(CLOSE), window_short=7, window_medium=0, window_long=28
            )

    def test_window_long_below_one_raises(self) -> None:
        """
        Verifies that ``window_long < 1`` raises ``ValueError``.
        """
        with pytest.raises(ValueError, match="window_long must be >= 1"):
            ultimate_oscillator(
                pl.col(HIGH), pl.col(LOW), pl.col(CLOSE), window_short=7, window_medium=14, window_long=0
            )

    def test_misordered_windows_raise(self) -> None:
        """
        Verifies that windows not ordered ``window_short <= window_medium <= window_long`` raise ``ValueError`` (the
        three windows must run shortest to longest), while the equal-window case is accepted.
        """
        with pytest.raises(ValueError, match="windows must be ordered window_short <= window_medium <= window_long"):
            ultimate_oscillator(
                pl.col(HIGH), pl.col(LOW), pl.col(CLOSE), window_short=14, window_medium=7, window_long=28
            )
        with pytest.raises(ValueError, match="windows must be ordered window_short <= window_medium <= window_long"):
            ultimate_oscillator(
                pl.col(HIGH), pl.col(LOW), pl.col(CLOSE), window_short=7, window_medium=28, window_long=14
            )
        assert isinstance(
            ultimate_oscillator(
                pl.col(HIGH), pl.col(LOW), pl.col(CLOSE), window_short=5, window_medium=5, window_long=5
            ),
            pl.Expr,
        )

    def test_warmup_null_count(self) -> None:
        """
        Verifies that the first ``max(windows) - 1`` rows are null (warm-up from the longest window).
        """
        high = [10.0, 11.0, 12.0, 11.5, 13.0]
        low = [9.0, 10.0, 11.0, 10.5, 12.0]
        close = [9.5, 10.5, 11.5, 11.0, 12.5]
        result = apply_ultimate_oscillator(high, low, close, 2, 3, 4)
        assert result[:3] == [None, None, None]
        assert result[3] is not None

    def test_empty(self) -> None:
        """
        Verifies that an empty input yields an empty output.
        """
        assert_matches(apply_ultimate_oscillator([], [], []), [])

    def test_all_null(self) -> None:
        """
        Verifies that an all-null series yields an all-null output.
        """
        nulls: list[float | None] = [None] * 8
        assert_matches(apply_ultimate_oscillator(nulls, nulls, nulls), [None] * 8)

    def test_single_row(self) -> None:
        """
        Verifies that a one-bar series is all warm-up for any windows above one.
        """
        assert_matches(apply_ultimate_oscillator([10.0], [9.0], [9.5], 2, 3, 4), [None])

    def test_window_exceeds_length(self) -> None:
        """
        Verifies that a series shorter than the longest window yields an all-null output.
        """
        high = [10.0, 11.0, 12.0]
        low = [9.0, 10.0, 11.0]
        close = [9.5, 10.5, 11.5]
        assert_matches(apply_ultimate_oscillator(high, low, close, 2, 3, 4), [None, None, None])

    def test_flat_window_is_nan(self) -> None:
        """
        Verifies that a flat series (true range sums to zero) yields ``NaN`` (``0 / 0``).
        """
        result = apply_ultimate_oscillator([10.0, 10.0, 10.0], [10.0, 10.0, 10.0], [10.0, 10.0, 10.0], 1, 1, 2)
        assert result[0] is None
        assert result[1] is not None
        assert math.isnan(result[1])

    def test_flat_window_is_nan_at_large_magnitude(self) -> None:
        """
        Verifies the exact-flat guard is residual-free: a constant series at a large magnitude still yields ``NaN``,
        because the genuine ``0 / 0`` is detected via the residual-free rolling maxima of the true range and the buying
        pressure rather than the summed quotient (which could accumulate a non-zero float residue at that scale).
        """
        flat = [1e9, 1e9, 1e9]
        result = apply_ultimate_oscillator(flat, flat, flat, 1, 1, 2)
        assert result[0] is None
        assert result[1] is not None
        assert math.isnan(result[1])

    def test_null_propagates(self) -> None:
        """
        Verifies that a null propagates (matching the naive reference).
        """
        high = [10.0, 11.0, 12.0, None, 13.0, 13.5, 14.0, 13.5, 15.0, 14.5]
        low = [9.0, 10.0, 11.0, 10.5, 12.0, 11.5, 13.0, 12.5, 14.0, 13.5]
        close = [9.5, 10.5, 11.5, 11.0, 12.5, 12.0, 13.5, 13.0, 14.5, 14.0]
        assert_matches(
            apply_ultimate_oscillator(high, low, close, 2, 3, 4),
            ultimate_oscillator_reference(high, low, close, 2, 3, 4),
        )

    def test_nan_propagates(self) -> None:
        """
        Verifies that a NaN propagates (matching the naive reference).
        """
        high = [10.0, 11.0, 12.0, 12.5, 13.0, math.nan, 14.0, 13.5, 15.0, 14.5]
        low = [9.0, 10.0, 11.0, 10.5, 12.0, 11.5, 13.0, 12.5, 14.0, 13.5]
        close = [9.5, 10.5, 11.5, 11.0, 12.5, 12.0, 13.5, 13.0, 14.5, 14.0]
        assert_matches(
            apply_ultimate_oscillator(high, low, close, 2, 3, 4),
            ultimate_oscillator_reference(high, low, close, 2, 3, 4),
        )


class TestUltimateOscillatorCorrectness:
    """
    Against the naive reference oracle and frozen golden-master values.
    """

    def test_matches_reference(self) -> None:
        """
        Verifies agreement with the naive closed-form reference across several window triples.
        """
        high = [10.0, 11.0, 12.0, 11.5, 13.0, 12.5, 14.0, 13.5, 15.0, 14.5]
        low = [9.0, 10.0, 11.0, 10.5, 12.0, 11.5, 13.0, 12.5, 14.0, 13.5]
        close = [9.5, 10.5, 11.5, 11.0, 12.5, 12.0, 13.5, 13.0, 14.5, 14.0]
        for window_short, window_medium, window_long in ((1, 1, 1), (2, 3, 4), (2, 4, 7), (1, 2, 5)):
            assert_matches(
                apply_ultimate_oscillator(high, low, close, window_short, window_medium, window_long),
                ultimate_oscillator_reference(high, low, close, window_short, window_medium, window_long),
                rel_tol=RELATIVE_TOLERANCE_REFERENCE,
                abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
            )

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference: ultimate_oscillator(2, 3, 4) over the sample series.
        """
        high = [10.0, 11.0, 12.0, 11.5, 13.0, 12.5, 14.0, 13.5, 15.0, 14.5]
        low = [9.0, 10.0, 11.0, 10.5, 12.0, 11.5, 13.0, 12.5, 14.0, 13.5]
        close = [9.5, 10.5, 11.5, 11.0, 12.5, 12.0, 13.5, 13.0, 14.5, 14.0]
        result = apply_ultimate_oscillator(high, low, close, 2, 3, 4)
        assert_matches(
            [None if v is None else round(v, 4) for v in result],
            [None, None, None, 60.7143, 66.6667, 65.0433, 67.619, 65.4762, 67.619, 65.4762],
        )


class TestUltimateOscillatorProperties:
    """
    Invariants that must hold for all inputs (property-based).
    """

    @given(case=_cases(coherent_hlc()))
    def test_matches_reference_for_any_input(
        self,
        case: tuple[list[tuple[float, float, float]], int, int, int],
    ) -> None:
        """
        Verifies that, for any positive series and windows, the implementation matches the naive reference.
        """
        rows, window_short, window_medium, window_long = case
        high, low, close = split_triples(rows)
        assert_matches(
            apply_ultimate_oscillator(high, low, close, window_short, window_medium, window_long),
            ultimate_oscillator_reference(high, low, close, window_short, window_medium, window_long),
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    @given(
        case=_cases(coherent_hlc()),
        exponent=st.sampled_from([-4, -3, -2, -1, 1, 2, 3, 4]),
    )
    def test_scale_invariance(
        self,
        case: tuple[list[tuple[float, float, float]], int, int, int],
        exponent: int,
    ) -> None:
        """
        Verifies that ``ultimate_oscillator`` is scale-invariant under a positive common rescaling of the bars. ``k`` is
        a power of two so the rescaling is lossless and cannot introduce a floating-point artifact.
        """
        k = 2.0**exponent
        rows, window_short, window_medium, window_long = case
        high, low, close = split_triples(rows)
        base = apply_ultimate_oscillator(high, low, close, window_short, window_medium, window_long)
        scaled = apply_ultimate_oscillator(
            [value * k for value in high],
            [value * k for value in low],
            [value * k for value in close],
            window_short,
            window_medium,
            window_long,
        )
        assert_scale_homogeneous(scaled, base, k=k, degree=0)

    @given(case=_cases(coherent_hlc()))
    def test_bounded(
        self,
        case: tuple[list[tuple[float, float, float]], int, int, int],
    ) -> None:
        """
        Verifies that every defined value lies within ``[0, 100]`` for well-formed OHLC bars (``low <= close <= high``).
        """
        rows, window_short, window_medium, window_long = case
        high, low, close = split_triples(rows)
        for value in apply_ultimate_oscillator(high, low, close, window_short, window_medium, window_long):
            if value is not None and not math.isnan(value):
                assert -BOUND_MARGIN <= value <= 100.0 + BOUND_MARGIN

    @given(case=_cases(coherent_hlc_with_missing()))
    def test_matches_reference_under_missing_data(
        self,
        case: tuple[list[tuple[float | None, float | None, float | None]], int, int, int],
    ) -> None:
        """
        Verifies that, for positive inputs freely mixing null / NaN, the implementation matches the naive reference.
        """
        rows, window_short, window_medium, window_long = case
        high, low, close = split_triples(rows)
        assert_matches(
            apply_ultimate_oscillator(high, low, close, window_short, window_medium, window_long),
            ultimate_oscillator_reference(high, low, close, window_short, window_medium, window_long),
        )
