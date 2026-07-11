"""
Tests for ``pomata.indicators.dx`` — the Directional Index (DX).

``dx`` is multi-input (high, low, close), so tests use a local ``apply_dx`` helper to materialize the factory over a
three-column ``Float64`` frame; ``assert_matches`` and the naive ``dx_reference`` oracle are shared across the suite. It
is a ratio bounded in ``[0, 100]`` and scale-invariant — so it carries scale-invariance and boundedness properties in
place of homogeneity / large-magnitude.

The ladder is the canonical one: contract, edge (window floor / warm-up / null / NaN), correctness (vs the closed-form
reference and a frozen golden master), and properties. Categories are split into classes; cross-cutting categories use
markers (see ``tests/README.md``).
"""

import math
from collections.abc import Sequence

import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st
from tests.indicators.oracles import dx_reference
from tests.support import (
    ABSOLUTE_TOLERANCE_PROPERTY,
    BOUND_MARGIN,
    CLOSE,
    GROUP_KEY,
    HIGH,
    LOW,
    RELATIVE_TOLERANCE_PROPERTY,
    assert_matches,
    assert_scale_homogeneous,
    coherent_hlc,
    coherent_hlc_with_missing,
    materialize,
    split_triples,
)

from pomata.indicators import dx

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- derived, not chosen; the rationale is the shared method in CORRECTNESS.md, the numbers are this
# indicator's. To add an indicator, set its three facts here; the property tier below is then the same shape as every
# other indicator's.
#   1. warmup  W(window) = window - 1   (inherited from the directional indicators: the first ``window - 1`` rows are
#              null before the first index is emitted)
#   2. memory  the oracle shares pomata's recursive Wilder seeding, so the property holds from the first defined row
#              (M = 0); each example carries D in [window, 2 * window] defined bars -- one window of output, never all
#              warm-up
#   3. domain  coherent_hlc() bars (low <= close <= high, positive finite), widened to coherent_hlc_with_missing()
#              where the test mixes null / NaN
# Windows span 1 .. WINDOW_MAX. Repetitions N are the shared CI profile (tests/conftest.py); override per-test only if
# its parameter space is larger.
# ----------------------------------------------------------------------------------------------------------------------
WINDOW_MAX = 15


@st.composite
def _cases[T](
    draw: st.DrawFn,
    bars: st.SearchStrategy[T],
    window_min: int = 1,
) -> tuple[list[T], int]:
    """
    A (series, window) pair sized from the facts above: ``window`` over its regimes, length = warm-up + a window of
    defined bars, so every example has output to check (never an all-warm-up series, the waste a ``window`` decoupled
    from the length would cause).
    """
    window = draw(st.integers(min_value=window_min, max_value=WINDOW_MAX))
    defined = draw(st.integers(min_value=window, max_value=2 * window))
    length = (window - 1) + defined
    return draw(st.lists(bars, min_size=length, max_size=length)), window


def apply_dx(
    high: Sequence[float | None],
    low: Sequence[float | None],
    close: Sequence[float | None],
    window: int,
) -> list[float | None]:
    """
    Materialize ``dx`` over a three-column ``Float64`` frame built from the aligned HLC lists.
    """
    return materialize({HIGH: high, LOW: low, CLOSE: close}, dx(pl.col(HIGH), pl.col(LOW), pl.col(CLOSE), window))


class TestDxContract:
    """
    Type, shape, and lazy/eager guarantees.
    """

    def test_over_partitions_independently(self) -> None:
        """
        Verifies that under ``.over`` the recursions reset per group and never span boundaries.
        """
        frame = pl.DataFrame(
            {
                GROUP_KEY: ["a"] * 4 + ["b"] * 4,
                HIGH: [10.0, 11.0, 12.0, 11.5, 20.0, 21.0, 22.0, 21.5],
                LOW: [9.0, 10.0, 11.0, 10.5, 19.0, 20.0, 21.0, 20.5],
                CLOSE: [9.5, 10.5, 11.5, 11.0, 19.5, 20.5, 21.5, 21.0],
            }
        )
        expr = dx(pl.col(HIGH), pl.col(LOW), pl.col(CLOSE), 2).over(GROUP_KEY)
        grouped = frame.select(expr.alias("y"))["y"].to_list()
        group_a = apply_dx([10.0, 11.0, 12.0, 11.5], [9.0, 10.0, 11.0, 10.5], [9.5, 10.5, 11.5, 11.0], 2)
        group_b = apply_dx([20.0, 21.0, 22.0, 21.5], [19.0, 20.0, 21.0, 20.5], [19.5, 20.5, 21.5, 21.0], 2)
        assert_matches(grouped, group_a + group_b)


class TestDxEdge:
    """
    Boundaries, warm-up, and null / NaN handling.
    """

    def test_window_below_one_raises(self) -> None:
        """
        Verifies that ``window < 1`` raises ``ValueError``.
        """
        with pytest.raises(ValueError, match="window must be >= 1"):
            dx(pl.col(HIGH), pl.col(LOW), pl.col(CLOSE), 0)

    def test_single_row(self) -> None:
        """
        Verifies that a one-row series with ``window > 1`` is all warm-up (one null).
        """
        assert_matches(apply_dx([10.0], [9.0], [9.5], 2), [None])

    def test_all_null(self) -> None:
        """
        Verifies that an all-null series yields an all-null output.
        """
        assert_matches(apply_dx([None] * 4, [None] * 4, [None] * 4, 2), [None, None, None, None])

    def test_null_bridged(self) -> None:
        """
        Verifies that an interior ``null`` is bridged: the recursion carries its state across the gap.
        """
        high = [10.0, 11.0, 12.0, None, 13.0, 13.0, 14.0, 13.5]
        low = [9.0, 10.0, 11.0, 10.5, 12.0, 11.5, 13.0, 12.5]
        close = [9.5, 10.5, 11.5, 11.0, 12.5, 12.0, 13.5, 13.0]
        assert_matches(apply_dx(high, low, close, 2), dx_reference(high, low, close, 2))

    def test_nan_latches(self) -> None:
        """
        Verifies that a NaN latches (matching the naive reference).
        """
        high = [10.0, 11.0, 12.0, 12.0, 13.0, math.nan, 14.0, 13.5]
        low = [9.0, 10.0, 11.0, 10.5, 12.0, 11.5, 13.0, 12.5]
        close = [9.5, 10.5, 11.5, 11.0, 12.5, 12.0, 13.5, 13.0]
        assert_matches(apply_dx(high, low, close, 2), dx_reference(high, low, close, 2))

    def test_warmup_null_count(self) -> None:
        """
        Verifies that the first ``window - 1`` rows are null (warm-up).
        """
        result = apply_dx([10.0, 11.0, 12.0, 11.5], [9.0, 10.0, 11.0, 10.5], [9.5, 10.5, 11.5, 11.0], 2)
        assert result[0] is None
        assert result[1] is not None

    def test_window_exceeds_length(self) -> None:
        """
        Verifies that a window exceeding the series length yields an all-null output.
        """
        assert_matches(apply_dx([10.0, 11.0, 12.0], [9.0, 10.0, 11.0], [9.5, 10.5, 11.5], 5), [None, None, None])

    def test_flat_window_is_nan(self) -> None:
        """
        Verifies that a fully flat window yields ``NaN`` after warm-up: with no movement either way both directional
        indicators are ``NaN``, so the indeterminate ``0 / 0`` spread propagates.
        """
        flat = [10.0] * 8
        assert_matches(
            apply_dx(flat, flat, flat, 3),
            [None, None, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan],
        )


class TestDxCorrectness:
    """
    Against the naive reference oracle and frozen golden-master values.
    """

    def test_matches_reference(self) -> None:
        """
        Verifies agreement with the naive closed-form reference across several windows.
        """
        high = [10.0, 11.0, 12.0, 11.5, 13.0, 12.5, 14.0, 13.5, 15.0, 14.5]
        low = [9.0, 10.0, 11.0, 10.5, 12.0, 11.5, 13.0, 12.5, 14.0, 13.5]
        close = [9.5, 10.5, 11.5, 11.0, 12.5, 12.0, 13.5, 13.0, 14.5, 14.0]
        for window in (1, 2, 3, 5):
            assert_matches(
                apply_dx(high, low, close, window),
                dx_reference(high, low, close, window),
                rel_tol=RELATIVE_TOLERANCE_PROPERTY,
                abs_tol=ABSOLUTE_TOLERANCE_PROPERTY,
            )

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference: dx(window=2) over the sample series.
        """
        high = [10.0, 11.0, 12.0, 11.5, 13.0, 12.5, 14.0]
        low = [9.0, 10.0, 11.0, 10.5, 12.0, 11.5, 13.0]
        close = [9.5, 10.5, 11.5, 11.0, 12.5, 12.0, 13.5]
        result = apply_dx(high, low, close, 2)
        assert_matches(
            [None if v is None else round(v, 4) for v in result],
            [None, 100.0, 100.0, 20.0, 76.4706, 20.0, 72.6027],
        )


class TestDxProperties:
    """
    Invariants that must hold for all inputs (property-based).
    """

    @given(case=_cases(coherent_hlc()))
    def test_matches_reference_for_any_input(
        self,
        case: tuple[list[tuple[float, float, float]], int],
    ) -> None:
        """
        Verifies that, for any positive series and window, the implementation matches the naive reference.
        """
        rows, window = case
        high, low, close = split_triples(rows)
        assert_matches(
            apply_dx(high, low, close, window),
            dx_reference(high, low, close, window),
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=ABSOLUTE_TOLERANCE_PROPERTY,
        )

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
            apply_dx(high, low, close, window),
            dx_reference(high, low, close, window),
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=ABSOLUTE_TOLERANCE_PROPERTY,
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
        Verifies that ``dx`` is scale-invariant: scaling every input value by a constant ``k`` leaves the output
        unchanged -- ``dx(k * x) == dx(x)``. ``k`` is a power of two, so the rescale is exact and adds no
        floating-point error.
        """
        k = 2.0**exponent
        rows, window = case
        high, low, close = split_triples(rows)
        base = apply_dx(high, low, close, window)
        scaled = apply_dx(
            [value * k for value in high], [value * k for value in low], [value * k for value in close], window
        )
        assert_scale_homogeneous(scaled, base, k=k, degree=0)

    @given(case=_cases(coherent_hlc()))
    def test_bounded(
        self,
        case: tuple[list[tuple[float, float, float]], int],
    ) -> None:
        """
        Verifies that every defined value lies within ``[0, 100]`` for well-formed OHLC bars.
        """
        rows, window = case
        high, low, close = split_triples(rows)
        for value in apply_dx(high, low, close, window):
            if value is not None and not math.isnan(value):
                assert -BOUND_MARGIN <= value <= 100.0 + BOUND_MARGIN
