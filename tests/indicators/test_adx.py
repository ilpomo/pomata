"""
Tests for ``pomata.indicators.adx`` — the Average Directional Index (ADX).

``adx`` is multi-input (high, low, close), so tests use a local ``apply_adx`` helper to materialize the factory over a
three-column ``Float64`` frame; ``assert_matches`` and the ``adx_reference`` oracle are shared across the suite. It is a
trend-strength index bounded in ``[0, 100]`` and scale-invariant — so it carries scale-invariance and boundedness
properties in place of homogeneity / large-magnitude.

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
from tests.indicators.oracles import adx_reference
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

from pomata.indicators import adx

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- derived, not chosen; the rationale is the shared method in CORRECTNESS.md, the numbers are this
# indicator's. To add an indicator, set its three facts here; the property tier below is then the same shape as every
# other indicator's.
#   1. warmup  W(window) = 2 * (window - 1)   (a deep warm-up: the ADX smooths the already-smoothed ``dx``, so the
#              ``dx`` warm-up of ``window - 1`` stacks with the ``rma`` warm-up of another ``window - 1``)
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
    length = 2 * (window - 1) + defined
    return draw(st.lists(bars, min_size=length, max_size=length)), window


def apply_adx(
    high: Sequence[float | None],
    low: Sequence[float | None],
    close: Sequence[float | None],
    window: int,
) -> list[float | None]:
    """
    Materialize ``adx`` over a three-column ``Float64`` frame built from the aligned HLC lists.
    """
    return materialize({HIGH: high, LOW: low, CLOSE: close}, adx(pl.col(HIGH), pl.col(LOW), pl.col(CLOSE), window))


class TestAdxContract:
    """
    Type, shape, and lazy/eager guarantees.
    """

    def test_over_partitions_independently(self) -> None:
        """
        Verifies that under ``.over`` the recursions reset per group and never span boundaries.
        """
        frame = pl.DataFrame(
            {
                GROUP_KEY: ["a"] * 5 + ["b"] * 5,
                HIGH: [10.0, 11.0, 12.0, 11.5, 13.0, 20.0, 21.0, 22.0, 21.5, 23.0],
                LOW: [9.0, 10.0, 11.0, 10.5, 12.0, 19.0, 20.0, 21.0, 20.5, 22.0],
                CLOSE: [9.5, 10.5, 11.5, 11.0, 12.5, 19.5, 20.5, 21.5, 21.0, 22.5],
            }
        )
        expr = adx(pl.col(HIGH), pl.col(LOW), pl.col(CLOSE), 2).over(GROUP_KEY)
        grouped = frame.select(expr.alias("y"))["y"].to_list()
        group_a = apply_adx(
            [10.0, 11.0, 12.0, 11.5, 13.0], [9.0, 10.0, 11.0, 10.5, 12.0], [9.5, 10.5, 11.5, 11.0, 12.5], 2
        )
        group_b = apply_adx(
            [20.0, 21.0, 22.0, 21.5, 23.0], [19.0, 20.0, 21.0, 20.5, 22.0], [19.5, 20.5, 21.5, 21.0, 22.5], 2
        )
        assert_matches(grouped, group_a + group_b)


class TestAdxEdge:
    """
    Boundaries, warm-up, and null / NaN handling.
    """

    def test_window_below_one_raises(self) -> None:
        """
        Verifies that ``window < 1`` raises ``ValueError``.
        """
        with pytest.raises(ValueError, match="window must be >= 1"):
            adx(pl.col(HIGH), pl.col(LOW), pl.col(CLOSE), 0)

    def test_all_null(self) -> None:
        """
        Verifies that an all-null series yields an all-null output.
        """
        assert_matches(apply_adx([None] * 4, [None] * 4, [None] * 4, 2), [None, None, None, None])

    def test_single_row(self) -> None:
        """
        Verifies that a one-row series with ``window > 1`` is all warm-up (one null).
        """
        assert_matches(apply_adx([10.0], [9.0], [9.5], 2), [None])

    def test_window_exceeds_length(self) -> None:
        """
        Verifies that a window exceeding the series length yields an all-null output.
        """
        assert_matches(apply_adx([10.0, 11.0, 12.0], [9.0, 10.0, 11.0], [9.5, 10.5, 11.5], 5), [None, None, None])

    def test_warmup_null_count(self) -> None:
        """
        Verifies the deep warm-up: the ADX smooths the already-smoothed DX, so the first two rows are null at window 2.
        """
        high = [10.0, 11.0, 12.0, 11.5, 13.0, 12.5]
        low = [9.0, 10.0, 11.0, 10.5, 12.0, 11.5]
        close = [9.5, 10.5, 11.5, 11.0, 12.5, 12.0]
        result = apply_adx(high, low, close, 2)
        assert result[:2] == [None, None]
        assert result[2] is not None

    def test_null_propagates(self) -> None:
        """
        Verifies that a ``null`` propagates (matching the naive reference).
        """
        high = [10.0, 11.0, 12.0, None, 13.0, 13.5, 14.0, 13.5, 15.0]
        low = [9.0, 10.0, 11.0, 10.5, 12.0, 11.5, 13.0, 12.5, 14.0]
        close = [9.5, 10.5, 11.5, 11.0, 12.5, 12.0, 13.5, 13.0, 14.5]
        assert_matches(apply_adx(high, low, close, 2), adx_reference(high, low, close, 2))

    def test_nan_latches(self) -> None:
        """
        Verifies that a ``NaN`` propagates (matching the naive reference).
        """
        high = [10.0, 11.0, 12.0, 12.5, 13.0, math.nan, 14.0, 13.5, 15.0]
        low = [9.0, 10.0, 11.0, 10.5, 12.0, 11.5, 13.0, 12.5, 14.0]
        close = [9.5, 10.5, 11.5, 11.0, 12.5, 12.0, 13.5, 13.0, 14.5]
        assert_matches(apply_adx(high, low, close, 2), adx_reference(high, low, close, 2))

    def test_flat_window_is_nan(self) -> None:
        """
        Verifies that a fully flat window yields ``NaN`` after warm-up: the underlying :func:`dx` is the indeterminate
        ``0 / 0`` (both directional indicators are zero), which then poisons the smoothing recursion.
        """
        flat = [10.0] * 8
        assert_matches(
            apply_adx(flat, flat, flat, 3),
            [None, None, None, None, math.nan, math.nan, math.nan, math.nan],
        )


class TestAdxCorrectness:
    """
    Against the naive reference oracle and frozen golden-master values.
    """

    def test_matches_reference(self) -> None:
        """
        Verifies agreement with the naive closed-form reference across several windows.
        """
        high = [10.0, 11.0, 12.0, 11.5, 13.0, 12.5, 14.0, 13.5, 15.0, 14.5, 16.0, 15.5]
        low = [9.0, 10.0, 11.0, 10.5, 12.0, 11.5, 13.0, 12.5, 14.0, 13.5, 15.0, 14.5]
        close = [9.5, 10.5, 11.5, 11.0, 12.5, 12.0, 13.5, 13.0, 14.5, 14.0, 15.5, 15.0]
        for window in (1, 2, 3, 5):
            assert_matches(
                apply_adx(high, low, close, window),
                adx_reference(high, low, close, window),
                rel_tol=RELATIVE_TOLERANCE_PROPERTY,
                abs_tol=ABSOLUTE_TOLERANCE_PROPERTY,
            )

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference: adx(window=2) over the sample series.
        """
        high = [10.0, 11.0, 12.0, 11.5, 13.0, 12.5, 14.0, 13.5, 15.0, 14.5]
        low = [9.0, 10.0, 11.0, 10.5, 12.0, 11.5, 13.0, 12.5, 14.0, 13.5]
        close = [9.5, 10.5, 11.5, 11.0, 12.5, 12.0, 13.5, 13.0, 14.5, 14.0]
        result = apply_adx(high, low, close, 2)
        assert_matches(
            [None if v is None else round(v, 4) for v in result],
            [None, None, 100.0, 60.0, 68.2353, 44.1176, 58.3602, 39.1801, 55.4486, 37.7243],
        )


class TestAdxProperties:
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
            apply_adx(high, low, close, window),
            adx_reference(high, low, close, window),
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
        Verifies that ``adx`` is scale-invariant under a positive common rescaling of high / low / close. ``k`` is a
        power of two so the rescaling is lossless and cannot perturb the directional comparisons that drive the index.
        """
        k = 2.0**exponent
        rows, window = case
        high, low, close = split_triples(rows)
        base = apply_adx(high, low, close, window)
        scaled = apply_adx(
            [value * k for value in high], [value * k for value in low], [value * k for value in close], window
        )
        assert_scale_homogeneous(scaled, base, k=k, degree=0)

    @given(case=_cases(coherent_hlc()))
    def test_bounded(
        self,
        case: tuple[list[tuple[float, float, float]], int],
    ) -> None:
        """
        Verifies that every defined value lies within ``[0, 100]``.
        """
        rows, window = case
        high, low, close = split_triples(rows)
        for value in apply_adx(high, low, close, window):
            if value is not None and not math.isnan(value):
                assert -BOUND_MARGIN <= value <= 100.0 + BOUND_MARGIN

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
            apply_adx(high, low, close, window),
            adx_reference(high, low, close, window),
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=ABSOLUTE_TOLERANCE_PROPERTY,
        )
