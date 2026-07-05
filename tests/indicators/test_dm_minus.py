"""
Tests for ``pomata.indicators.dm_minus`` — the Wilder-smoothed Minus Directional Movement.

``dm_minus`` is multi-input (high, low), so tests use a local ``apply_dm_minus`` helper to materialize the factory
over a two-column ``Float64`` frame; ``assert_matches`` and the naive ``dm_minus_reference`` oracle are shared across
the suite.
It is homogeneous of degree 1 in a positive rescaling, so it carries scale-homogeneity and large-magnitude properties.

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
from tests.indicators.oracles import dm_minus_reference
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

from pomata.indicators import dm_minus

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- derived, not chosen; the rationale is the shared method in CORRECTNESS.md, the numbers are this
# indicator's. To add an indicator, set its three facts here; the property tier below is then the same shape as every
# other indicator's.
#   1. warmup  W(window) = window - 1   (the rma over the raw directional movement emits only once ``window`` non-null
#              raw movements have accrued; the raw movement itself is defined from row 0, where it seeds at 0)
#   2. memory  the oracle shares pomata's recursive Wilder seeding, so the property holds from the first defined row
#              (M = 0); each example carries D in [window, 2 * window] defined bars -- one window of output, never all
#              warm-up
#   3. domain  coherent_hl(): coherent (high >= low) positive-finite bars -- directional movement is a price-unit range
#              expansion defined on well-formed bars; windows span 1 .. WINDOW_MAX
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


def apply_dm_minus(
    high: Sequence[float | None],
    low: Sequence[float | None],
    window: int,
) -> list[float | None]:
    """
    Materialize ``dm_minus`` over a two-column ``Float64`` frame built from the aligned high / low lists.
    """
    return materialize({HIGH: high, LOW: low}, dm_minus(pl.col(HIGH), pl.col(LOW), window))


class TestDmMinusContract:
    """
    Type, shape, and lazy/eager guarantees.
    """

    def test_over_partitions_independently(self) -> None:
        """
        Verifies that under ``.over`` the differencing and recursion reset per group and never span boundaries.
        """
        frame = pl.DataFrame(
            {
                GROUP_KEY: ["a"] * 4 + ["b"] * 4,
                HIGH: [10.0, 11.0, 12.0, 11.5, 20.0, 21.0, 22.0, 21.5],
                LOW: [9.0, 10.0, 11.0, 10.5, 19.0, 20.0, 21.0, 20.5],
            }
        )
        expr = dm_minus(pl.col(HIGH), pl.col(LOW), 2).over(GROUP_KEY)
        grouped = frame.select(expr.alias("y"))["y"].to_list()
        group_a = apply_dm_minus([10.0, 11.0, 12.0, 11.5], [9.0, 10.0, 11.0, 10.5], 2)
        group_b = apply_dm_minus([20.0, 21.0, 22.0, 21.5], [19.0, 20.0, 21.0, 20.5], 2)
        assert_matches(grouped, group_a + group_b)


class TestDmMinusEdge:
    """
    Boundaries, warm-up, and null / NaN handling.
    """

    def test_window_below_one_raises(self) -> None:
        """
        Verifies that ``window < 1`` raises ``ValueError``.
        """
        with pytest.raises(ValueError, match="window must be >= 1"):
            dm_minus(pl.col(HIGH), pl.col(LOW), 0)

    def test_all_null(self) -> None:
        """
        Verifies that an all-null series yields a warm-up null then zeros: a ``null`` high or low makes the raw
        directional movement ``0`` (the ``when`` comparison is not satisfied), which the ``rma`` then smooths to ``0``.
        """
        assert_matches(apply_dm_minus([None] * 4, [None] * 4, 2), [None, 0.0, 0.0, 0.0])

    def test_single_row(self) -> None:
        """
        Verifies that a one-row series with ``window > 1`` is all warm-up (one null).
        """
        assert_matches(apply_dm_minus([10.0], [9.0], 2), [None])

    def test_window_exceeds_length(self) -> None:
        """
        Verifies that a window exceeding the series length yields an all-null output.
        """
        assert_matches(apply_dm_minus([10.0, 11.0, 12.0], [9.0, 10.0, 11.0], 5), [None, None, None])

    def test_warmup_null_count(self) -> None:
        """
        Verifies that the first ``window - 1`` rows are null (warm-up, inherited from the rma).
        """
        result = apply_dm_minus([10.0, 11.0, 12.0, 11.5], [9.0, 10.0, 11.0, 10.5], 2)
        assert result[0] is None
        assert result[1] is not None

    def test_null_bridged(self) -> None:
        """
        Verifies that an interior ``null`` is bridged: the recursion carries its state across the gap.
        """
        high = [10.0, 11.0, 12.0, 11.5, 13.0, 12.5, 14.0, 13.5]
        low = [9.0, 10.0, 11.0, None, 12.0, 12.5, 13.0, 12.5]
        assert_matches(apply_dm_minus(high, low, 2), dm_minus_reference(high, low, 2))

    def test_nan_latches(self) -> None:
        """
        Verifies that a NaN propagates (matching the naive reference).
        """
        high = [10.0, 11.0, 12.0, 11.5, 13.0, 12.5, 14.0, 13.5]
        low = [9.0, 10.0, 11.0, 11.5, 12.0, math.nan, 13.0, 12.5]
        assert_matches(apply_dm_minus(high, low, 2), dm_minus_reference(high, low, 2))


class TestDmMinusCorrectness:
    """
    Against the naive reference oracle and frozen golden-master values.
    """

    def test_matches_reference(self) -> None:
        """
        Verifies agreement with the naive closed-form reference across several windows.
        """
        high = [10.0, 11.0, 12.0, 11.5, 13.0, 12.5, 14.0, 13.5, 15.0, 14.5]
        low = [9.0, 10.0, 11.0, 10.5, 12.0, 11.5, 13.0, 12.5, 14.0, 13.5]
        for window in (1, 2, 3, 5):
            assert_matches(apply_dm_minus(high, low, window), dm_minus_reference(high, low, window))

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference: dm_minus(window=2) over the sample series.
        """
        high = [10.0, 11.0, 12.0, 11.5, 13.0, 12.5, 14.0]
        low = [9.0, 10.0, 11.0, 10.5, 12.0, 11.5, 13.0]
        result = apply_dm_minus(high, low, 2)
        assert_matches(
            [None if v is None else round(v, 4) for v in result],
            [None, 0.0, 0.0, 0.25, 0.125, 0.3125, 0.1562],
        )


class TestDmMinusProperties:
    """
    Invariants that must hold for all inputs (property-based).
    """

    @given(case=_cases(coherent_hl()))
    def test_matches_reference_for_any_input(
        self,
        case: tuple[list[tuple[float, float]], int],
    ) -> None:
        """
        Verifies that, for any positive series and window, the implementation matches the naive reference.
        """
        rows, window = case
        high, low = split_pairs(rows)
        assert_matches(
            apply_dm_minus(high, low, window),
            dm_minus_reference(high, low, window),
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=input_scale([*high, *low]) * EXACT_TOLERANCE_FACTOR,
        )

    @given(
        case=_cases(coherent_hl()),
        exponent=st.sampled_from([-4, -3, -2, -1, 1, 2, 3, 4]),
    )
    def test_scale_homogeneity(
        self,
        case: tuple[list[tuple[float, float]], int],
        exponent: int,
    ) -> None:
        """
        Verifies that, for positive ``k``, ``dm_minus`` is homogeneous of degree 1: ``dm_minus(k * x) == k *
        dm_minus(x)``. ``k`` is a power of two so the rescaling is lossless and cannot introduce a floating-point
        artifact.
        """
        k = 2.0**exponent
        rows, window = case
        high, low = split_pairs(rows)
        base = apply_dm_minus(high, low, window)
        scaled = apply_dm_minus([value * k for value in high], [value * k for value in low], window)
        assert_scale_homogeneous(scaled, base, k=k, degree=1)

    @given(
        case=_cases(coherent_hl()),
        scale=st.sampled_from([1e-6, 1e6, 1e9]),
    )
    def test_matches_reference_at_large_magnitude(
        self,
        case: tuple[list[tuple[float, float]], int],
        scale: float,
    ) -> None:
        """
        Verifies that at extreme magnitudes the implementation stays finite where the reference is and agrees.
        """
        rows, window = case
        high = [row[0] * scale for row in rows]
        low = [row[1] * scale for row in rows]
        assert_matches(
            apply_dm_minus(high, low, window),
            dm_minus_reference(high, low, window),
            rel_tol=RELATIVE_TOLERANCE_SCALE,
            abs_tol=input_scale([*high, *low]) * EXACT_TOLERANCE_FACTOR,
        )

    @given(case=_cases(coherent_hl_with_missing()))
    def test_matches_reference_under_missing_data(
        self,
        case: tuple[list[tuple[float | None, float | None]], int],
    ) -> None:
        """
        Verifies that, for positive inputs freely mixing null / NaN, the implementation matches the naive reference.
        """
        rows, window = case
        high, low = split_pairs(rows)
        assert_matches(
            apply_dm_minus(high, low, window),
            dm_minus_reference(high, low, window),
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=input_scale([*high, *low]) * EXACT_TOLERANCE_FACTOR,
        )
