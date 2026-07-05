"""
Tests for ``pomata.indicators.midprice`` — the mean of a window's highest high and lowest low.

``midprice`` is multi-input and windowed, so tests use a local ``apply_midprice`` helper to materialize the factory over
a two-column ``Float64`` frame; ``assert_matches`` and the naive ``midprice_reference`` oracle are shared across the
suite.

The ladder is the canonical one: contract (type / shape / lazy-eager / ``.over`` independence), edge (warm-up / window
boundaries / single-row / null / NaN), correctness (vs the closed-form reference and a frozen golden master), and
properties (reference agreement incl. missing data, scale-homogeneity, and large-magnitude stability). Categories are
split into classes; cross-cutting categories use markers (see ``tests/README.md``).
"""

import math
from collections.abc import Sequence

import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st
from tests.indicators.oracles import midprice_reference
from tests.support import (
    ABSOLUTE_TOLERANCE_REFERENCE,
    ABSOLUTE_TOLERANCE_STREAMING,
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
    materialize,
    split_pairs,
)

from pomata.indicators import midprice

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- derived, not chosen; the rationale is the shared method in CORRECTNESS.md, the numbers are this
# indicator's. To add an indicator, set its three facts here; the property tier below is then the same shape as every
# other indicator's.
#   1. warmup  W(window) = window - 1   (the window must hold ``window`` non-null values before a result is emitted)
#   2. memory  the oracle is windowed like pomata, so the property holds from the first defined row (M = 0); each
#              example carries D in [window, 2 * window] defined bars -- one window of output, never all warm-up
#   3. domain  coherent (high >= low) positive-finite bars over the test's regime; midprice takes the window max of
#              ``high`` and min of ``low`` (no squaring), so no subnormal-square floor is needed
# Windows span ``window_min`` .. WINDOW_MAX. Repetitions N are the shared CI profile (tests/conftest.py); override
# per-test only if its parameter space is larger.
# ----------------------------------------------------------------------------------------------------------------------


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


def apply_midprice(
    high: Sequence[float | None],
    low: Sequence[float | None],
    window: int,
) -> list[float | None]:
    """
    Materialize ``midprice`` over a two-column ``Float64`` frame built from the aligned ``high`` and ``low`` lists.
    """
    return materialize({HIGH: high, LOW: low}, midprice(pl.col(HIGH), pl.col(LOW), window))


class TestMidpriceContract:
    """
    Type, shape, and lazy/eager guarantees.
    """

    def test_over_partitions_independently(self) -> None:
        """
        Verifies that under ``.over`` the window resets per group and never spans group boundaries.
        """
        frame = pl.DataFrame(
            {
                GROUP_KEY: ["a", "a", "a", "b", "b", "b"],
                HIGH: [11.0, 12.0, 13.0, 21.0, 22.0, 23.0],
                LOW: [9.0, 10.0, 11.0, 19.0, 20.0, 21.0],
            }
        )
        result = frame.select(midprice(pl.col(HIGH), pl.col(LOW), 2).over(GROUP_KEY).alias("y"))["y"].to_list()
        assert_matches(result, [None, 10.5, 11.5, None, 20.5, 21.5])


class TestMidpriceEdge:
    """
    Boundaries, warm-up, and null / NaN handling.
    """

    def test_window_below_one_raises(self) -> None:
        """
        Verifies that ``window < 1`` raises ``ValueError``.
        """
        with pytest.raises(ValueError, match="window must be >= 1"):
            midprice(pl.col(HIGH), pl.col(LOW), 0)

    def test_all_null(self) -> None:
        """
        Verifies that an all-null input yields an all-null output (each rolling extreme needs ``window`` non-null
        values).
        """
        assert_matches(apply_midprice([None, None, None], [None, None, None], 2), [None, None, None])

    def test_warmup_null_count(self) -> None:
        """
        Verifies that the first ``window - 1`` rows are null (warm-up) and the first full window is defined.
        """
        result = apply_midprice([11.0, 12.0, 13.0, 14.0, 15.0], [9.0, 10.0, 11.0, 12.0, 13.0], 3)
        assert result[:2] == [None, None]
        assert result[2] is not None

    def test_window_one_is_price_median(self) -> None:
        """
        Verifies that ``window == 1`` reduces to the per-bar ``(high + low) / 2``.
        """
        result = apply_midprice([11.0, 12.0, 13.0], [9.0, 10.0, 11.0], 1)
        assert_matches(result, [10.0, 11.0, 12.0])

    def test_single_row(self) -> None:
        """
        Verifies behavior on a one-element series: ``window == 1`` returns the midpoint, a larger window is warm-up.
        """
        assert_matches(apply_midprice([11.0], [9.0], 1), [10.0])
        assert_matches(apply_midprice([11.0], [9.0], 3), [None])

    def test_window_exceeds_length(self) -> None:
        """
        Verifies that a short series whose window exceeds the length is all warm-up (all-null output).
        """
        assert_matches(apply_midprice([11.0, 12.0, 13.0], [9.0, 10.0, 11.0], 5), [None, None, None])

    def test_null_in_window_is_null(self) -> None:
        """
        Verifies that a ``null`` in either input's window yields ``null`` there, recovering once the window clears.
        """
        result = apply_midprice([11.0, None, 13.0, 14.0], [9.0, 10.0, 11.0, 12.0], 2)
        assert_matches(result, [None, None, None, 12.5])

    def test_nan_propagates(self) -> None:
        """
        Verifies that a ``NaN`` in the window yields ``NaN`` there (``null`` still takes precedence over ``NaN``).
        """
        result = apply_midprice([11.0, math.nan, 13.0, 14.0], [9.0, 10.0, 11.0, 12.0], 2)
        assert_matches(result, [None, math.nan, math.nan, 12.5])


class TestMidpriceCorrectness:
    """
    Against the naive reference oracle and frozen golden-master values.
    """

    def test_matches_reference(self) -> None:
        """
        Verifies agreement with the naive closed-form reference across several windows.
        """
        high = [11.0, 12.0, 13.0, 12.5, 14.0, 15.0, 14.5, 16.0]
        low = [9.0, 10.0, 11.0, 11.0, 12.0, 13.0, 12.5, 14.0]
        for window in (1, 2, 3, 4, 5):
            result = apply_midprice(high, low, window)
            assert_matches(result, midprice_reference(high, low, window))

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference: midprice(window=3) over the sample bars == [None, None, 11, 11.5, 12.5].
        """
        result = apply_midprice([11.0, 12.0, 13.0, 12.5, 14.0], [9.0, 10.0, 11.0, 11.0, 12.0], 3)
        assert_matches(result, [None, None, 11.0, 11.5, 12.5])


class TestMidpriceProperties:
    """
    Invariants that must hold for all inputs (property-based).
    """

    # NOTE: exact transform -- implementation and oracle compute identical arithmetic, residual is zero, so a fixed
    # reference band applies here (not input_scale-sized like the sum-based degree-1 kernels).
    @given(case=_cases(coherent_hl()))
    def test_matches_reference_for_any_input(
        self,
        case: tuple[list[tuple[float, float]], int],
    ) -> None:
        """
        Verifies that, for any aligned high/low series and window, the implementation matches the naive reference.
        """
        rows, window = case
        high, low = split_pairs(rows)
        assert_matches(
            apply_midprice(high, low, window),
            midprice_reference(high, low, window),
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
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
        Verifies that, for positive ``k``, midprice is homogeneous of degree 1: ``midprice(k * h, k * l) == k * …``.
        ``k`` is a power of two so the rescaling is lossless and cannot perturb the windowed extremes.
        """
        k = 2.0**exponent
        rows, window = case
        high, low = split_pairs(rows)
        result_base = apply_midprice(high, low, window)
        result_scaled = apply_midprice([value * k for value in high], [value * k for value in low], window)
        assert_scale_homogeneous(result_scaled, result_base, k=k, degree=1)

    @given(case=_cases(coherent_hl_with_missing()))
    def test_matches_reference_under_missing_data(
        self,
        case: tuple[list[tuple[float | None, float | None]], int],
    ) -> None:
        """
        Verifies that, for inputs freely mixing null / NaN / finite, the implementation matches the naive reference.
        """
        rows, window = case
        high, low = split_pairs(rows)
        assert_matches(
            apply_midprice(high, low, window),
            midprice_reference(high, low, window),
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

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
        Verifies that at extreme positive magnitudes the implementation stays finite where the reference is and agrees.
        """
        rows, window = case
        high = [row[0] * scale for row in rows]
        low = [row[1] * scale for row in rows]
        assert_matches(
            apply_midprice(high, low, window),
            midprice_reference(high, low, window),
            rel_tol=RELATIVE_TOLERANCE_SCALE,
            abs_tol=ABSOLUTE_TOLERANCE_STREAMING,
        )
