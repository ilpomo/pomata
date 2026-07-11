"""
Tests for ``pomata.indicators.trix`` — the one-period rate of change of a triple-smoothed EMA.

``trix`` is single-input and built from three chained EMAs plus a rate of change, so tests use the shared ``apply_expr``
helper to materialize the factory over a one-column ``Float64`` frame; ``assert_matches`` and the naive
``trix_reference`` oracle are shared across the suite. Inputs are positive (the triple EMA in the rate-of-change
denominator stays positive), and TRIX is scale-invariant — so it carries a scale-invariance property in place of the
homogeneity / large-magnitude tests used for scale-dependent indicators.

The ladder is the canonical one: contract, edge, correctness (vs the closed-form reference and a frozen golden master),
and properties (reference agreement incl. missing data, scale-invariance). Categories are split into classes;
cross-cutting categories use markers (see ``tests/README.md``).
"""

import math

import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st
from tests.indicators.oracles import trix_reference
from tests.support import (
    ABSOLUTE_TOLERANCE_REFERENCE,
    COLUMN_X,
    GROUP_KEY,
    RELATIVE_TOLERANCE_PROPERTY,
    apply_expr,
    assert_matches,
    assert_scale_homogeneous,
    positive_missing_data,
)

from pomata.indicators import trix

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- derived, not chosen; the rationale is the shared method in CORRECTNESS.md, the numbers are this
# indicator's. To add an indicator, set its three facts here; the property tier below is then the same shape as every
# other indicator's.
#   1. warmup  W(window) = 3 * (window - 1) + 1   (three chained EMAs plus the one-period rate of change of the
#              triple-smoothed line)
#   2. memory  the oracle shares pomata's recursive EMA seeding, so the property holds from the first defined row
#              (M = 0); each example carries D in [window, 2 * window] defined values -- one window of output, never all
#              warm-up
#   3. domain  positive finite values (the triple EMA in the rate-of-change denominator stays positive); windows span
#              1 .. WINDOW_MAX
# TRIX is scale-INVARIANT (a percentage rate of change): its value is O(1) whatever the input magnitude, so its
# tolerance is ABSOLUTE (never input_scale-sized), and it carries a scale-INVARIANCE property in place of the
# homogeneity / large-magnitude tests of a scale-dependent indicator -- a large-magnitude test would be vacuous because
# the common scale cancels in the ratio. Repetitions N are the shared CI profile (tests/conftest.py); override per-test
# only if its parameter space is larger.
# ----------------------------------------------------------------------------------------------------------------------
WINDOW_MAX = 15


@st.composite
def _cases[T](draw: st.DrawFn, values: st.SearchStrategy[T]) -> tuple[list[T], int]:
    """
    A (series, window) pair sized from the facts above: ``window`` over its regimes, length = warm-up + a window of
    defined values, so every example has output to check (never an all-warm-up series, the waste a ``window`` decoupled
    from the length would cause).
    """
    window = draw(st.integers(min_value=1, max_value=WINDOW_MAX))
    defined = draw(st.integers(min_value=window, max_value=2 * window))
    length = (3 * (window - 1) + 1) + defined
    return draw(st.lists(values, min_size=length, max_size=length)), window


class TestTrixContract:
    """
    Type, shape, and lazy/eager guarantees.
    """

    def test_over_partitions_independently(self) -> None:
        """
        Verifies that under ``.over`` the EMA chain resets per group and never spans group boundaries.
        """
        frame = pl.DataFrame(
            {
                GROUP_KEY: ["a"] * 6 + ["b"] * 6,
                COLUMN_X: [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0],
            }
        )
        expr = trix(pl.col(COLUMN_X), 2).over(GROUP_KEY).round(4)
        result = frame.select(expr.alias("y"))["y"].to_list()
        assert_matches(result, [None, None, None, None, 8.6957, 8.0, None, None, None, None, 8.6957, 8.0])


class TestTrixEdge:
    """
    Boundaries, warm-up, and null / NaN handling.
    """

    def test_window_below_one_raises(self) -> None:
        """
        Verifies that ``window < 1`` raises ``ValueError``.
        """
        with pytest.raises(ValueError, match="window must be >= 1"):
            trix(pl.col(COLUMN_X), 0)

    def test_single_row(self) -> None:
        """
        Verifies behavior on a one-element series: the lone value is always warm-up.
        """
        assert_matches(apply_expr([42.0], trix(pl.col(COLUMN_X), 2)), [None])

    def test_all_null(self) -> None:
        """
        Verifies that an all-null series yields all null.
        """
        assert_matches(apply_expr([None, None, None, None], trix(pl.col(COLUMN_X), 2)), [None, None, None, None])

    def test_null_bridged(self) -> None:
        """
        Verifies that an interior ``null`` is bridged: the recursion carries its state across the gap.
        """
        values = [10.0, 11.0, 12.0, None, 14.0, 14.0, 16.0, 17.0]
        assert_matches(apply_expr(values, trix(pl.col(COLUMN_X), 2)), trix_reference(values, 2))

    def test_nan_latches(self) -> None:
        """
        Verifies that a NaN latches (matching the naive reference).
        """
        values = [10.0, 11.0, 12.0, 12.0, 14.0, math.nan, 16.0, 17.0]
        assert_matches(apply_expr(values, trix(pl.col(COLUMN_X), 2)), trix_reference(values, 2))

    def test_warmup_null_count(self) -> None:
        """
        Verifies the warm-up is ``3 * (window - 1) + 1`` rows (three chained EMAs plus the one-period rate of change).
        """
        result = apply_expr([10.0, 11.0, 12.0, 13.0, 14.0, 15.0], trix(pl.col(COLUMN_X), 2))
        assert result[:4] == [None, None, None, None]
        assert result[4] is not None

    def test_window_exceeds_length(self) -> None:
        """
        Verifies that when ``window`` exceeds the series length the whole output is null (the chain never warms up).
        """
        assert_matches(apply_expr([1.0, 2.0, 3.0], trix(pl.col(COLUMN_X), 5)), [None, None, None])


class TestTrixCorrectness:
    """
    Against the naive reference oracle and frozen golden-master values.
    """

    def test_matches_reference(self) -> None:
        """
        Verifies agreement with the naive closed-form reference across several windows.
        """
        values = [10.0, 11.0, 12.0, 11.0, 13.0, 14.0, 13.0, 15.0, 14.0, 16.0, 15.0, 17.0]
        for window in (1, 2, 3):
            assert_matches(apply_expr(values, trix(pl.col(COLUMN_X), window)), trix_reference(values, window))

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference: trix(window=2) over the sample series.
        """
        result = apply_expr([10.0, 11.0, 12.0, 11.0, 13.0, 14.0, 13.0, 15.0], trix(pl.col(COLUMN_X), 2).round(4))
        assert_matches(result, [None, None, None, None, 5.4718, 7.4466, 2.989, 5.4253])


class TestTrixProperties:
    """
    Invariants that must hold for all inputs (property-based).
    """

    @given(
        case=_cases(st.floats(min_value=1.0, max_value=1e4, allow_nan=False, allow_infinity=False)),
    )
    def test_matches_reference_for_any_input(
        self,
        case: tuple[list[float], int],
    ) -> None:
        """
        Verifies that, for any positive series and window, the implementation matches the naive reference.
        """
        values, window = case
        assert_matches(
            apply_expr(values, trix(pl.col(COLUMN_X), window)),
            trix_reference(values, window),
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    @given(
        case=_cases(positive_missing_data()),
    )
    def test_matches_reference_under_missing_data(
        self,
        case: tuple[list[float | None], int],
    ) -> None:
        """
        Verifies that, for positive inputs freely mixing null / NaN, the implementation matches the naive reference.
        """
        values, window = case
        assert_matches(
            apply_expr(values, trix(pl.col(COLUMN_X), window)),
            trix_reference(values, window),
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    @given(
        case=_cases(st.floats(min_value=1.0, max_value=1e3, allow_nan=False, allow_infinity=False)),
        exponent=st.sampled_from([-4, -3, -2, -1, 1, 2, 3, 4]),
    )
    def test_scale_invariance(
        self,
        case: tuple[list[float], int],
        exponent: int,
    ) -> None:
        """
        Verifies that ``trix`` is scale-invariant: scaling every input value by a constant ``k`` leaves the output
        unchanged -- ``trix(k * x) == trix(x)``. ``k`` is a power of two, so the rescale is exact and adds no
        floating-point error.
        """
        k = 2.0**exponent
        values, window = case
        result_base = apply_expr(values, trix(pl.col(COLUMN_X), window))
        result_scaled = apply_expr([value * k for value in values], trix(pl.col(COLUMN_X), window))
        assert_scale_homogeneous(result_scaled, result_base, k=k, degree=0)
