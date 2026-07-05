"""
Tests for ``pomata.indicators.percentage_price_oscillator`` — the Percentage Price Oscillator (the APO gap as a
percent of the slow EMA).

``percentage_price_oscillator`` is single-input and built from two EMAs, so tests use the shared ``apply_expr`` helper
to materialize the factory over a one-column ``Float64`` frame; ``assert_matches`` and the naive
``percentage_price_oscillator_reference`` oracle are shared across the suite. Because PPO divides by the slow EMA, the
tested inputs are positive (so the denominator stays positive and well-conditioned), and PPO is scale-invariant rather
than scale-homogeneous — so it carries a scale-invariance property in place of the homogeneity / large-magnitude tests
used for scale-dependent indicators.

The ladder is the canonical one: contract, edge, correctness (vs the closed-form reference and a frozen golden master),
and properties (reference agreement incl. missing data, scale-invariance). Categories are split into classes;
cross-cutting categories use markers (see ``tests/README.md``).
"""

import math

import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st
from tests.indicators.oracles import percentage_price_oscillator_reference
from tests.support import (
    ABSOLUTE_TOLERANCE_EXACT,
    ABSOLUTE_TOLERANCE_REFERENCE,
    COLUMN_X,
    GROUP_KEY,
    RELATIVE_TOLERANCE_PROPERTY,
    RELATIVE_TOLERANCE_REFERENCE,
    apply_expr,
    assert_matches,
    assert_scale_homogeneous,
    positive_missing_data,
)

from pomata.indicators import percentage_price_oscillator

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- derived, not chosen; the rationale is the shared method in CORRECTNESS.md, the numbers are this
# indicator's. To add an indicator, set its three facts here; the property tier below is then the same shape as every
# other indicator's.
#   1. warmup  W = max(window_fast, window_slow) - 1   (values are null until the slower EMA leaves its warm-up; the
#              contract requires window_fast <= window_slow, so the slow EMA is the one that warms up last)
#   2. memory  the oracle shares pomata's recursive EMA seeding, so the property holds from the first defined row
#              (M = 0); each example carries D in [window_slow, 2 * window_slow] defined values -- one slow window of
#              output, never all warm-up
#   3. domain  positive finite values (PPO divides by the slow EMA, so the denominator stays positive and
#              well-conditioned); windows span 1 .. WINDOW_MAX
# PPO is scale-INVARIANT (the price's unit cancels in the ratio): its value is O(1) whatever the input magnitude, so its
# tolerance is ABSOLUTE (never input_scale-sized), and it carries a scale-INVARIANCE property in place of the
# homogeneity / large-magnitude tests of a scale-dependent indicator -- a large-magnitude test would be vacuous because
# the common scale cancels in the ratio. Repetitions N are the shared CI profile (tests/conftest.py); override per-test
# only if its parameter space is larger.
# ----------------------------------------------------------------------------------------------------------------------
WINDOW_MAX = 15


@st.composite
def _cases[T](draw: st.DrawFn, values: st.SearchStrategy[T]) -> tuple[list[T], int, int]:
    """
    A (series, window_fast, window_slow) triple sized from the facts above honoring the contract
    ``1 <= window_fast <= window_slow``: ``window_slow`` is drawn first and ``window_fast`` is then bounded above by it
    (so the now-rejected reversed regime is never sampled), and length = warm-up (driven by ``window_slow``) + a slow
    window of defined values, so every example has output to check (never an all-warm-up series, the waste a window
    decoupled from the length would cause).
    """
    window_slow = draw(st.integers(min_value=1, max_value=WINDOW_MAX))
    window_fast = draw(st.integers(min_value=1, max_value=window_slow))
    defined = draw(st.integers(min_value=window_slow, max_value=2 * window_slow))
    length = (window_slow - 1) + defined
    return draw(st.lists(values, min_size=length, max_size=length)), window_fast, window_slow


class TestPercentagePriceOscillatorContract:
    """
    Type, shape, and lazy/eager guarantees.
    """

    def test_over_partitions_independently(self) -> None:
        """
        Verifies that under ``.over`` each EMA resets per group and never spans group boundaries.
        """
        frame = pl.DataFrame(
            {GROUP_KEY: ["a"] * 4 + ["b"] * 4, COLUMN_X: [10.0, 11.0, 12.0, 11.0, 20.0, 22.0, 24.0, 22.0]}
        )
        expr = percentage_price_oscillator(pl.col(COLUMN_X), window_fast=2, window_slow=3).over(GROUP_KEY).round(4)
        result = frame.select(expr.alias("y"))["y"].to_list()
        assert_matches(result, [None, None, 4.5455, 1.5152, None, None, 4.5455, 1.5152])


class TestPercentagePriceOscillatorEdge:
    """
    Boundaries, warm-up, and null / NaN handling.
    """

    def test_windows_below_one_raises(self) -> None:
        """
        Verifies that a window ``< 1`` raises ``ValueError``.
        """
        with pytest.raises(ValueError, match="window_fast must be >= 1"):
            percentage_price_oscillator(pl.col(COLUMN_X), window_fast=0, window_slow=3)
        with pytest.raises(ValueError, match="window_slow must be >= 1"):
            percentage_price_oscillator(pl.col(COLUMN_X), window_fast=2, window_slow=0)

    def test_fast_above_slow_raises(self) -> None:
        """
        Verifies that ``window_fast > window_slow`` raises ``ValueError`` (the fast leg must be the shorter one), while
        the equal-window case is accepted.
        """
        with pytest.raises(ValueError, match="windows must be ordered window_fast <= window_slow"):
            percentage_price_oscillator(pl.col(COLUMN_X), window_fast=5, window_slow=3)
        assert isinstance(percentage_price_oscillator(pl.col(COLUMN_X), window_fast=3, window_slow=3), pl.Expr)

    def test_warmup_null_count(self) -> None:
        """
        Verifies that values are null until the slow EMA leaves its warm-up (the first ``window_slow - 1`` rows).
        """
        result = apply_expr(
            [10.0, 11.0, 12.0, 13.0, 14.0], percentage_price_oscillator(pl.col(COLUMN_X), window_fast=2, window_slow=3)
        )
        assert result[:2] == [None, None]
        assert result[2] is not None

    def test_equal_windows_are_zero(self) -> None:
        """
        Verifies that equal fast and slow windows cancel to ``0`` once warmed up (the two EMAs are identical).
        """
        result = apply_expr(
            [10.0, 11.0, 12.0], percentage_price_oscillator(pl.col(COLUMN_X), window_fast=2, window_slow=2)
        )
        assert_matches(result, [None, 0.0, 0.0])

    def test_single_row(self) -> None:
        """
        Verifies behavior on a one-element series: the slow EMA never warms up, so the result is all warm-up.
        """
        assert_matches(
            apply_expr([42.0], percentage_price_oscillator(pl.col(COLUMN_X), window_fast=2, window_slow=3)), [None]
        )

    def test_window_exceeds_length(self) -> None:
        """
        Verifies that when the longest window exceeds the series length the whole output is null (no slow-EMA value).
        """
        assert_matches(
            apply_expr([1.0, 2.0], percentage_price_oscillator(pl.col(COLUMN_X), window_fast=2, window_slow=3)),
            [None, None],
        )

    def test_all_null(self) -> None:
        """
        Verifies that an all-null series yields all null.
        """
        assert_matches(
            apply_expr([None, None, None], percentage_price_oscillator(pl.col(COLUMN_X), window_fast=2, window_slow=3)),
            [None, None, None],
        )

    def test_null_bridged(self) -> None:
        """
        Verifies that an interior ``null`` is bridged: it yields ``null`` at its own row while the recursive EMAs resume
        afterward (matching the naive reference), rather than contaminating every later row.
        """
        values = [10.0, 11.0, 12.0, None, 14.0, 15.0, 16.0, 17.0]
        assert_matches(
            apply_expr(values, percentage_price_oscillator(pl.col(COLUMN_X), window_fast=2, window_slow=3)),
            percentage_price_oscillator_reference(values, 2, 3),
        )

    def test_nan_latches(self) -> None:
        """
        Verifies that a ``NaN`` propagates through both EMAs, yielding ``NaN``.
        """
        result = apply_expr(
            [10.0, 11.0, 12.0, math.nan, 15.0],
            percentage_price_oscillator(pl.col(COLUMN_X), window_fast=2, window_slow=3),
        )
        assert result[-1] is not None
        assert math.isnan(result[-1])

    def test_zero_slow_ema_is_nan(self) -> None:
        """
        Verifies the documented IEEE-754 behavior when the slow EMA in the denominator is ``0``: an all-zero series
        drives both EMAs to exactly ``0``, so the gap is ``0`` and ``0 / 0`` surfaces as ``NaN`` rather than raising.
        The naive reference agrees (it guards the zero denominator instead of dividing bare).
        """
        values = [0.0, 0.0, 0.0, 0.0]
        result = apply_expr(values, percentage_price_oscillator(pl.col(COLUMN_X), window_fast=2, window_slow=3))
        assert_matches(result, percentage_price_oscillator_reference(values, 2, 3))
        assert_matches(result, [None, None, math.nan, math.nan])


class TestPercentagePriceOscillatorCorrectness:
    """
    Against the naive reference oracle and frozen golden-master values.
    """

    def test_matches_reference(self) -> None:
        """
        Verifies agreement with the naive closed-form reference across several window pairs.
        """
        values = [10.0, 11.0, 12.0, 11.0, 13.0, 14.0, 13.0, 15.0, 14.0, 16.0]
        for window_fast, window_slow in ((2, 3), (3, 5), (1, 4), (4, 4)):
            result = apply_expr(
                values, percentage_price_oscillator(pl.col(COLUMN_X), window_fast=window_fast, window_slow=window_slow)
            )
            assert_matches(
                result,
                percentage_price_oscillator_reference(values, window_fast, window_slow),
                rel_tol=RELATIVE_TOLERANCE_REFERENCE,
                abs_tol=ABSOLUTE_TOLERANCE_EXACT,
            )

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference: percentage_price_oscillator(fast=2, slow=3) over the sample series.
        """
        result = apply_expr(
            [10.0, 11.0, 12.0, 11.0, 13.0, 14.0, 13.0, 15.0],
            percentage_price_oscillator(pl.col(COLUMN_X), window_fast=2, window_slow=3).round(4),
        )
        assert_matches(result, [None, None, 4.5455, 1.5152, 3.2407, 3.5613, 1.1871, 2.7484])


class TestPercentagePriceOscillatorProperties:
    """
    Invariants that must hold for all inputs (property-based).
    """

    @given(
        case=_cases(st.floats(min_value=1e-3, max_value=1e6, allow_nan=False, allow_infinity=False)),
    )
    def test_matches_reference_for_any_input(
        self,
        case: tuple[list[float], int, int],
    ) -> None:
        """
        Verifies that, for any positive series and window pair, the implementation matches the naive reference.
        """
        values, window_fast, window_slow = case
        assert_matches(
            apply_expr(
                values, percentage_price_oscillator(pl.col(COLUMN_X), window_fast=window_fast, window_slow=window_slow)
            ),
            percentage_price_oscillator_reference(values, window_fast, window_slow),
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    @given(
        case=_cases(st.floats(min_value=1.0, max_value=1e3, allow_nan=False, allow_infinity=False)),
        exponent=st.sampled_from([-4, -3, -2, -1, 1, 2, 3, 4]),
    )
    def test_scale_invariance(
        self,
        case: tuple[list[float], int, int],
        exponent: int,
    ) -> None:
        """
        Verifies that PPO is scale-invariant:
        ``percentage_price_oscillator(k * x) == percentage_price_oscillator(x)`` for positive ``k`` (the price's
        unit cancels). ``k`` is a power of two so the rescaling is lossless and cannot perturb the result through
        floating-point drift.
        """
        k = 2.0**exponent
        values, window_fast, window_slow = case
        result_base = apply_expr(
            values, percentage_price_oscillator(pl.col(COLUMN_X), window_fast=window_fast, window_slow=window_slow)
        )
        result_scaled = apply_expr(
            [value * k for value in values],
            percentage_price_oscillator(pl.col(COLUMN_X), window_fast=window_fast, window_slow=window_slow),
        )
        assert_scale_homogeneous(result_scaled, result_base, k=k, degree=0)

    @given(
        case=_cases(positive_missing_data(high=1e6)),
    )
    def test_matches_reference_under_missing_data(
        self,
        case: tuple[list[float | None], int, int],
    ) -> None:
        """
        Verifies that, for positive inputs freely mixing null / NaN, the implementation matches the naive reference.
        """
        values, window_fast, window_slow = case
        assert_matches(
            apply_expr(
                values, percentage_price_oscillator(pl.col(COLUMN_X), window_fast=window_fast, window_slow=window_slow)
            ),
            percentage_price_oscillator_reference(values, window_fast, window_slow),
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )
