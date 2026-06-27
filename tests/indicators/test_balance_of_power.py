"""
Tests for ``pomata.indicators.balance_of_power`` — the Balance of Power (a per-bar close-vs-open over range gauge).

``balance_of_power`` is multi-input and elementwise, so tests use a local ``apply_balance_of_power`` helper to
materialize the factory over a four-column ``Float64`` frame; ``assert_matches`` and the naive
``balance_of_power_reference`` oracle are shared across the suite.

The ladder is the canonical one: contract (type / shape / lazy-eager / ``.over`` identity), edge (flat bar / single-row
/ null / NaN), correctness (vs the closed-form reference and a frozen golden master), and properties (reference
agreement incl. missing data and scale-invariance — ``balance_of_power`` is scale-invariant, so it carries no
large-magnitude tier). Categories are split into classes; cross-cutting categories use markers (see
``tests/README.md``).
"""

import math
from collections.abc import Sequence

import polars as pl
from hypothesis import given
from hypothesis import strategies as st
from polars.testing import assert_frame_equal
from tests.indicators.oracles import balance_of_power_reference
from tests.support import (
    CLOSE,
    GROUP_KEY,
    HIGH,
    LOW,
    OPEN,
    assert_matches,
    assert_scale_homogeneous,
    coherent_ohlc,
    coherent_ohlc_with_missing,
    materialize,
    split_quads,
)

from pomata.indicators import balance_of_power

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- derived, not chosen; the rationale is the shared method in CORRECTNESS.md, the numbers are this
# indicator's. To add an indicator, set its three facts here; the property tier below is then the same shape as every
# other indicator's.
#   1. warmup  W = 0   (windowless and elementwise: every row is defined from row 0 -- each bar uses only its own OHLC)
#   2. memory  the oracle shares pomata's per-row recomputation, so the property holds from row 0 (M = 0); with W = 0
#              there is no warm-up to outlast, so a case is simply a series of bars -- every row is output
#   3. domain  coherent_ohlc(): coherent positive-finite OHLC bars; SERIES_MAX bars span several total sizes
# balance_of_power is a bounded scale-INVARIANT ratio ((close - open) / (high - low)): its value is O(1) (in [-1, 1] for
# a well-formed bar) whatever the input magnitude, so its tolerance is ABSOLUTE (never input_scale-sized), and it
# carries a scale-INVARIANCE property in place of the homogeneity / large-magnitude tests of a scale-dependent indicator
# -- a large-magnitude test would be vacuous because the common scale cancels in the ratio. balance_of_power has no
# window parameter, so ``_cases`` draws only the series (no window to couple, hence no ``window`` in the unpacked pair).
# Repetitions N are the shared CI profile (tests/conftest.py); override per-test only if its parameter space is larger.
# ----------------------------------------------------------------------------------------------------------------------
SERIES_MAX = 50


@st.composite
def _cases[T](draw: st.DrawFn, bars: st.SearchStrategy[T]) -> list[T]:
    """
    A series of bars sized from the facts above. balance_of_power is windowless (W = 0), so -- unlike the windowed
    indicators' ``(series, window)`` pair -- a case is just the series: every row is output, never warm-up.
    """
    # NOTE: windowless -- returns the bare series (no window to couple length to); the W + D coupling of the windowed
    # ``_cases`` is vacuous here because W = 0 and every drawn row is already a defined output.
    return draw(st.lists(bars, min_size=1, max_size=SERIES_MAX))


def apply_balance_of_power(
    open: Sequence[float | None],
    high: Sequence[float | None],
    low: Sequence[float | None],
    close: Sequence[float | None],
) -> list[float | None]:
    """
    Materialize ``balance_of_power`` over a four-column ``Float64`` frame built from the aligned OHLC lists.
    """
    return materialize(
        {OPEN: open, HIGH: high, LOW: low, CLOSE: close},
        balance_of_power(pl.col(OPEN), pl.col(HIGH), pl.col(LOW), pl.col(CLOSE)),
    )


class TestBalanceOfPowerContract:
    """
    Type, shape, and lazy/eager guarantees.
    """

    def test_returns_expr(self) -> None:
        """
        Verifies that the factory returns a ``pl.Expr`` without touching a frame.
        """
        assert isinstance(balance_of_power(pl.col(OPEN), pl.col(HIGH), pl.col(LOW), pl.col(CLOSE)), pl.Expr)

    def test_preserves_length_and_dtype(self) -> None:
        """
        Verifies that the output has one value per input row and is ``Float64``.
        """
        frame = pl.DataFrame({OPEN: [10.0, 11.0], HIGH: [11.0, 13.0], LOW: [9.0, 10.0], CLOSE: [10.5, 12.0]})
        result = frame.select(balance_of_power(pl.col(OPEN), pl.col(HIGH), pl.col(LOW), pl.col(CLOSE)).alias("y"))
        assert result.height == frame.height
        assert result.schema["y"] == pl.Float64

    def test_lazy_eager_parity(self) -> None:
        """
        Verifies that eager and lazy application produce identical materialized output.
        """
        frame = pl.DataFrame({OPEN: [10.0, 11.0], HIGH: [11.0, 13.0], LOW: [9.0, 10.0], CLOSE: [10.5, 12.0]})
        expr = balance_of_power(pl.col(OPEN), pl.col(HIGH), pl.col(LOW), pl.col(CLOSE)).alias("y")
        result_eager = frame.select(expr)
        result_lazy = frame.lazy().select(expr).collect()
        assert_frame_equal(result_eager, result_lazy)

    def test_over_is_identity(self) -> None:
        """
        Verifies that ``.over`` is optional for this elementwise transform: partitioning by group equals the
        un-partitioned call (no cross-bar state can leak across group boundaries).
        """
        frame = pl.DataFrame(
            {
                GROUP_KEY: ["a", "a", "b", "b"],
                OPEN: [10.0, 11.0, 20.0, 21.0],
                HIGH: [11.0, 13.0, 21.0, 23.0],
                LOW: [9.0, 10.0, 19.0, 20.0],
                CLOSE: [10.5, 12.0, 20.5, 22.0],
            }
        )
        expr = balance_of_power(pl.col(OPEN), pl.col(HIGH), pl.col(LOW), pl.col(CLOSE))
        plain = frame.select(expr.alias("y"))["y"].to_list()
        grouped = frame.select(expr.over(GROUP_KEY).alias("y"))["y"].to_list()
        assert_matches(plain, grouped)


class TestBalanceOfPowerEdge:
    """
    Flat bar, single-row, and null / NaN handling.
    """

    def test_flat_bar_is_zero(self) -> None:
        """
        Verifies that a flat bar (``high == low``) yields ``0`` (zero range, no directional power).
        """
        assert_matches(apply_balance_of_power([10.0, 12.0], [11.0, 12.0], [9.0, 12.0], [10.5, 11.0]), [0.25, 0.0])

    def test_single_row(self) -> None:
        """
        Verifies behavior on a one-element series: the per-bar value is defined from row 0.
        """
        assert_matches(apply_balance_of_power([10.0], [12.0], [8.0], [11.0]), [0.25])

    def test_empty(self) -> None:
        """
        Verifies behavior on an empty series.
        """
        assert_matches(apply_balance_of_power([], [], [], []), [])

    def test_all_null(self) -> None:
        """
        Verifies that all-null inputs yield all null.
        """
        assert_matches(
            apply_balance_of_power([None, None, None], [None, None, None], [None, None, None], [None, None, None]),
            [None, None, None],
        )

    def test_null_propagates(self) -> None:
        """
        Verifies that a ``null`` in any input yields ``null`` for that row.
        """
        assert_matches(apply_balance_of_power([10.0, 10.0], [12.0, 12.0], [8.0, 8.0], [11.0, None]), [0.25, None])

    def test_nan_propagates(self) -> None:
        """
        Verifies that a ``NaN`` in any input (non-flat bar) yields ``NaN`` for that row.
        """
        assert_matches(
            apply_balance_of_power([10.0, 10.0], [12.0, math.nan], [8.0, 8.0], [11.0, 11.0]), [0.25, math.nan]
        )


class TestBalanceOfPowerCorrectness:
    """
    Against the naive reference oracle and frozen golden-master values.
    """

    def test_matches_reference(self) -> None:
        """
        Verifies agreement with the naive closed-form reference on a sample of bars.
        """
        open_ = [10.0, 11.0, 12.0, 11.0, 13.0]
        high = [11.0, 13.0, 12.0, 13.0, 14.0]
        low = [9.0, 10.0, 11.0, 10.0, 12.0]
        close = [10.5, 12.0, 11.5, 12.0, 13.5]
        assert_matches(
            apply_balance_of_power(open_, high, low, close), balance_of_power_reference(open_, high, low, close)
        )

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference: balance_of_power over [open 10, high 12, low 8, close 11/10/9] ==
        [0.25, 0, -0.25].
        """
        result = apply_balance_of_power([10.0, 10.0, 10.0], [12.0, 12.0, 12.0], [8.0, 8.0, 8.0], [11.0, 10.0, 9.0])
        assert_matches(result, [0.25, 0.0, -0.25])


class TestBalanceOfPowerProperties:
    """
    Invariants that must hold for all inputs (property-based).
    """

    @given(
        case=_cases(coherent_ohlc()),
    )
    def test_matches_reference_for_any_input(
        self,
        case: list[tuple[float, float, float, float]],
    ) -> None:
        """
        Verifies that, for any OHLC series, the implementation matches the naive reference.
        """
        rows = case
        open_, high, low, close = split_quads(rows)
        assert_matches(
            apply_balance_of_power(open_, high, low, close), balance_of_power_reference(open_, high, low, close)
        )

    @given(
        case=_cases(coherent_ohlc()),
        exponent=st.sampled_from([-4, -3, -2, -1, 1, 2, 3, 4]),
    )
    def test_scale_invariance(
        self,
        case: list[tuple[float, float, float, float]],
        exponent: int,
    ) -> None:
        """
        Verifies that ``balance_of_power`` is scale-invariant: scaling all of OHLC by a positive ``k`` leaves it
        unchanged. ``k`` is a power of two so the rescaling is lossless and cannot introduce a floating-point artifact.
        """
        k = 2.0**exponent
        rows = case
        open_, high, low, close = split_quads(rows)
        base = apply_balance_of_power(open_, high, low, close)
        scaled = apply_balance_of_power(
            [value * k for value in open_],
            [value * k for value in high],
            [value * k for value in low],
            [value * k for value in close],
        )
        assert_scale_homogeneous(scaled, base, k=k, degree=0)

    @given(
        case=_cases(coherent_ohlc_with_missing()),
    )
    def test_matches_reference_under_missing_data(
        self,
        case: list[tuple[float | None, float | None, float | None, float | None]],
    ) -> None:
        """
        Verifies that, for inputs freely mixing null / NaN / finite, the implementation matches the naive reference.
        """
        rows = case
        open_, high, low, close = split_quads(rows)
        assert_matches(
            apply_balance_of_power(open_, high, low, close), balance_of_power_reference(open_, high, low, close)
        )
