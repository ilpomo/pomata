"""
Tests for ``pomata.indicators.chande_momentum_oscillator`` — the Chande Momentum Oscillator (gains minus losses
over their total).

``chande_momentum_oscillator`` is single-input, so tests use the shared ``apply_expr`` helper to materialize the
factory over a one-column ``Float64`` frame; ``assert_matches`` and the naive ``chande_momentum_oscillator_reference``
oracle are shared across the suite. ``chande_momentum_oscillator`` is a ratio bounded in ``[-100, 100]`` and
scale-invariant — so it carries a scale-invariance property (and a boundedness property) in place of the homogeneity /
large-magnitude tests used for scale-dependent indicators.

The ladder is the canonical one: contract, edge (warm-up / flat window / saturation / null / NaN), correctness (vs the
closed-form reference and a frozen golden master), and properties (reference agreement incl. missing data,
scale-invariance, boundedness). Categories are split into classes; cross-cutting categories use markers (see
``tests/README.md``).
"""

import math

import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st
from tests.indicators.oracles import chande_momentum_oscillator_reference
from tests.support import (
    ABSOLUTE_TOLERANCE_PROPERTY,
    BOUND_MARGIN,
    COLUMN_X,
    GROUP_KEY,
    RELATIVE_TOLERANCE_PROPERTY,
    apply_expr,
    assert_matches,
    assert_scale_homogeneous,
    missing_data_floats,
    subnormal_safe_floats,
)

from pomata.indicators import chande_momentum_oscillator

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- derived, not chosen; the rationale is the shared method in CORRECTNESS.md, the numbers are this
# indicator's. To add an indicator, set its three facts here; the property tier below is then the same shape as every
# other indicator's.
#   1. warmup  W(window) = window   (row 0 has no change, and the rolling gain / loss sums need ``window`` non-null
#              changes before emitting)
#   2. memory  the oracle shares pomata's seeding, so the property holds from the first defined row (M = 0); each
#              example carries D in [window, 2 * window] defined values -- one window of output, never all warm-up
#   3. domain  CMO divides two rolling gain/loss SUMS and is ill-conditioned where those sums fall tiny relative to the
#              terms that entered the window: Polars' subtract-on-exit rolling_sum keeps a residual a fresh compensated
#              sum does not. The random fuzz never builds that whole-window-near-flat regime, so the agreement tiers
#              draw the full signed [-1e6, 1e6] range and agree to 1e-9; the directed near-flat window, where the clip
#              bounds the value, is pinned deterministically in Edge
# CMO is a bounded scale-INVARIANT ratio in ``[-100, 100]``: its value is O(1) whatever the input magnitude, so its
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
    length = window + defined
    return draw(st.lists(values, min_size=length, max_size=length)), window


class TestChandeMomentumOscillatorContract:
    """
    Type, shape, and lazy/eager guarantees.
    """

    def test_over_partitions_independently(self) -> None:
        """
        Verifies that under ``.over`` the differencing and rolling sums reset per group and never span boundaries.
        """
        frame = pl.DataFrame(
            {
                GROUP_KEY: ["a"] * 5 + ["b"] * 5,
                COLUMN_X: [10.0, 11.0, 12.0, 11.0, 13.0, 20.0, 19.0, 21.0, 22.0, 20.0],
            }
        )
        expr = chande_momentum_oscillator(pl.col(COLUMN_X), 3).over(GROUP_KEY).round(4)
        result = frame.select(expr.alias("y"))["y"].to_list()
        assert_matches(result, [None, None, None, 33.3333, 50.0, None, None, None, 50.0, 20.0])


class TestChandeMomentumOscillatorEdge:
    """
    Boundaries, warm-up, saturation, and null / NaN handling.
    """

    def test_window_below_one_raises(self) -> None:
        """
        Verifies that ``window < 1`` raises ``ValueError``.
        """
        with pytest.raises(ValueError, match="window must be >= 1"):
            chande_momentum_oscillator(pl.col(COLUMN_X), 0)

    def test_single_row(self) -> None:
        """
        Verifies behavior on a one-element series: the lone value is always warm-up.
        """
        assert_matches(apply_expr([42.0], chande_momentum_oscillator(pl.col(COLUMN_X), 3)), [None])

    def test_all_null(self) -> None:
        """
        Verifies that an all-null series yields all null.
        """
        assert_matches(
            apply_expr([None, None, None, None], chande_momentum_oscillator(pl.col(COLUMN_X), 3)),
            [None, None, None, None],
        )

    def test_null_in_window_is_null(self) -> None:
        """
        Verifies that an interior ``null`` nulls every window that overlaps it, then the output recovers.
        """
        values = [10.0, 11.0, 12.0, None, 14.0, 15.0, 16.0, 17.0]
        assert_matches(
            apply_expr(values, chande_momentum_oscillator(pl.col(COLUMN_X), 3)),
            chande_momentum_oscillator_reference(values, 3),
        )

    def test_nan_propagates(self) -> None:
        """
        Verifies that a NaN propagates (matching the naive reference).
        """
        values = [10.0, 11.0, 12.0, 13.0, 14.0, math.nan, 16.0, 17.0]
        assert_matches(
            apply_expr(values, chande_momentum_oscillator(pl.col(COLUMN_X), 3)),
            chande_momentum_oscillator_reference(values, 3),
        )

    def test_warmup_null_count(self) -> None:
        """
        Verifies the warm-up is ``window`` rows (row 0 has no change; the rolling sums need ``window`` changes).
        """
        result = apply_expr([10.0, 11.0, 12.0, 11.0, 13.0], chande_momentum_oscillator(pl.col(COLUMN_X), 3))
        assert result[:3] == [None, None, None]
        assert result[3] is not None

    def test_window_exceeds_length(self) -> None:
        """
        Verifies that when ``window`` exceeds the series length the whole output is null (no full window of changes).
        """
        assert_matches(apply_expr([1.0, 2.0, 3.0], chande_momentum_oscillator(pl.col(COLUMN_X), 5)), [None, None, None])

    def test_flat_window_is_nan(self) -> None:
        """
        Verifies that a window with no movement (all changes zero) yields ``NaN`` (the ``0 / 0`` total).
        """
        result = apply_expr([10.0, 10.0, 10.0, 10.0, 10.0], chande_momentum_oscillator(pl.col(COLUMN_X), 3))
        assert_matches(result, [None, None, None, math.nan, math.nan])

    def test_flat_tail_after_movement_is_nan(self) -> None:
        """
        Verifies that a window which goes flat *after* large moves yields ``NaN`` (the ``0 / 0`` total), the case the
        all-flat-from-row-0 series above cannot reach: here the rolling sums must subtract previously-added large
        changes out of the window, leaving a sub-ULP residual that without the explicit flat-window guard would fake a
        saturated ``+/-100`` reading instead of the documented ``NaN``.
        """
        values = [
            30426.583515139646,
            30426.583514906622,
            30426.583514906622,
            30426.58351574153,
            126995.79017007923,
            126995.79017011753,
            112548.45267126478,
            112548.45267126478,
            112548.4526722116,
            -512653.3416246533,
            -512653.3416243748,
            -1000000.0,
            *([-1000000.0] * 12),
        ]
        result = apply_expr(values, chande_momentum_oscillator(pl.col(COLUMN_X), 10))
        # The final three windows hold only the flat ``-1000000.0`` tail, so every change in them is exactly zero.
        assert_matches(result[-3:], [math.nan, math.nan, math.nan])
        assert_matches(result, chande_momentum_oscillator_reference(values, 10))

    def test_near_flat_tail_after_movement_stays_bounded(self) -> None:
        """
        Verifies the conditioning-limit contract: after three large bars the series settles to changes so small
        (last-ULP) that the streaming rolling sums are residual-dominated, not exactly zero. The quotient there
        can no longer be trusted (without the clip it read ``-1100``, ``+200`` and ``-inf`` on this input), but the clip
        keeps every value inside ``[-100, +100]`` and the residual-free flat guard does not over-fire: past a sane
        dynamic range the value degrades yet stays finite and in range (see ``CORRECTNESS.md``), so the bound is
        asserted rather than agreement with the oracle.
        """
        values = [
            -393234.29432880785,
            -120581.32877667283,
            -981164.4022553843,
            158476.49063369818,
            158476.49063369833,
            158476.4906336983,
            158476.49063369824,
            158476.49063369838,
            158476.49063369824,
            158476.49063369833,
            158476.49063369833,
            158476.4906336983,
            158476.49063369835,
            158476.4906336982,
        ]
        result = apply_expr(values, chande_momentum_oscillator(pl.col(COLUMN_X), 4))
        assert all(value is not None and not math.isnan(value) and -100.0 <= value <= 100.0 for value in result[7:])

    def test_saturates_at_plus_minus_100(self) -> None:
        """
        Verifies that an all-up window reads ``+100`` and an all-down window reads ``-100``.
        """
        assert_matches(
            apply_expr([10.0, 11.0, 12.0, 13.0, 14.0], chande_momentum_oscillator(pl.col(COLUMN_X), 3)),
            [None, None, None, 100.0, 100.0],
        )
        assert_matches(
            apply_expr([14.0, 13.0, 12.0, 11.0, 10.0], chande_momentum_oscillator(pl.col(COLUMN_X), 3)),
            [None, None, None, -100.0, -100.0],
        )


class TestChandeMomentumOscillatorCorrectness:
    """
    Against the naive reference oracle and frozen golden-master values.
    """

    def test_matches_reference(self) -> None:
        """
        Verifies agreement with the naive closed-form reference across several windows.
        """
        values = [10.0, 11.0, 12.0, 11.0, 13.0, 14.0, 13.0, 15.0, 14.0, 16.0, 15.0, 17.0]
        for window in (1, 2, 3, 5):
            assert_matches(
                apply_expr(values, chande_momentum_oscillator(pl.col(COLUMN_X), window)),
                chande_momentum_oscillator_reference(values, window),
            )

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference: chande_momentum_oscillator(window=3) over the sample series.
        """
        result = apply_expr(
            [10.0, 11.0, 12.0, 11.0, 13.0, 14.0, 13.0, 15.0], chande_momentum_oscillator(pl.col(COLUMN_X), 3).round(4)
        )
        assert_matches(result, [None, None, None, 33.3333, 50.0, 50.0, 50.0, 50.0])


class TestChandeMomentumOscillatorProperties:
    """
    Invariants that must hold for all inputs (property-based).
    """

    @given(case=_cases(subnormal_safe_floats(bound=1e6)))
    def test_matches_reference_for_any_input(
        self,
        case: tuple[list[float], int],
    ) -> None:
        """
        Verifies that, for any series and window, the implementation matches the naive reference.
        """
        values, window = case
        assert_matches(
            apply_expr(values, chande_momentum_oscillator(pl.col(COLUMN_X), window)),
            chande_momentum_oscillator_reference(values, window),
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=ABSOLUTE_TOLERANCE_PROPERTY,
        )

    @given(case=_cases(missing_data_floats()))
    def test_matches_reference_under_missing_data(
        self,
        case: tuple[list[float | None], int],
    ) -> None:
        """
        Verifies that, for inputs freely mixing null / NaN / finite, the implementation matches the naive reference.
        """
        values, window = case
        assert_matches(
            apply_expr(values, chande_momentum_oscillator(pl.col(COLUMN_X), window)),
            chande_momentum_oscillator_reference(values, window),
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=ABSOLUTE_TOLERANCE_PROPERTY,
        )

    @given(
        case=_cases(subnormal_safe_floats(bound=1e6)),
        exponent=st.sampled_from([-4, -3, -2, -1, 1, 2, 3, 4]),
    )
    def test_scale_invariance(
        self,
        case: tuple[list[float], int],
        exponent: int,
    ) -> None:
        """
        Verifies that ``chande_momentum_oscillator`` is scale-invariant: scaling every input value by a constant
        ``k`` leaves the output unchanged -- ``chande_momentum_oscillator(k * x) == chande_momentum_oscillator(x)``.
        ``k`` is a power of two, so the rescale is exact and adds no floating-point error.
        """
        k = 2.0**exponent
        values, window = case
        result_base = apply_expr(values, chande_momentum_oscillator(pl.col(COLUMN_X), window))
        result_scaled = apply_expr(
            [value * k for value in values], chande_momentum_oscillator(pl.col(COLUMN_X), window)
        )
        assert_scale_homogeneous(result_scaled, result_base, k=k, degree=0)

    @given(case=_cases(subnormal_safe_floats(bound=1e6)))
    def test_bounded(
        self,
        case: tuple[list[float], int],
    ) -> None:
        """
        Verifies that every defined value lies within ``[-100, 100]``.
        """
        values, window = case
        for value in apply_expr(values, chande_momentum_oscillator(pl.col(COLUMN_X), window)):
            if value is not None and not math.isnan(value):
                assert -100.0 - BOUND_MARGIN <= value <= 100.0 + BOUND_MARGIN
