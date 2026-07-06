"""
Tests for ``pomata.indicators.trend_mode`` — Ehlers' Hilbert-transform trend vs cycle mode flag.

``trend_mode`` is a parameter-free, sequential ``map_batches`` kernel reading the shared Ehlers pipeline: it combines
the sine-wave crossings, the dominant-cycle phase rate, and the smoothed price's deviation from the instantaneous
trendline into a ``1.0`` (trending) / ``0.0`` (cycling) flag. The local ``apply_trend_mode`` helper materializes it over
a one-column ``Float64`` frame; ``assert_matches`` and the naive ``trend_mode_reference`` oracle (an independent
transcription of the same pipeline) are shared across the suite. The flag is scale-invariant — the mode does not depend
on the price scale — and every emitted value is exactly ``0.0`` or ``1.0``.

The ladder is the canonical one: contract, edge (warm-up / null / NaN latch), correctness (vs the closed-form reference
and a frozen golden master), and properties (reference agreement incl. missing data, scale-invariance, large-magnitude
stability, flag membership). Categories are split into classes; cross-cutting categories use markers (see
``tests/README.md``).
"""

import math
from collections.abc import Sequence

import polars as pl
from hypothesis import given
from hypothesis import strategies as st
from tests.indicators.oracles import trend_mode_reference
from tests.support import (
    ABSOLUTE_TOLERANCE_EXACT,
    ABSOLUTE_TOLERANCE_REFERENCE,
    COLUMN_X,
    GROUP_KEY,
    RELATIVE_TOLERANCE_REFERENCE,
    apply_expr,
    assert_matches,
    assert_scale_homogeneous,
    count_leading_nulls,
    spans_even_lag_repeat,
    two_segment_missing_data,
)

from pomata.indicators import trend_mode

# A deterministic 80-bar carrier (a clean 20-bar cycle): long enough to clear the 63-bar warm-up and emit seventeen
# flags (a mix of trend / cycle).
_SAMPLE = [100.0 + 10.0 * math.sin(2 * math.pi * index / 20) for index in range(80)]
_WARMUP = 63

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- derived, not chosen; the rationale is the shared method in CORRECTNESS.md, the numbers are this
# indicator's. To add an indicator, set its three facts here; the property tier below is then the same shape as every
# other indicator's.
#   1. warmup  W = 63   (the smoothers' settling plus the dominant-cycle look-back; parameter-free, so W is the constant
#              ``_WARMUP`` above)
#   2. memory  the oracle shares pomata's seeding, so the property holds from the first defined row (M = 0); each
#              example carries D in [1, SERIES_MAX] defined rows on top of W, so there is always output to check --
#              never an all-warm-up series
#   3. domain  finite values over the test's regime (any-input / scale / missing-data / large-magnitude), widened per
#              test below; the agreement tiers exclude any even-lag repeat (flat run / period-two alternation), and the
#              missing-data tier draws a finite prefix past the warm-up then a missing tail (two_segment_missing_data)
# The flag is scale-INVARIANT: a 0/1 value whatever the input magnitude, so its tolerance is ABSOLUTE
# (ABSOLUTE_TOLERANCE_REFERENCE), never input_scale-sized. The scale test stays a power-of-two, bit-exact comparison.
# Repetitions N are the shared CI profile (tests/conftest.py); override per-test only if its parameter space is larger.
# ----------------------------------------------------------------------------------------------------------------------
SERIES_MAX = 17


@st.composite
def _cases[T](draw: st.DrawFn, values: st.SearchStrategy[T]) -> list[T]:
    """
    A series sized from the facts above: the indicator is parameter-free, so -- unlike the windowed indicators'
    ``(series, window)`` pair -- a case is just the series, floored at ``_WARMUP + 1`` rows so there is always at least
    one defined output (never an all-warm-up series).
    """
    defined = draw(st.integers(min_value=1, max_value=SERIES_MAX))
    length = _WARMUP + defined
    return draw(st.lists(values, min_size=length, max_size=length))


def apply_trend_mode(values: Sequence[float | None]) -> list[float | None]:
    """
    Materialize ``trend_mode`` over a one-column ``Float64`` frame built from ``values``.
    """
    return apply_expr(values, trend_mode(pl.col(COLUMN_X)))


class TestTrendModeContract:
    """
    Type, shape, and lazy/eager guarantees.
    """

    def test_over_partitions_independently(self) -> None:
        """
        Verifies that under ``.over`` the recurrence resets per group and never spans group boundaries.
        """
        group_b_input = [50.0 + 5.0 * math.sin(2 * math.pi * index / 20) for index in range(80)]
        frame = pl.DataFrame({GROUP_KEY: ["a"] * 80 + ["b"] * 80, COLUMN_X: _SAMPLE + group_b_input})
        result = frame.select(trend_mode(pl.col(COLUMN_X)).over(GROUP_KEY).alias("y"))["y"].to_list()
        expected = apply_trend_mode(_SAMPLE) + apply_trend_mode(group_b_input)
        assert_matches(result, expected)


class TestTrendModeEdge:
    """
    Warm-up and null / NaN latching.
    """

    def test_all_null(self) -> None:
        """
        Verifies that an all-null series yields an all-null output.
        """
        assert_matches(apply_trend_mode([None, None, None, None, None]), [None, None, None, None, None])

    def test_warmup_null_count(self) -> None:
        """
        Verifies that the leading-null run is exactly ``63`` and every later row is defined.
        """
        result = apply_trend_mode(_SAMPLE)
        leading_nulls = count_leading_nulls(result)
        assert leading_nulls == _WARMUP
        assert all(value is not None for value in result[_WARMUP:])

    def test_null_latches(self) -> None:
        """
        Verifies that a null latches null for every row from there (matching the naive reference).
        """
        values: list[float | None] = [*_SAMPLE[:70], None, *_SAMPLE[71:]]
        result = apply_trend_mode(values)
        assert result[69] is not None
        assert all(value is None for value in result[70:])
        assert_matches(result, trend_mode_reference(values))

    def test_nan_latches(self) -> None:
        """
        Verifies that a NaN latches null for every row from there (matching the naive reference).
        """
        values: list[float | None] = [*_SAMPLE[:70], math.nan, *_SAMPLE[71:]]
        result = apply_trend_mode(values)
        assert all(value is None for value in result[70:])
        assert_matches(result, trend_mode_reference(values))


class TestTrendModeCorrectness:
    """
    Against the reference oracle (internal-consistency for this recurrence) and frozen golden-master values.
    """

    def test_matches_reference(self) -> None:
        """
        Verifies agreement with the naive reference on the sample carrier.
        """
        assert_matches(
            apply_trend_mode(_SAMPLE),
            trend_mode_reference(_SAMPLE),
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
            abs_tol=ABSOLUTE_TOLERANCE_EXACT,
        )

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference: the seventeen emitted flags over the sample carrier.
        """
        result = apply_trend_mode(_SAMPLE)
        assert_matches(
            [None if value is None else round(value, 4) for value in result[_WARMUP:]],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        )


class TestTrendModeProperties:
    """
    Invariants that must hold for all inputs (property-based).
    """

    @given(
        case=_cases(st.floats(min_value=1.0, max_value=1e3, allow_nan=False, allow_infinity=False)).filter(
            lambda series: not spans_even_lag_repeat(series)
        )
    )
    def test_matches_reference_for_any_input(self, case: list[float]) -> None:
        """
        Verifies that, for any series with no even-lag repeat, the implementation matches the naive reference.

        The flag is read off the sine / lead-sine crossings of this shared pipeline, so it inherits the same flat-run
        degeneracy the sibling lines have: Ehlers' six-tap quadrature filter reads the four-bar smooth at even lags, so
        the in-phase component collapses to a pure cancellation residual whenever the smooth repeats two bars apart — a
        flat run, but also a period-two alternation, both of which leave ``x[i] == x[i - 2]``. There the explicit FIR
        and the oracle's compensated ``sum()`` round to opposite sides of the ``imag != 0`` branch, and the flipped
        phase propagates through the persistent ``days_in_trend`` counter into a different emitted ``0.0`` / ``1.0``
        flag. Neither side is wrong — it is a branch artifact on a mathematically-undefined-phase flat run, not a
        correctness gap — so the filter excludes any even-lag repeat (see :func:`spans_even_lag_repeat`), matching the
        sine_wave / mama tiers on this pipeline.
        """
        values = case
        assert_matches(
            apply_trend_mode(values),
            trend_mode_reference(values),
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
            abs_tol=ABSOLUTE_TOLERANCE_EXACT,
        )

    @given(
        case=_cases(st.floats(min_value=1.0, max_value=1e3, allow_nan=False, allow_infinity=False)),
        exponent=st.sampled_from([-4, -3, -2, -1, 1, 2, 3, 4]),
    )
    def test_scale_invariance(self, case: list[float], exponent: int) -> None:
        """
        Verifies that ``trend_mode`` is scale-invariant: scaling every input value by a constant ``k`` leaves the
        output unchanged -- ``trend_mode(k * x) == trend_mode(x)``. ``k`` is a power of two, so the rescale is exact
        and adds no floating-point error.
        """
        factor = 2.0**exponent
        values = case
        base = apply_trend_mode(values)
        scaled = apply_trend_mode([value * factor for value in values])
        assert_scale_homogeneous(scaled, base, k=factor, degree=0)

    @given(
        case=_cases(st.floats(min_value=1.0, max_value=1e3, allow_nan=False, allow_infinity=False)).filter(
            lambda series: not spans_even_lag_repeat(series)
        ),
        scale=st.sampled_from([2.0**-30, 2.0**30, 2.0**40]),
    )
    def test_matches_reference_at_large_magnitude(self, case: list[float], scale: float) -> None:
        """
        Verifies that at extreme magnitudes the implementation stays finite where the reference is and agrees.

        The carrier is required to have no even-lag repeat: a value equal to the value two bars earlier (a flat run, or
        a period-two alternation) drives the in-phase component to a pure cancellation residual that flips the
        ``imag != 0`` branch and so the emitted flag — a branch artifact on a mathematically-undefined-phase flat run,
        not a magnitude effect (see :func:`spans_even_lag_repeat`).
        """
        scaled = [value * scale for value in case]
        assert_matches(
            apply_trend_mode(scaled),
            trend_mode_reference(scaled),
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    @given(case=two_segment_missing_data(_WARMUP))
    def test_matches_reference_under_missing_data(self, case: list[float | None]) -> None:
        """
        Verifies that, for a finite prefix clearing the warm-up followed by a null / NaN / finite tail, the
        implementation matches the naive reference.

        Drawn as two segments (finite prefix longer than the warm-up, then a missing-data tail) so a defined flag is
        emitted and then meets ``null`` / ``NaN`` — a single missing value anywhere in the warm-up latches the whole
        output to ``null``, so an all-mixed draw would almost always compare all-``null`` against all-``null`` and check
        nothing numeric.
        """
        values = case
        assert_matches(
            apply_trend_mode(values),
            trend_mode_reference(values),
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
            abs_tol=ABSOLUTE_TOLERANCE_EXACT,
        )

    @given(case=_cases(st.floats(min_value=1.0, max_value=1e3, allow_nan=False, allow_infinity=False)))
    def test_flag_membership(self, case: list[float]) -> None:
        """
        Verifies that every emitted value is exactly ``0.0`` or ``1.0`` (the flag is a two-valued membership, not a
        range).
        """
        for value in apply_trend_mode(case):
            if value is not None:
                assert value in (0.0, 1.0)
