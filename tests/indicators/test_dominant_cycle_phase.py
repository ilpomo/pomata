"""
Tests for ``pomata.indicators.dominant_cycle_phase`` — Ehlers' Hilbert-transform dominant-cycle phase.

``dominant_cycle_phase`` is a parameter-free, sequential ``map_batches`` kernel: a four-bar smooth, a Hilbert-transform
detrend into in-phase / quadrature components, and a homodyne discriminator yield the dominant-cycle length, off which a
running discrete transform of the smoothed price reads the instantaneous phase, in degrees, then lag-compensates it. The
local ``apply_dominant_cycle_phase`` helper materializes it over a one-column ``Float64`` frame; ``assert_matches`` and
the naive ``dominant_cycle_phase_reference`` oracle (a transcription of the same pipeline) are shared
across the suite. The phase is scale-invariant — a phase does not depend on the price scale.

The ladder is the canonical one: contract, edge (warm-up / null / NaN latch), correctness (vs the closed-form reference
and a frozen golden master), and properties (reference agreement incl. missing data, scale-invariance, large-magnitude
stability). Categories are split into classes; cross-cutting categories use markers (see ``tests/README.md``).
"""

import math
from collections.abc import Sequence

import polars as pl
from hypothesis import given
from hypothesis import strategies as st
from polars.testing import assert_frame_equal
from tests.indicators.oracles import dominant_cycle_phase_reference
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

from pomata.indicators import dominant_cycle_phase

# A deterministic 80-bar carrier (a clean 20-bar cycle): long enough to clear the 63-bar warm-up and emit seventeen
# values.
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
#              test below
# The phase is scale-INVARIANT: an O(1) value (degrees) whatever the input magnitude, so its tolerance is ABSOLUTE
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


def apply_dominant_cycle_phase(values: Sequence[float | None]) -> list[float | None]:
    """
    Materialize ``dominant_cycle_phase`` over a one-column ``Float64`` frame built from ``values``.
    """
    return apply_expr(values, dominant_cycle_phase(pl.col(COLUMN_X)))


class TestDominantCyclePhaseContract:
    """
    Type, shape, and lazy/eager guarantees.
    """

    def test_returns_expr(self) -> None:
        """
        Verifies that the factory returns a ``pl.Expr`` without touching a frame.
        """
        assert isinstance(dominant_cycle_phase(pl.col(COLUMN_X)), pl.Expr)

    def test_preserves_length_and_dtype(self) -> None:
        """
        Verifies that the output has one value per input row and is ``Float64``.
        """
        result = pl.DataFrame({COLUMN_X: pl.Series(COLUMN_X, _SAMPLE)}).select(
            dominant_cycle_phase(pl.col(COLUMN_X)).alias("y")
        )
        assert result.height == len(_SAMPLE)
        assert result.schema["y"] == pl.Float64

    def test_lazy_eager_parity(self) -> None:
        """
        Verifies that eager and lazy application produce identical materialized output.
        """
        frame = pl.DataFrame({COLUMN_X: pl.Series(COLUMN_X, _SAMPLE)})
        expr = dominant_cycle_phase(pl.col(COLUMN_X)).alias("y")
        assert_frame_equal(frame.select(expr), frame.lazy().select(expr).collect())

    def test_over_partitions_independently(self) -> None:
        """
        Verifies that under ``.over`` the recurrence resets per group and never spans group boundaries.
        """
        group_b_input = [50.0 + 5.0 * math.sin(2 * math.pi * index / 20) for index in range(80)]
        frame = pl.DataFrame({GROUP_KEY: ["a"] * 80 + ["b"] * 80, COLUMN_X: _SAMPLE + group_b_input})
        result = frame.select(dominant_cycle_phase(pl.col(COLUMN_X)).over(GROUP_KEY).alias("y"))["y"].to_list()
        expected = apply_dominant_cycle_phase(_SAMPLE) + apply_dominant_cycle_phase(group_b_input)
        assert_matches(result, expected)


class TestDominantCyclePhaseEdge:
    """
    Warm-up and null / NaN latching.
    """

    def test_empty(self) -> None:
        """
        Verifies that an empty input yields an empty output.
        """
        assert_matches(apply_dominant_cycle_phase([]), [])

    def test_all_null(self) -> None:
        """
        Verifies that an all-null series yields an all-null output.
        """
        assert_matches(apply_dominant_cycle_phase([None, None, None, None, None]), [None, None, None, None, None])

    def test_warmup_null_count(self) -> None:
        """
        Verifies that the leading-null run is exactly ``63`` and every later row is defined.
        """
        result = apply_dominant_cycle_phase(_SAMPLE)
        leading_nulls = count_leading_nulls(result)
        assert leading_nulls == _WARMUP
        assert all(value is not None for value in result[_WARMUP:])

    def test_null_latches(self) -> None:
        """
        Verifies that a null latches null for every row from there (matching the naive reference).
        """
        values: list[float | None] = [*_SAMPLE[:66], None, *_SAMPLE[67:]]
        result = apply_dominant_cycle_phase(values)
        assert result[65] is not None
        assert all(value is None for value in result[66:])
        assert_matches(result, dominant_cycle_phase_reference(values))

    def test_nan_latches(self) -> None:
        """
        Verifies that a NaN latches null for every row from there (matching the naive reference).
        """
        values: list[float | None] = [*_SAMPLE[:66], math.nan, *_SAMPLE[67:]]
        result = apply_dominant_cycle_phase(values)
        assert all(value is None for value in result[66:])
        assert_matches(result, dominant_cycle_phase_reference(values))


class TestDominantCyclePhaseCorrectness:
    """
    Against the reference oracle (internal-consistency for this recurrence) and frozen golden-master values.
    """

    def test_matches_reference(self) -> None:
        """
        Verifies agreement with the naive reference on the sample carrier.
        """
        assert_matches(
            apply_dominant_cycle_phase(_SAMPLE),
            dominant_cycle_phase_reference(_SAMPLE),
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
            abs_tol=ABSOLUTE_TOLERANCE_EXACT,
        )

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference: the seventeen emitted phases over the sample carrier.
        """
        result = apply_dominant_cycle_phase(_SAMPLE)
        assert_matches(
            [None if value is None else round(value, 4) for value in result[_WARMUP:]],
            [
                54.1853,
                72.1855,
                90.1782,
                108.1678,
                126.1594,
                144.1573,
                162.1633,
                180.1763,
                198.1917,
                216.204,
                234.2083,
                252.2035,
                270.1915,
                288.177,
                306.1651,
                -35.84,
                -17.8363,
            ],
        )


class TestDominantCyclePhaseProperties:
    """
    Invariants that must hold for all inputs (property-based).
    """

    # NOTE: the carrier must have no even-lag repeat (x[i] == x[i-2]) -- on a flat run or a period-two alternation the
    # in-phase / quadrature vector is a cancellation residual, the phase is mathematically undefined (atan2 of a ~zero
    # vector), and the implementation and oracle resolve the branch ~180 degrees apart. The same exclusion the mama /
    # sine_wave tests apply on this shared pipeline (see spans_even_lag_repeat).
    @given(
        case=_cases(st.floats(min_value=1.0, max_value=1e3, allow_nan=False, allow_infinity=False)).filter(
            lambda series: not spans_even_lag_repeat(series)
        ),
    )
    def test_matches_reference_for_any_input(self, case: list[float]) -> None:
        """
        Verifies that, for any series with no even-lag repeat, the implementation matches the naive reference.
        """
        values = case
        assert_matches(
            apply_dominant_cycle_phase(values),
            dominant_cycle_phase_reference(values),
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
            abs_tol=ABSOLUTE_TOLERANCE_EXACT,
        )

    @given(
        case=_cases(st.floats(min_value=1.0, max_value=1e3, allow_nan=False, allow_infinity=False)),
        exponent=st.sampled_from([-4, -3, -2, -1, 1, 2, 3, 4]),
    )
    def test_scale_invariance(self, case: list[float], exponent: int) -> None:
        """
        Verifies that the phase is scale-invariant: ``dominant_cycle_phase(k * x) == dominant_cycle_phase(x)``.

        The factor is a power of two, so the rescaling is lossless and the in-phase / quadrature ratio that fixes the
        phase is unchanged to the bit, not merely within a tolerance.
        """
        factor = 2.0**exponent
        values = case
        base = apply_dominant_cycle_phase(values)
        scaled = apply_dominant_cycle_phase([value * factor for value in values])
        assert_scale_homogeneous(scaled, base, k=factor, degree=0)

    # NOTE: no even-lag repeat, as in test_matches_reference_for_any_input -- a flat run or period-two alternation makes
    # the phase undefined and flips the atan2 branch ~180 degrees between implementation and oracle.
    @given(
        case=_cases(st.floats(min_value=1.0, max_value=1e3, allow_nan=False, allow_infinity=False)).filter(
            lambda series: not spans_even_lag_repeat(series)
        ),
        scale=st.sampled_from([2.0**-30, 2.0**30, 2.0**40]),
    )
    def test_matches_reference_at_large_magnitude(self, case: list[float], scale: float) -> None:
        """
        Verifies that at extreme magnitudes the implementation stays finite where the reference is and agrees.
        """
        values = case
        scaled = [value * scale for value in values]
        assert_matches(
            apply_dominant_cycle_phase(scaled),
            dominant_cycle_phase_reference(scaled),
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    @given(case=two_segment_missing_data(_WARMUP))
    def test_matches_reference_under_missing_data(self, case: list[float | None]) -> None:
        """
        Verifies that, for a finite prefix clearing the warm-up followed by a null / NaN / finite tail, the
        implementation matches the naive reference.

        Drawn as two segments (finite prefix longer than the warm-up, then a missing-data tail) so a defined phase is
        emitted and then meets ``null`` / ``NaN`` — a single missing value anywhere in the warm-up latches the whole
        output to ``null``, so an all-mixed draw would almost always compare all-``null`` against all-``null``. The
        prefix has no even-lag repeat (the same flat-run guard the agreement tiers use), so the defined region is
        well-conditioned and the phase is not compared on its mathematically-undefined flat-run branch.
        """
        values = case
        assert_matches(
            apply_dominant_cycle_phase(values),
            dominant_cycle_phase_reference(values),
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
            abs_tol=ABSOLUTE_TOLERANCE_EXACT,
        )
