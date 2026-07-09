"""
Tests for ``pomata.indicators.dominant_cycle_period`` — Ehlers' Hilbert-transform dominant-cycle period.

``dominant_cycle_period`` is a parameter-free, sequential ``map_batches`` kernel: a four-bar smooth, a Hilbert-transform
detrend into in-phase / quadrature components, and a homodyne discriminator yield the instantaneous cycle length, which
is clamped to ``[6, 50]`` bars and smoothed. The local ``apply_dominant_cycle_period`` helper materializes it over a
one-column ``Float64`` frame; ``assert_matches`` and the naive ``dominant_cycle_period_reference`` oracle (a
transcription of the same pipeline) are shared across the suite. The period is scale-invariant — a cycle
length does not depend on the price scale.

The ladder is the canonical one: contract, edge (warm-up / null / NaN latch), correctness (vs the closed-form reference
and a frozen golden master), and properties (reference agreement incl. missing data, scale-invariance, large-magnitude
stability). Categories are split into classes; cross-cutting categories use markers (see ``tests/README.md``).
"""

import math
from collections.abc import Sequence

import polars as pl
from hypothesis import given
from hypothesis import strategies as st
from tests.indicators.oracles import dominant_cycle_period_reference
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
    two_segment_missing_data,
)

from pomata.indicators import dominant_cycle_period

# A deterministic 40-bar carrier (a clean 20-bar cycle): long enough to clear the 32-bar warm-up and emit eight values.
_SAMPLE = [100.0 + 10.0 * math.sin(2 * math.pi * index / 20) for index in range(40)]
_WARMUP = 32

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- derived, not chosen; the rationale is the shared method in CORRECTNESS.md, the numbers are this
# indicator's. To add an indicator, set its three facts here; the property tier below is then the same shape as every
# other indicator's.
#   1. warmup  W = 32   (the leading-null run the recursive smoothers need to settle; parameter-free, so W is the
#              constant ``_WARMUP`` above)
#   2. memory  the oracle shares pomata's seeding, so the property holds from the first defined row (M = 0); each
#              example carries D in [1, SERIES_MAX] defined rows on top of W, so there is always output to check --
#              never an all-warm-up series
#   3. domain  finite values over the test's regime (any-input / scale / missing-data / large-magnitude), widened per
#              test below; the missing-data tier draws a finite prefix past the warm-up then a missing tail
#              (two_segment_missing_data), the prefix free of even-lag repeats to keep the defined region conditioned
# The period is scale-INVARIANT: an O(1) cycle length whatever the input magnitude, so its tolerance is ABSOLUTE
# (ABSOLUTE_TOLERANCE_REFERENCE), never input_scale-sized. The scale test stays a power-of-two, bit-exact comparison.
# Repetitions N are the shared CI profile (tests/conftest.py); override per-test only if its parameter space is larger.
# ----------------------------------------------------------------------------------------------------------------------
SERIES_MAX = 48


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


def apply_dominant_cycle_period(values: Sequence[float | None]) -> list[float | None]:
    """
    Materialize ``dominant_cycle_period`` over a one-column ``Float64`` frame built from ``values``.
    """
    return apply_expr(values, dominant_cycle_period(pl.col(COLUMN_X)))


class TestDominantCyclePeriodContract:
    """
    Type, shape, and lazy/eager guarantees.
    """

    def test_over_partitions_independently(self) -> None:
        """
        Verifies that under ``.over`` the recurrence resets per group and never spans group boundaries.
        """
        group_b_input = [50.0 + 5.0 * math.sin(2 * math.pi * index / 20) for index in range(40)]
        frame = pl.DataFrame({GROUP_KEY: ["a"] * 40 + ["b"] * 40, COLUMN_X: _SAMPLE + group_b_input})
        result = frame.select(dominant_cycle_period(pl.col(COLUMN_X)).over(GROUP_KEY).alias("y"))["y"].to_list()
        expected = apply_dominant_cycle_period(_SAMPLE) + apply_dominant_cycle_period(group_b_input)
        assert_matches(result, expected)


class TestDominantCyclePeriodEdge:
    """
    Warm-up and null / NaN latching.
    """

    def test_single_row(self) -> None:
        """
        Verifies that a one-row series is all warm-up (one null): far inside the 32-bar warm-up.
        """
        assert_matches(apply_dominant_cycle_period([100.0]), [None])

    def test_all_null(self) -> None:
        """
        Verifies that an all-null series yields an all-null output.
        """
        assert_matches(apply_dominant_cycle_period([None, None, None, None, None]), [None, None, None, None, None])

    def test_null_latches(self) -> None:
        """
        Verifies that a null latches null for every row from there (matching the naive reference).
        """
        values: list[float | None] = [*_SAMPLE[:35], None, *_SAMPLE[36:]]
        result = apply_dominant_cycle_period(values)
        assert result[34] is not None
        assert all(value is None for value in result[35:])
        assert_matches(result, dominant_cycle_period_reference(values))

    def test_nan_latches(self) -> None:
        """
        Verifies that a NaN latches null for every row from there (matching the naive reference).
        """
        values: list[float | None] = [*_SAMPLE[:35], math.nan, *_SAMPLE[36:]]
        result = apply_dominant_cycle_period(values)
        assert all(value is None for value in result[35:])
        assert_matches(result, dominant_cycle_period_reference(values))

    def test_warmup_null_count(self) -> None:
        """
        Verifies that the leading-null run is exactly ``32`` and every later row is defined.
        """
        result = apply_dominant_cycle_period(_SAMPLE)
        leading_nulls = count_leading_nulls(result)
        assert leading_nulls == _WARMUP
        assert all(value is not None for value in result[_WARMUP:])


class TestDominantCyclePeriodCorrectness:
    """
    Against the reference oracle (internal-consistency for this recurrence) and frozen golden-master values.
    """

    def test_matches_reference(self) -> None:
        """
        Verifies agreement with the naive reference on the sample carrier.
        """
        assert_matches(
            apply_dominant_cycle_period(_SAMPLE),
            dominant_cycle_period_reference(_SAMPLE),
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
            abs_tol=ABSOLUTE_TOLERANCE_EXACT,
        )

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference: the eight emitted periods over the sample carrier.
        """
        result = apply_dominant_cycle_period(_SAMPLE)
        assert_matches(
            [None if value is None else round(value, 4) for value in result[_WARMUP:]],
            [19.0186, 19.3994, 19.7391, 20.051, 20.3271, 20.5471, 20.6936, 20.7611],
        )


class TestDominantCyclePeriodProperties:
    """
    Invariants that must hold for all inputs (property-based).
    """

    @given(case=_cases(st.floats(min_value=1.0, max_value=1e3, allow_nan=False, allow_infinity=False)))
    def test_matches_reference_for_any_input(self, case: list[float]) -> None:
        """
        Verifies that, for any series, the implementation matches the naive reference.
        """
        values = case
        assert_matches(
            apply_dominant_cycle_period(values),
            dominant_cycle_period_reference(values),
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
            abs_tol=ABSOLUTE_TOLERANCE_EXACT,
        )

    @given(case=two_segment_missing_data(_WARMUP))
    def test_matches_reference_under_missing_data(self, case: list[float | None]) -> None:
        """
        Verifies that, for a finite prefix clearing the warm-up followed by a null / NaN / finite tail, the
        implementation matches the naive reference.

        Drawn as two segments (finite prefix longer than the warm-up, then a missing-data tail) so a defined period is
        emitted and then meets ``null`` / ``NaN`` — a single missing value anywhere in the warm-up latches the whole
        output to ``null``, so an all-mixed draw would almost always compare all-``null`` against all-``null``. The
        prefix has no even-lag repeat (the same flat-run guard the sibling agreement tiers use), keeping the defined
        region well-conditioned: on a flat run the implementation and oracle differ by ~3.6e-13, which would brush the
        default 1e-12 tolerance.
        """
        values = case
        assert_matches(
            apply_dominant_cycle_period(values),
            dominant_cycle_period_reference(values),
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
            abs_tol=ABSOLUTE_TOLERANCE_EXACT,
        )

    @given(
        case=_cases(st.floats(min_value=1.0, max_value=1e3, allow_nan=False, allow_infinity=False)),
        exponent=st.sampled_from([-4, -3, -2, -1, 1, 2, 3, 4]),
    )
    def test_scale_invariance(self, case: list[float], exponent: int) -> None:
        """
        Verifies that ``dominant_cycle_period`` is scale-invariant: scaling every input value by a constant ``k``
        leaves the output unchanged -- ``dominant_cycle_period(k * x) == dominant_cycle_period(x)``. ``k`` is a
        power of two, so the rescale is exact and adds no floating-point error.
        """
        factor = 2.0**exponent
        values = case
        base = apply_dominant_cycle_period(values)
        scaled = apply_dominant_cycle_period([value * factor for value in values])
        assert_scale_homogeneous(scaled, base, k=factor, degree=0)

    @given(
        case=_cases(st.floats(min_value=1.0, max_value=1e3, allow_nan=False, allow_infinity=False)),
        scale=st.sampled_from([2.0**-30, 2.0**30, 2.0**40]),
    )
    def test_matches_reference_at_large_magnitude(self, case: list[float], scale: float) -> None:
        """
        Verifies that at extreme magnitudes the implementation stays finite where the reference is and agrees.
        """
        scaled = [value * scale for value in case]
        assert_matches(
            apply_dominant_cycle_period(scaled),
            dominant_cycle_period_reference(scaled),
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )
