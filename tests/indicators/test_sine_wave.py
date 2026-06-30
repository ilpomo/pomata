"""
Tests for ``pomata.indicators.sine_wave`` — Ehlers' Hilbert-transform sine-wave indicator.

``sine_wave`` is the parameter-free, sequential ``map_batches`` kernel shared by the cycle family: a four-bar smooth, a
Hilbert-transform detrend into in-phase / quadrature components, a homodyne discriminator for the dominant-cycle period,
and a running discrete transform for its phase, of which it emits the sine and a lead sine advanced by ``45°`` as a
struct with the fields ``sine`` / ``lead_sine``. The local ``apply_sine_wave`` helper materializes each field over a
one-column ``Float64`` frame into a dict of lists, so the shared ``assert_matches`` and the naive
``sine_wave_reference`` oracle (a transcription of the same pipeline) compare field by field. Both lines
are the sine of a phase, so they are bounded in ``[-1, 1]`` and scale-invariant — they carry scale-invariance and
boundedness properties in place of the homogeneity test used for scale-dependent indicators.

The ladder is the canonical one: contract, edge (warm-up / null / NaN latch), correctness (vs the closed-form reference
and a frozen golden master), and properties (reference agreement incl. missing data, scale-invariance, large-magnitude
stability, boundedness). Categories are split into classes; cross-cutting categories use markers (see
``tests/README.md``).
"""

import math
from collections.abc import Sequence

import polars as pl
from hypothesis import given
from hypothesis import strategies as st
from polars.testing import assert_frame_equal
from tests.indicators.oracles import sine_wave_reference
from tests.support import (
    ABSOLUTE_TOLERANCE_EXACT,
    ABSOLUTE_TOLERANCE_REFERENCE,
    BOUND_MARGIN,
    COLUMN_X,
    GROUP_KEY,
    RELATIVE_TOLERANCE_REFERENCE,
    assert_matches,
    assert_scale_homogeneous,
    count_leading_nulls,
    materialize_struct,
    spans_even_lag_repeat,
    two_segment_missing_data,
)

from pomata.indicators import sine_wave

# A deterministic 80-bar carrier (a clean 20-bar cycle): long enough to clear the 63-bar warm-up and emit 17 values.
_SAMPLE = [100.0 + 10.0 * math.sin(2 * math.pi * index / 20) for index in range(80)]
_WARMUP = 63
FIELDS = ("sine", "lead_sine")

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- derived, not chosen; the rationale is the shared method in CORRECTNESS.md, the numbers are this
# indicator's. To add an indicator, set its three facts here; the property tier below is then the same shape as every
# other indicator's.
#   1. warmup  W = 63   (a constant, not window-dependent: the recursive smoothers' settling plus the dominant-cycle
#              look-back the running discrete transform needs before the first sine value is emitted)
#   2. memory  the oracle is a transcription of the same pipeline and seeding, so the property holds from
#              the first defined row (M = 0); each example carries D in [1, W] defined rows past the warm-up -- output
#              to check, never an all-warm-up series
#   3. domain  positive floats (the carrier is a price); the missing-data tier draws a finite prefix past the warm-up
#              then a missing tail (two_segment_missing_data), and the agreement / scale tiers require a series with no
#              even-lag repeat (see those tests). Lengths span W + 1 .. 2 * W bars
# sine_wave is a parameter-free recursive cycle kernel whose lines are the sine of a phase: bounded in [-1, 1] and
# scale-INVARIANT (an O(1) output, not a price level), so the scale tier asserts invariance under a power-of-two
# rescaling (kept bit-exact, abs_tol 1e-12) rather than degree-1 homogeneity, and the large-magnitude tier uses the
# ABSOLUTE reference tolerance -- an O(1) value is never sized to the input magnitude. It has no window parameter, so
# ``_cases`` draws only the series (no window to couple), coupling the length to W instead. Repetitions N are the
# shared CI profile (tests/conftest.py); override per-test only if its parameter space is larger.
# ----------------------------------------------------------------------------------------------------------------------


@st.composite
def _cases[T](draw: st.DrawFn, values: st.SearchStrategy[T]) -> list[T]:
    """
    A series sized from the facts above. sine_wave is windowless (constant warm-up W = 63), so -- unlike the windowed
    indicators' ``(series, window)`` pair -- a case is just the series, with length = warm-up + a run of defined rows
    so every example has output to check (never an all-warm-up series).
    """
    defined = draw(st.integers(min_value=1, max_value=_WARMUP))
    length = _WARMUP + defined
    return draw(st.lists(values, min_size=length, max_size=length))


def apply_sine_wave(values: Sequence[float | None]) -> dict[str, list[float | None]]:
    """
    Materialize each field of ``sine_wave`` over a one-column ``Float64`` frame, as a dict mirroring the oracle output.
    """
    return materialize_struct(
        {COLUMN_X: values},
        sine_wave(pl.col(COLUMN_X)),
        FIELDS,
    )


class TestSineWaveContract:
    """
    Type, struct schema, shape, and lazy/eager guarantees.
    """

    def test_returns_expr(self) -> None:
        """
        Verifies that the factory returns a ``pl.Expr`` without touching a frame.
        """
        assert isinstance(sine_wave(pl.col(COLUMN_X)), pl.Expr)

    def test_output_is_struct_with_named_fields(self) -> None:
        """
        Verifies that the output is a ``Float64`` struct with exactly the fields ``sine`` / ``lead_sine``.
        """
        frame = pl.DataFrame({COLUMN_X: pl.Series(COLUMN_X, _SAMPLE)})
        dtype = frame.select(sine_wave(pl.col(COLUMN_X)).alias("s")).schema["s"]
        assert isinstance(dtype, pl.Struct)
        assert [field.name for field in dtype.fields] == ["sine", "lead_sine"]
        assert all(field.dtype == pl.Float64 for field in dtype.fields)

    def test_preserves_length(self) -> None:
        """
        Verifies that the output has one struct per input row.
        """
        result = pl.DataFrame({COLUMN_X: pl.Series(COLUMN_X, _SAMPLE)}).select(sine_wave(pl.col(COLUMN_X)).alias("s"))
        assert result.height == len(_SAMPLE)

    def test_lazy_eager_parity(self) -> None:
        """
        Verifies that eager and lazy application produce identical materialized output.
        """
        frame = pl.DataFrame({COLUMN_X: pl.Series(COLUMN_X, _SAMPLE)})
        expr = sine_wave(pl.col(COLUMN_X)).alias("s")
        assert_frame_equal(frame.select(expr), frame.lazy().select(expr).collect())

    def test_over_partitions_independently(self) -> None:
        """
        Verifies that under ``.over`` the recurrence resets per group and never spans group boundaries.
        """
        group_b_input = [50.0 + 5.0 * math.sin(2 * math.pi * index / 20) for index in range(80)]
        frame = pl.DataFrame({GROUP_KEY: ["a"] * 80 + ["b"] * 80, COLUMN_X: _SAMPLE + group_b_input})
        sine = sine_wave(pl.col(COLUMN_X)).over(GROUP_KEY).struct.field("sine")
        result = frame.select(sine.alias("y"))["y"].to_list()
        expected = apply_sine_wave(_SAMPLE)["sine"] + apply_sine_wave(group_b_input)["sine"]
        assert_matches(result, expected)


class TestSineWaveEdge:
    """
    Warm-up and null / NaN latching.
    """

    def test_empty(self) -> None:
        """
        Verifies that an empty input yields an empty output on both fields.
        """
        result = apply_sine_wave([])
        for field in FIELDS:
            assert_matches(result[field], [])

    def test_all_null(self) -> None:
        """
        Verifies that an all-null series yields an all-null output on both fields.
        """
        result = apply_sine_wave([None, None, None, None, None])
        for field in FIELDS:
            assert_matches(result[field], [None, None, None, None, None])

    def test_warmup_null_count(self) -> None:
        """
        Verifies that the leading-null run is exactly ``63`` on each line and every later row is defined.
        """
        bands = apply_sine_wave(_SAMPLE)
        for field in FIELDS:
            leading_nulls = count_leading_nulls(bands[field])
            assert leading_nulls == _WARMUP
            assert all(value is not None for value in bands[field][_WARMUP:])

    def test_null_latches(self) -> None:
        """
        Verifies that a null latches null on each line for every row from there (matching the naive reference).
        """
        values: list[float | None] = [*_SAMPLE[:70], None, *_SAMPLE[71:]]
        bands = apply_sine_wave(values)
        reference = sine_wave_reference(values)
        for field in FIELDS:
            assert bands[field][69] is not None
            assert all(value is None for value in bands[field][70:])
            assert_matches(bands[field], reference[field])

    def test_nan_latches(self) -> None:
        """
        Verifies that a NaN latches null on each line for every row from there (matching the naive reference).
        """
        values: list[float | None] = [*_SAMPLE[:70], math.nan, *_SAMPLE[71:]]
        bands = apply_sine_wave(values)
        reference = sine_wave_reference(values)
        for field in FIELDS:
            assert all(value is None for value in bands[field][70:])
            assert_matches(bands[field], reference[field])


class TestSineWaveCorrectness:
    """
    Against the reference oracle (internal-consistency for this recurrence) and frozen golden-master values.
    """

    def test_matches_reference(self) -> None:
        """
        Verifies that both lines agree with the naive reference on the sample carrier.
        """
        bands = apply_sine_wave(_SAMPLE)
        reference = sine_wave_reference(_SAMPLE)
        for field in FIELDS:
            assert_matches(
                bands[field],
                reference[field],
                rel_tol=RELATIVE_TOLERANCE_REFERENCE,
                abs_tol=ABSOLUTE_TOLERANCE_EXACT,
            )

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference: the 17 emitted sine / lead-sine values over the sample carrier.
        """
        bands = apply_sine_wave(_SAMPLE)
        assert_matches(
            [None if value is None else round(value, 4) for value in bands["sine"][_WARMUP:]],
            [
                0.8109,
                0.9521,
                1.0,
                0.9501,
                0.8074,
                0.5856,
                0.3063,
                -0.0031,
                -0.3122,
                -0.5907,
                -0.8111,
                -0.9521,
                -1.0,
                -0.9501,
                -0.8073,
                -0.5855,
                -0.3063,
            ],
        )
        assert_matches(
            [None if value is None else round(value, 4) for value in bands["lead_sine"][_WARMUP:]],
            [
                0.9872,
                0.8895,
                0.7049,
                0.4514,
                0.1537,
                -0.1591,
                -0.4565,
                -0.7093,
                -0.8925,
                -0.9882,
                -0.9871,
                -0.8894,
                -0.7047,
                -0.4512,
                -0.1536,
                0.1592,
                0.4565,
            ],
        )


class TestSineWaveProperties:
    """
    Invariants that must hold for all inputs (property-based).
    """

    @given(
        case=_cases(st.floats(min_value=1.0, max_value=1e3, allow_nan=False, allow_infinity=False)).filter(
            lambda series: not spans_even_lag_repeat(series)
        ),
    )
    def test_matches_reference_for_any_input(self, case: list[float]) -> None:
        """
        Verifies that, for any series with no even-lag repeat, both lines match the naive reference.

        Ehlers' six-tap quadrature filter reads the four-bar smooth at even lags, so the in-phase component collapses to
        a pure cancellation residual whenever the smooth repeats two bars apart — a flat run, but also a period-two
        alternation, both of which leave ``x[i] == x[i - 2]``. There the implementation's explicit FIR and the oracle's
        compensated ``sum()`` round to opposite sides of zero, flipping the ``inphase != 0`` phasor branch and so the
        phase that fixes both lines; the two transcriptions cannot be expected to agree across that
        discontinuity. The plain ``earlier != later`` check misses the alternation, so the filter excludes any even-lag
        repeat (see :func:`spans_even_lag_repeat`).
        """
        values = case
        bands = apply_sine_wave(values)
        reference = sine_wave_reference(values)
        for field in FIELDS:
            assert_matches(
                bands[field],
                reference[field],
                rel_tol=RELATIVE_TOLERANCE_REFERENCE,
                abs_tol=ABSOLUTE_TOLERANCE_EXACT,
            )

    @given(
        case=_cases(st.floats(min_value=1.0, max_value=1e3, allow_nan=False, allow_infinity=False)),
        exponent=st.sampled_from([-4, -3, -2, -1, 1, 2, 3, 4]),
    )
    def test_scale_invariance(self, case: list[float], exponent: int) -> None:
        """
        Verifies that the sine wave is scale-invariant: ``sine_wave(k * x) == sine_wave(x)``, for ANY carrier.

        The factor is a power of two, so the rescaling is lossless and the whole pipeline is exactly scale-equivariant:
        the phase that fixes both lines is unchanged to the bit. This holds for every carrier, including a low-amplitude
        one, because the phase guard tests the imaginary part against an EXACT zero rather than a fixed magnitude
        threshold (a fixed cutoff would snap a small-amplitude carrier to a different branch under the rescale).
        """
        factor = 2.0**exponent
        values = case
        base = apply_sine_wave(values)
        scaled = apply_sine_wave([value * factor for value in values])
        for field in FIELDS:
            assert_scale_homogeneous(scaled[field], base[field], k=factor, degree=0)

    @given(
        case=_cases(st.floats(min_value=1.0, max_value=1e3, allow_nan=False, allow_infinity=False)).filter(
            lambda series: not spans_even_lag_repeat(series)
        ),
        scale=st.sampled_from([2.0**-30, 2.0**30, 2.0**40]),
    )
    def test_matches_reference_at_large_magnitude(self, case: list[float], scale: float) -> None:
        """
        Verifies that at extreme magnitudes both lines stay finite where the reference is and agree.

        The carrier is required to have no even-lag repeat: a value equal to the value two bars earlier (a flat run, or
        a period-two alternation) drives the in-phase component to a pure cancellation residual that the explicit FIR
        and the oracle's tap-tuple sum round to opposite sides of zero, flipping the ``inphase != 0`` phasor branch and
        so the phase — an intrinsic discontinuity between the two transcriptions, and not a magnitude
        effect (see :func:`spans_even_lag_repeat`).
        """
        values = case
        scaled = [value * scale for value in values]
        bands = apply_sine_wave(scaled)
        reference = sine_wave_reference(scaled)
        for field in FIELDS:
            # NOTE: scale-invariant O(1) output (sine of a phase, bounded [-1, 1]) -> ABSOLUTE tolerance, never
            # input_scale-sized, per the magnitude-relative-factors home in tests/support/tolerances.py.
            assert_matches(
                bands[field],
                reference[field],
                rel_tol=RELATIVE_TOLERANCE_REFERENCE,
                abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
            )

    @given(case=two_segment_missing_data(_WARMUP))
    def test_matches_reference_under_missing_data(self, case: list[float | None]) -> None:
        """
        Verifies that, for a finite prefix clearing the warm-up followed by a null / NaN / finite tail, both lines match
        the naive reference.

        Drawn as two segments (finite prefix longer than the warm-up, then a missing-data tail) so defined output is
        emitted and then meets ``null`` / ``NaN`` — a single missing value anywhere in the warm-up latches the whole
        output to ``null``, so an all-mixed draw would almost always compare all-``null`` against all-``null`` and check
        nothing numeric.
        """
        values = case
        bands = apply_sine_wave(values)
        reference = sine_wave_reference(values)
        for field in FIELDS:
            assert_matches(
                bands[field], reference[field], rel_tol=RELATIVE_TOLERANCE_REFERENCE, abs_tol=ABSOLUTE_TOLERANCE_EXACT
            )

    @given(case=_cases(st.floats(min_value=1.0, max_value=1e3, allow_nan=False, allow_infinity=False)))
    def test_bounded(self, case: list[float]) -> None:
        """
        Verifies that every defined value of both lines lies within ``[-1, 1]``.
        """
        values = case
        bands = apply_sine_wave(values)
        for field in FIELDS:
            for value in bands[field]:
                if value is not None and not math.isnan(value):
                    assert -1.0 - BOUND_MARGIN <= value <= 1.0 + BOUND_MARGIN
