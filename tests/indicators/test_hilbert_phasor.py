"""
Tests for ``pomata.indicators.hilbert_phasor`` — Ehlers' Hilbert-transform phasor (in-phase / quadrature).

``hilbert_phasor`` is multi-output: the detrended price resolved by the Hilbert transform into the complex phasor whose
rotation traces the dominant cycle, returned as a single struct ``pl.Expr`` with the fields ``in_phase`` /
``quadrature``. The local ``apply_hilbert_phasor`` helper materializes each field over a one-column ``Float64`` frame
into a dict of lists, so the shared ``assert_matches`` and the naive ``hilbert_phasor_reference`` oracle (an independent
transcription of the same pipeline) compare field by field. Both components carry the price's units, so they are
homogeneous of degree 1 — a rescaling of the price rescales each component by the same factor.

The ladder is the canonical one: contract, edge (warm-up / null / NaN latch), correctness (vs the naive reference and a
frozen golden master), and properties (reference agreement incl. missing data, degree-1 scale-homogeneity,
large-magnitude stability). Categories are split into classes; cross-cutting categories use markers (see
``tests/README.md``).
"""

import math
from collections.abc import Sequence

import polars as pl
from hypothesis import given
from hypothesis import strategies as st
from polars.testing import assert_frame_equal
from tests.indicators.oracles import hilbert_phasor_reference
from tests.support import (
    ABSOLUTE_TOLERANCE_EXACT,
    COLUMN_X,
    EXACT_TOLERANCE_FACTOR,
    GROUP_KEY,
    RELATIVE_TOLERANCE_REFERENCE,
    assert_matches,
    assert_scale_homogeneous,
    count_leading_nulls,
    input_scale,
    materialize_struct,
    spans_even_lag_repeat,
    two_segment_missing_data,
)

from pomata.indicators import hilbert_phasor

# A deterministic 40-bar carrier (a clean 20-bar cycle): long enough to clear the 32-bar warm-up and emit eight values.
_SAMPLE = [100.0 + 10.0 * math.sin(2 * math.pi * index / 20) for index in range(40)]
_WARMUP = 32
FIELDS = ("in_phase", "quadrature")

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- derived, not chosen; the rationale is the shared method in CORRECTNESS.md, the numbers are this
# indicator's. To add an indicator, set its three facts here; the property tier below is then the same shape as every
# other indicator's.
#   1. warmup  W = 32   (a constant, not window-dependent: the recursive Hilbert-transform smoothers need ~32 bars to
#              settle before the first phasor value is emitted)
#   2. memory  the oracle is a transcription of the same pipeline and seeding, so the property holds from
#              the first defined row (M = 0); each example carries D in [1, W] defined rows past the warm-up -- output
#              to check, never an all-warm-up series
#   3. domain  positive floats (the carrier is a price); the missing-data tier draws a finite prefix past the warm-up
#              then a missing tail (two_segment_missing_data), and the any-input tier excludes any even-lag repeat (flat
#              run / period-two alternation). Lengths span W + 1 .. 2 * W bars
# hilbert_phasor is a parameter-free recursive cycle kernel, homogeneous of degree 1 (each component carries the price's
# units); its scale tier is sized by EXACT_TOLERANCE_FACTOR (an exact recursion under a power-of-two rescaling, so
# the residual is essentially zero and the factor is generous slack). It has no window parameter, so ``_cases`` draws
# only the series (no window to couple), coupling the length to W instead. Repetitions N are the shared CI profile
# (tests/conftest.py); override per-test only if its parameter space is larger.
# ----------------------------------------------------------------------------------------------------------------------


@st.composite
def _cases[T](draw: st.DrawFn, values: st.SearchStrategy[T]) -> list[T]:
    """
    A series sized from the facts above. hilbert_phasor is windowless (constant warm-up W = 32), so -- unlike the
    windowed indicators' ``(series, window)`` pair -- a case is just the series, with length = warm-up + a run of
    defined rows so every example has output to check (never an all-warm-up series).
    """
    defined = draw(st.integers(min_value=1, max_value=_WARMUP))
    length = _WARMUP + defined
    return draw(st.lists(values, min_size=length, max_size=length))


def apply_hilbert_phasor(values: Sequence[float | None]) -> dict[str, list[float | None]]:
    """
    Materialize each field of ``hilbert_phasor`` over a one-column frame, as a dict mirroring the oracle's output.
    """
    return materialize_struct(
        {COLUMN_X: values},
        hilbert_phasor(pl.col(COLUMN_X)),
        FIELDS,
    )


class TestHilbertPhasorContract:
    """
    Type, struct schema, shape, and lazy/eager guarantees.
    """

    def test_returns_expr(self) -> None:
        """
        Verifies that the factory returns a ``pl.Expr`` without touching a frame.
        """
        assert isinstance(hilbert_phasor(pl.col(COLUMN_X)), pl.Expr)

    def test_output_is_struct_with_named_fields(self) -> None:
        """
        Verifies that the output is a ``Float64`` struct with exactly the fields ``in_phase`` / ``quadrature``.
        """
        frame = pl.DataFrame({COLUMN_X: pl.Series(COLUMN_X, _SAMPLE)})
        dtype = frame.select(hilbert_phasor(pl.col(COLUMN_X)).alias("a")).schema["a"]
        assert isinstance(dtype, pl.Struct)
        assert [field.name for field in dtype.fields] == ["in_phase", "quadrature"]
        assert all(field.dtype == pl.Float64 for field in dtype.fields)

    def test_preserves_length(self) -> None:
        """
        Verifies that the output has one struct per input row.
        """
        frame = pl.DataFrame({COLUMN_X: pl.Series(COLUMN_X, _SAMPLE)})
        result = frame.select(hilbert_phasor(pl.col(COLUMN_X)).alias("a"))
        assert result.height == len(_SAMPLE)

    def test_lazy_eager_parity(self) -> None:
        """
        Verifies that eager and lazy application produce identical materialized output.
        """
        frame = pl.DataFrame({COLUMN_X: pl.Series(COLUMN_X, _SAMPLE)})
        expr = hilbert_phasor(pl.col(COLUMN_X)).alias("a")
        assert_frame_equal(frame.select(expr), frame.lazy().select(expr).collect())

    def test_over_partitions_independently(self) -> None:
        """
        Verifies that under ``.over`` the recurrence resets per group and never spans group boundaries.
        """
        group_b_input = [50.0 + 5.0 * math.sin(2 * math.pi * index / 20) for index in range(40)]
        frame = pl.DataFrame({GROUP_KEY: ["a"] * 40 + ["b"] * 40, COLUMN_X: _SAMPLE + group_b_input})
        in_phase = hilbert_phasor(pl.col(COLUMN_X)).over(GROUP_KEY).struct.field("in_phase")
        grouped = frame.select(in_phase.alias("y"))["y"].to_list()
        expected = apply_hilbert_phasor(_SAMPLE)["in_phase"] + apply_hilbert_phasor(group_b_input)["in_phase"]
        assert_matches(grouped, expected)


class TestHilbertPhasorEdge:
    """
    Warm-up and null / NaN latching.
    """

    def test_empty(self) -> None:
        """
        Verifies that an empty input yields an empty output on both fields.
        """
        result = apply_hilbert_phasor([])
        for field in FIELDS:
            assert_matches(result[field], [])

    def test_all_null(self) -> None:
        """
        Verifies that an all-null series yields an all-null output on both fields.
        """
        result = apply_hilbert_phasor([None, None, None, None, None])
        for field in FIELDS:
            assert_matches(result[field], [None, None, None, None, None])

    def test_warmup_null_count(self) -> None:
        """
        Verifies that the leading-null run is exactly ``32`` on both fields and every later row is defined.
        """
        result = apply_hilbert_phasor(_SAMPLE)
        for field in FIELDS:
            leading_nulls = count_leading_nulls(result[field])
            assert leading_nulls == _WARMUP
            assert all(value is not None for value in result[field][_WARMUP:])

    def test_null_latches(self) -> None:
        """
        Verifies that a null latches null on both fields for every row from there (matching the naive reference).
        """
        values: list[float | None] = [*_SAMPLE[:35], None, *_SAMPLE[36:]]
        result = apply_hilbert_phasor(values)
        reference = hilbert_phasor_reference(values)
        for field in FIELDS:
            assert result[field][34] is not None
            assert all(value is None for value in result[field][35:])
            assert_matches(result[field], reference[field])

    def test_nan_latches(self) -> None:
        """
        Verifies that a NaN latches null on both fields for every row from there (matching the naive reference).
        """
        values: list[float | None] = [*_SAMPLE[:35], math.nan, *_SAMPLE[36:]]
        result = apply_hilbert_phasor(values)
        reference = hilbert_phasor_reference(values)
        for field in FIELDS:
            assert all(value is None for value in result[field][35:])
            assert_matches(result[field], reference[field])


class TestHilbertPhasorCorrectness:
    """
    Against the reference oracle (internal-consistency for this recurrence) and frozen golden-master values.
    """

    def test_matches_reference(self) -> None:
        """
        Verifies that both fields agree with the naive reference on the sample carrier.
        """
        result = apply_hilbert_phasor(_SAMPLE)
        reference = hilbert_phasor_reference(_SAMPLE)
        for field in FIELDS:
            assert_matches(
                result[field], reference[field], rel_tol=RELATIVE_TOLERANCE_REFERENCE, abs_tol=ABSOLUTE_TOLERANCE_EXACT
            )

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference: the eight emitted in-phase / quadrature values over the sample carrier.
        """
        result = apply_hilbert_phasor(_SAMPLE)
        assert_matches(
            [None if value is None else round(value, 4) for value in result["in_phase"][_WARMUP:]],
            [-0.0296, -2.9751, -5.7341, -7.9735, -9.445, -10.0056, -9.5949, -8.227],
        )
        assert_matches(
            [None if value is None else round(value, 4) for value in result["quadrature"][_WARMUP:]],
            [-9.5596, -9.5537, -8.4419, -6.3428, -3.5083, -0.2378, 3.1502, 6.3025],
        )


class TestHilbertPhasorProperties:
    """
    Invariants that must hold for all inputs (property-based).
    """

    # NOTE: no even-lag repeat (x[i] == x[i-2]) -- a flat run or period-two alternation drives the phasor's in-phase /
    # quadrature toward a cancellation residual that exceeds the near-machine band; the same exclusion mama / sine_wave
    # apply on this pipeline (see spans_even_lag_repeat).
    @given(
        case=_cases(st.floats(min_value=1.0, max_value=1e3, allow_nan=False, allow_infinity=False)).filter(
            lambda series: not spans_even_lag_repeat(series)
        )
    )
    def test_matches_reference_for_any_input(self, case: list[float]) -> None:
        """
        Verifies that, for any series with no even-lag repeat, both fields match the naive reference.
        """
        values = case
        result = apply_hilbert_phasor(values)
        reference = hilbert_phasor_reference(values)
        for field in FIELDS:
            assert_matches(
                result[field], reference[field], rel_tol=RELATIVE_TOLERANCE_REFERENCE, abs_tol=ABSOLUTE_TOLERANCE_EXACT
            )

    @given(
        case=_cases(st.floats(min_value=1.0, max_value=1e3, allow_nan=False, allow_infinity=False)),
        exponent=st.sampled_from([-4, -3, -2, -1, 1, 2, 3, 4]),
    )
    def test_scale_homogeneity(self, case: list[float], exponent: int) -> None:
        """
        Verifies that ``hilbert_phasor`` is homogeneous of degree 1: each component scales with the price, so
        ``field(k * x) == k * field(x)``. ``k`` is a power of two so the rescaling is lossless.
        """
        k = 2.0**exponent
        values = case
        base = apply_hilbert_phasor(values)
        scaled = apply_hilbert_phasor([value * k for value in values])
        for field in FIELDS:
            assert_scale_homogeneous(scaled[field], base[field], k=k, degree=1)

    @given(
        case=_cases(st.floats(min_value=1.0, max_value=1e3, allow_nan=False, allow_infinity=False)),
        scale=st.sampled_from([2.0**-30, 2.0**30, 2.0**40]),
    )
    def test_matches_reference_at_large_magnitude(self, case: list[float], scale: float) -> None:
        """
        Verifies that at extreme magnitudes both fields stay finite where the reference is and agree.
        """
        values = case
        scaled = [value * scale for value in values]
        result = apply_hilbert_phasor(scaled)
        reference = hilbert_phasor_reference(scaled)
        for field in FIELDS:
            assert_matches(
                result[field],
                reference[field],
                rel_tol=RELATIVE_TOLERANCE_REFERENCE,
                abs_tol=input_scale(scaled) * EXACT_TOLERANCE_FACTOR,
            )

    @given(case=two_segment_missing_data(_WARMUP))
    def test_matches_reference_under_missing_data(self, case: list[float | None]) -> None:
        """
        Verifies that, for a finite prefix clearing the warm-up followed by a null / NaN / finite tail, both fields
        match the naive reference.

        Drawn as two segments (finite prefix longer than the warm-up, then a missing-data tail) so defined output is
        emitted and then meets ``null`` / ``NaN`` — a single missing value anywhere in the warm-up latches the whole
        output to ``null``, so an all-mixed draw would almost always compare all-``null`` against all-``null`` and check
        nothing numeric.
        """
        values = case
        result = apply_hilbert_phasor(values)
        reference = hilbert_phasor_reference(values)
        for field in FIELDS:
            assert_matches(result[field], reference[field])
