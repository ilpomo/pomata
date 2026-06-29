"""
Tests for ``pomata.indicators.mama`` — Ehlers' MESA Adaptive Moving Average and its companion FAMA.

``mama`` is the adaptive-average reader of the shared Hilbert-transform pipeline: it returns a single struct ``pl.Expr``
with the fields ``mama`` / ``fama``, whose smoothing constant tracks the rate of change of the dominant-cycle phase
between the ``limit_fast`` and ``limit_slow`` bounds. The local ``apply_mama`` helper materializes each field over a
one-column ``Float64`` frame into a dict of lists, so the shared ``assert_matches`` and the naive ``mama_reference``
oracle (a transcription of the same pipeline) compare field by field. Both lines ride the price scale, so
they are homogeneous of degree 1 — they carry the degree-1 scale-homogeneity and large-magnitude tests used for
scale-dependent indicators.

The ladder is the canonical one: contract, edge (parameter validation / warm-up / null / NaN latch), correctness (vs the
naive reference and a frozen golden master), and properties (reference agreement incl. missing data, degree-1
scale-homogeneity, large-magnitude stability). Categories are split into classes; cross-cutting categories use markers
(see ``tests/README.md``).
"""

import math
from collections.abc import Sequence

import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st
from polars.testing import assert_frame_equal
from tests.indicators.oracles import mama_reference
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

from pomata.indicators import mama

# A deterministic 40-bar carrier (a clean 20-bar cycle): long enough to clear the 32-bar warm-up and emit eight values.
_SAMPLE = [100.0 + 10.0 * math.sin(2 * math.pi * index / 20) for index in range(40)]
_WARMUP = 32
FIELDS = ("mama", "fama")

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- derived, not chosen; the rationale is the shared method in CORRECTNESS.md, the numbers are this
# indicator's. To add an indicator, set its three facts here; the property tier below is then the same shape as every
# other indicator's.
#   1. warmup  W = 32   (a constant, not window-dependent: the recursive Hilbert-transform smoothers need ~32 bars to
#              settle before the first adaptive-average value is emitted)
#   2. memory  the oracle is a transcription of the same pipeline and seeding, so the property holds from
#              the first defined row (M = 0); each example carries D in [1, W] defined rows past the warm-up -- output
#              to check, never an all-warm-up series
#   3. domain  positive floats (the carrier is a price); the missing-data tier draws a finite prefix past the warm-up
#              then a missing tail (two_segment_missing_data). The agreement / scale tiers require a series with no
#              even-lag repeat (see those tests). Lengths span W + 1 .. 2 * W bars
# mama is a parameter-free recursive cycle kernel (the ``limit_fast`` / ``limit_slow`` bounds are fixed defaults here),
# homogeneous of degree 1 (each line is a price level); its scale tier is sized by EXACT_TOLERANCE_FACTOR (an exact
# recursion under a power-of-two rescaling, so the residual is essentially zero and the factor is generous slack). It
# has no window parameter, so ``_cases`` draws only the series (no window to couple), coupling the length to W instead.
# Repetitions N are the shared CI profile (tests/conftest.py); override per-test only if its parameter space is larger.
# ----------------------------------------------------------------------------------------------------------------------


@st.composite
def _cases[T](draw: st.DrawFn, values: st.SearchStrategy[T]) -> list[T]:
    """
    A series sized from the facts above. mama is windowless (constant warm-up W = 32), so -- unlike the windowed
    indicators' ``(series, window)`` pair -- a case is just the series, with length = warm-up + a run of defined rows
    so every example has output to check (never an all-warm-up series).
    """
    defined = draw(st.integers(min_value=1, max_value=_WARMUP))
    length = _WARMUP + defined
    return draw(st.lists(values, min_size=length, max_size=length))


def apply_mama(
    values: Sequence[float | None],
    limit_fast: float = 0.5,
    limit_slow: float = 0.05,
) -> dict[str, list[float | None]]:
    """
    Materialize each field of ``mama`` over a one-column ``Float64`` frame, as a dict mirroring the oracle's output.
    """
    return materialize_struct(
        {COLUMN_X: values},
        mama(pl.col(COLUMN_X), limit_fast=limit_fast, limit_slow=limit_slow),
        FIELDS,
    )


class TestMamaContract:
    """
    Type, struct schema, shape, and lazy/eager guarantees.
    """

    def test_returns_expr(self) -> None:
        """
        Verifies that the factory returns a ``pl.Expr`` without touching a frame.
        """
        assert isinstance(mama(pl.col(COLUMN_X)), pl.Expr)

    def test_output_is_struct_with_named_fields(self) -> None:
        """
        Verifies that the output is a ``Float64`` struct with exactly the fields ``mama`` / ``fama``.
        """
        frame = pl.DataFrame({COLUMN_X: pl.Series(COLUMN_X, _SAMPLE)})
        dtype = frame.select(mama(pl.col(COLUMN_X)).alias("a")).schema["a"]
        assert isinstance(dtype, pl.Struct)
        assert [field.name for field in dtype.fields] == ["mama", "fama"]
        assert all(field.dtype == pl.Float64 for field in dtype.fields)

    def test_preserves_length(self) -> None:
        """
        Verifies that the output has one struct per input row.
        """
        result = pl.DataFrame({COLUMN_X: pl.Series(COLUMN_X, _SAMPLE)}).select(mama(pl.col(COLUMN_X)).alias("a"))
        assert result.height == len(_SAMPLE)

    def test_lazy_eager_parity(self) -> None:
        """
        Verifies that eager and lazy application produce identical materialized output.
        """
        frame = pl.DataFrame({COLUMN_X: pl.Series(COLUMN_X, _SAMPLE)})
        expr = mama(pl.col(COLUMN_X)).alias("a")
        assert_frame_equal(frame.select(expr), frame.lazy().select(expr).collect())

    def test_over_partitions_independently(self) -> None:
        """
        Verifies that under ``.over`` the recurrence resets per group and never spans group boundaries.
        """
        group_b_input = [50.0 + 5.0 * math.sin(2 * math.pi * index / 20) for index in range(40)]
        frame = pl.DataFrame({GROUP_KEY: ["a"] * 40 + ["b"] * 40, COLUMN_X: _SAMPLE + group_b_input})
        line = mama(pl.col(COLUMN_X)).over(GROUP_KEY).struct.field("mama")
        result = frame.select(line.alias("y"))["y"].to_list()
        expected = apply_mama(_SAMPLE)["mama"] + apply_mama(group_b_input)["mama"]
        assert_matches(result, expected)


class TestMamaEdge:
    """
    Parameter validation, warm-up, and null / NaN latching.
    """

    def test_limit_fast_not_positive_raises(self) -> None:
        """
        Verifies that ``limit_fast <= 0`` raises ``ValueError``.
        """
        with pytest.raises(ValueError, match="limit_fast must be in"):
            mama(pl.col(COLUMN_X), limit_fast=0.0)

    def test_limit_slow_not_positive_raises(self) -> None:
        """
        Verifies that ``limit_slow <= 0`` raises ``ValueError``.
        """
        with pytest.raises(ValueError, match="limit_slow must be in"):
            mama(pl.col(COLUMN_X), limit_slow=0.0)

    def test_limit_fast_above_one_raises(self) -> None:
        """
        Verifies that ``limit_fast > 1`` raises ``ValueError``: the smoothing constant is a weight in ``(0, 1]``, so a
        larger limit makes ``1 - alpha`` negative and the recurrence diverges.
        """
        with pytest.raises(ValueError, match="limit_fast must be in"):
            mama(pl.col(COLUMN_X), limit_fast=3.0)

    def test_limit_slow_above_one_raises(self) -> None:
        """
        Verifies that ``limit_slow > 1`` raises ``ValueError``: the smoothing constant is a weight in ``(0, 1]``, so a
        larger limit makes ``1 - alpha`` negative and the recurrence diverges.
        """
        with pytest.raises(ValueError, match="limit_slow must be in"):
            mama(pl.col(COLUMN_X), limit_slow=2.5)

    def test_limit_fast_below_limit_slow_raises(self) -> None:
        """
        Verifies that ``limit_fast < limit_slow`` raises ``ValueError``: ``limit_fast`` is the upper bound on the
        adaptive smoothing constant, so a value below ``limit_slow`` would pin it at ``limit_slow`` and make the bound
        false.
        """
        with pytest.raises(ValueError, match="limit_fast must be >= limit_slow"):
            mama(pl.col(COLUMN_X), limit_fast=0.05, limit_slow=0.5)

    def test_empty(self) -> None:
        """
        Verifies that an empty input yields an empty output on both fields.
        """
        bands = apply_mama([])
        for field in FIELDS:
            assert_matches(bands[field], [])

    def test_all_null(self) -> None:
        """
        Verifies that an all-null series yields an all-null output on both fields.
        """
        bands = apply_mama([None, None, None, None, None])
        for field in FIELDS:
            assert_matches(bands[field], [None, None, None, None, None])

    def test_warmup_null_count(self) -> None:
        """
        Verifies that, on each line, the leading-null run is exactly ``32`` and every later row is defined.
        """
        bands = apply_mama(_SAMPLE)
        for field in FIELDS:
            leading_nulls = count_leading_nulls(bands[field])
            assert leading_nulls == _WARMUP
            assert all(value is not None for value in bands[field][_WARMUP:])

    def test_null_latches(self) -> None:
        """
        Verifies that a null latches null on each line for every row from there (matching the naive reference).
        """
        values: list[float | None] = [*_SAMPLE[:35], None, *_SAMPLE[36:]]
        bands = apply_mama(values)
        reference = mama_reference(values)
        for field in FIELDS:
            assert bands[field][34] is not None
            assert all(value is None for value in bands[field][35:])
            assert_matches(bands[field], reference[field])

    def test_nan_latches(self) -> None:
        """
        Verifies that a NaN latches null on each line for every row from there (matching the naive reference).
        """
        values: list[float | None] = [*_SAMPLE[:35], math.nan, *_SAMPLE[36:]]
        bands = apply_mama(values)
        reference = mama_reference(values)
        for field in FIELDS:
            assert all(value is None for value in bands[field][35:])
            assert_matches(bands[field], reference[field])


class TestMamaCorrectness:
    """
    Against the reference oracle (internal-consistency for this recurrence) and frozen golden-master values.
    """

    def test_matches_reference(self) -> None:
        """
        Verifies that both lines agree with the naive reference on the sample carrier.
        """
        bands = apply_mama(_SAMPLE)
        reference = mama_reference(_SAMPLE)
        for field in FIELDS:
            assert_matches(
                bands[field], reference[field], rel_tol=RELATIVE_TOLERANCE_REFERENCE, abs_tol=ABSOLUTE_TOLERANCE_EXACT
            )

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference: the eight emitted values of each line over the sample carrier.
        """
        bands = apply_mama(_SAMPLE)
        assert_matches(
            [None if value is None else round(value, 4) for value in bands["mama"][_WARMUP:]],
            [97.9767, 97.6734, 97.3142, 96.9485, 96.6255, 96.3897, 96.2764, 96.308],
        )
        assert_matches(
            [None if value is None else round(value, 4) for value in bands["fama"][_WARMUP:]],
            [99.6954, 99.6448, 99.5866, 99.5206, 99.4482, 99.3718, 99.2944, 99.2197],
        )


class TestMamaProperties:
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
        smoothing constant; the two transcriptions cannot be expected to agree across that discontinuity.
        The plain ``earlier != later`` check misses the alternation, so the filter excludes any even-lag repeat (see
        :func:`spans_even_lag_repeat`).
        """
        values = case
        bands = apply_mama(values)
        reference = mama_reference(values)
        for field in FIELDS:
            assert_matches(
                bands[field], reference[field], rel_tol=RELATIVE_TOLERANCE_REFERENCE, abs_tol=ABSOLUTE_TOLERANCE_EXACT
            )

    @given(
        case=_cases(st.floats(min_value=1.0, max_value=1e3, allow_nan=False, allow_infinity=False)),
        exponent=st.sampled_from([-4, -3, -2, -1, 1, 2, 3, 4]),
    )
    def test_scale_homogeneity(self, case: list[float], exponent: int) -> None:
        """
        Verifies that both lines are homogeneous of degree 1: ``mama(k * x) == k * mama(x)`` (and likewise for FAMA).

        The smoothing constant is fixed by the phasor-phase rate, which is scale-invariant, so each line is exactly
        linear in the price. The factor is a power of two, so the rescaling is lossless: on a near-constant series the
        in-phase component is a sub-machine-epsilon cancellation residual whose vanishing flips the ``inphase != 0``
        branch under a non-dyadic rescale, and only a bit-exact factor keeps the two recurrences on the same branch.
        """
        k = 2.0**exponent
        values = case
        base = apply_mama(values)
        scaled = apply_mama([value * k for value in values])
        for field in FIELDS:
            assert_scale_homogeneous(scaled[field], base[field], k=k, degree=1)

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
        so the smoothing constant — an intrinsic discontinuity between the two transcriptions, and not a
        magnitude effect (see :func:`spans_even_lag_repeat`).
        """
        values = case
        scaled = [value * scale for value in values]
        bands = apply_mama(scaled)
        reference = mama_reference(scaled)
        for field in FIELDS:
            assert_matches(
                bands[field],
                reference[field],
                rel_tol=RELATIVE_TOLERANCE_REFERENCE,
                abs_tol=input_scale(scaled) * EXACT_TOLERANCE_FACTOR,
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
        bands = apply_mama(values)
        reference = mama_reference(values)
        for field in FIELDS:
            assert_matches(bands[field], reference[field])
