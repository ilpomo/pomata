"""
Tests for ``pomata.indicators.hilbert_trendline`` — Ehlers' Hilbert-transform instantaneous trendline.

``hilbert_trendline`` is a parameter-free, sequential ``map_batches`` kernel: a four-bar smooth, a Hilbert-transform
detrend into in-phase / quadrature components, and a homodyne discriminator give the dominant-cycle length, over which
the price is averaged and then smoothed — so the cyclic component cancels and only the trend remains. The local
``apply_hilbert_trendline`` helper materializes it over a one-column ``Float64`` frame; ``assert_matches`` and the naive
``hilbert_trendline_reference`` oracle (a transcription of the same pipeline) are shared across the suite.
The trendline rides the price scale — it is homogeneous of degree 1.

The ladder is the canonical one: contract, edge (warm-up / null / NaN latch), correctness (vs the closed-form reference
and a frozen golden master), and properties (reference agreement incl. missing data, degree-1 scale-homogeneity,
large-magnitude stability). Categories are split into classes; cross-cutting categories use markers (see
``tests/README.md``).
"""

import math
from collections.abc import Sequence

import polars as pl
from hypothesis import given
from hypothesis import strategies as st
from tests.indicators.oracles import hilbert_trendline_reference
from tests.support import (
    ABSOLUTE_TOLERANCE_EXACT,
    COLUMN_X,
    EXACT_TOLERANCE_FACTOR,
    GROUP_KEY,
    RELATIVE_TOLERANCE_REFERENCE,
    apply_expr,
    assert_matches,
    assert_scale_homogeneous,
    count_leading_nulls,
    input_scale,
    two_segment_missing_data,
)

from pomata.indicators import hilbert_trendline

# A deterministic 80-bar carrier (a 0.5/bar trend plus a clean 20-bar cycle): long enough to clear the 63-bar warm-up
# and emit seventeen values, and — unlike a pure cycle, whose trendline is flat — it varies bar to bar.
_SAMPLE = [100.0 + 0.5 * index + 10.0 * math.sin(2 * math.pi * index / 20) for index in range(80)]
_WARMUP = 63

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- derived, not chosen; the rationale is the shared method in CORRECTNESS.md, the numbers are this
# indicator's. To add an indicator, set its three facts here; the property tier below is then the same shape as every
# other indicator's.
#   1. warmup  W = 63   (a constant, not window-dependent: the recursive smoothers' settling plus the dominant-cycle
#              look-back the running discrete transform needs before the first trendline value is emitted)
#   2. memory  the oracle is a transcription of the same pipeline and seeding, so the property holds from
#              the first defined row (M = 0); each example carries D in [1, W] defined rows past the warm-up -- output
#              to check, never an all-warm-up series
#   3. domain  positive floats (the carrier is a price); the missing-data tier draws a finite prefix past the warm-up
#              then a missing tail (two_segment_missing_data). Lengths span W + 1 .. 2 * W bars
# hilbert_trendline is a parameter-free recursive cycle kernel, homogeneous of degree 1 (the trendline is a price
# level); its scale tier is sized by EXACT_TOLERANCE_FACTOR (an exact recursion under a power-of-two rescaling, so
# the residual is essentially zero and the factor is generous slack). It has no window parameter, so ``_cases`` draws
# only the series (no window to couple), coupling the length to W instead. Repetitions N are the shared CI profile
# (tests/conftest.py); override per-test only if its parameter space is larger.
# ----------------------------------------------------------------------------------------------------------------------


@st.composite
def _cases[T](draw: st.DrawFn, values: st.SearchStrategy[T]) -> list[T]:
    """
    A series sized from the facts above. hilbert_trendline is windowless (constant warm-up W = 63), so -- unlike the
    windowed indicators' ``(series, window)`` pair -- a case is just the series, with length = warm-up + a run of
    defined rows so every example has output to check (never an all-warm-up series).
    """
    defined = draw(st.integers(min_value=1, max_value=_WARMUP))
    length = _WARMUP + defined
    return draw(st.lists(values, min_size=length, max_size=length))


def apply_hilbert_trendline(values: Sequence[float | None]) -> list[float | None]:
    """
    Materialize ``hilbert_trendline`` over a one-column ``Float64`` frame built from ``values``.
    """
    return apply_expr(values, hilbert_trendline(pl.col(COLUMN_X)))


class TestHilbertTrendlineContract:
    """
    Type, shape, and lazy/eager guarantees.
    """

    def test_over_partitions_independently(self) -> None:
        """
        Verifies that under ``.over`` the recurrence resets per group and never spans group boundaries.
        """
        group_b_input = [50.0 + 0.25 * index + 5.0 * math.sin(2 * math.pi * index / 20) for index in range(80)]
        frame = pl.DataFrame({GROUP_KEY: ["a"] * 80 + ["b"] * 80, COLUMN_X: _SAMPLE + group_b_input})
        result = frame.select(hilbert_trendline(pl.col(COLUMN_X)).over(GROUP_KEY).alias("y"))["y"].to_list()
        expected = apply_hilbert_trendline(_SAMPLE) + apply_hilbert_trendline(group_b_input)
        assert_matches(result, expected)


class TestHilbertTrendlineEdge:
    """
    Warm-up and null / NaN latching.
    """

    def test_all_null(self) -> None:
        """
        Verifies that an all-null series yields an all-null output.
        """
        assert_matches(apply_hilbert_trendline([None, None, None, None, None]), [None, None, None, None, None])

    def test_warmup_null_count(self) -> None:
        """
        Verifies that the leading-null run is exactly ``63`` and every later row is defined.
        """
        result = apply_hilbert_trendline(_SAMPLE)
        leading_nulls = count_leading_nulls(result)
        assert leading_nulls == _WARMUP
        assert all(value is not None for value in result[_WARMUP:])

    def test_null_latches(self) -> None:
        """
        Verifies that a null latches null for every row from there (matching the naive reference).
        """
        values: list[float | None] = [*_SAMPLE[:70], None, *_SAMPLE[71:]]
        result = apply_hilbert_trendline(values)
        assert result[69] is not None
        assert all(value is None for value in result[70:])
        assert_matches(result, hilbert_trendline_reference(values))

    def test_nan_latches(self) -> None:
        """
        Verifies that a NaN latches null for every row from there (matching the naive reference).
        """
        values: list[float | None] = [*_SAMPLE[:70], math.nan, *_SAMPLE[71:]]
        result = apply_hilbert_trendline(values)
        assert all(value is None for value in result[70:])
        assert_matches(result, hilbert_trendline_reference(values))


class TestHilbertTrendlineCorrectness:
    """
    Against the reference oracle (internal-consistency for this recurrence) and frozen golden-master values.
    """

    def test_matches_reference(self) -> None:
        """
        Verifies agreement with the naive reference on the sample carrier.
        """
        assert_matches(
            apply_hilbert_trendline(_SAMPLE),
            hilbert_trendline_reference(_SAMPLE),
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
            abs_tol=ABSOLUTE_TOLERANCE_EXACT,
        )

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference: the seventeen emitted trendline values over the sample carrier.
        """
        result = apply_hilbert_trendline(_SAMPLE)
        assert_matches(
            [None if value is None else round(value, 4) for value in result[_WARMUP:]],
            [
                126.2134,
                126.7457,
                127.253,
                127.75,
                128.25,
                128.75,
                129.35,
                129.825,
                130.3,
                130.775,
                131.25,
                131.75,
                131.9595,
                132.251,
                132.6398,
                133.1343,
                133.7348,
            ],
        )


class TestHilbertTrendlineProperties:
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
            apply_hilbert_trendline(values),
            hilbert_trendline_reference(values),
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
            abs_tol=ABSOLUTE_TOLERANCE_EXACT,
        )

    @given(
        case=_cases(st.floats(min_value=1.0, max_value=1e3, allow_nan=False, allow_infinity=False)),
        exponent=st.sampled_from([-4, -3, -2, -1, 1, 2, 3, 4]),
    )
    def test_scale_homogeneity(self, case: list[float], exponent: int) -> None:
        """
        Verifies that ``hilbert_trendline`` is homogeneous of degree 1: ``hilbert_trendline(k * x) == k *
        hilbert_trendline(x)``. ``k`` is a power of two so the rescaling is lossless.
        """
        k = 2.0**exponent
        values = case
        base = apply_hilbert_trendline(values)
        scaled = apply_hilbert_trendline([value * k for value in values])
        assert_scale_homogeneous(scaled, base, k=k, degree=1)

    @given(
        case=_cases(st.floats(min_value=1.0, max_value=1e3, allow_nan=False, allow_infinity=False)),
        scale=st.sampled_from([2.0**-30, 2.0**30, 2.0**40]),
    )
    def test_matches_reference_at_large_magnitude(self, case: list[float], scale: float) -> None:
        """
        Verifies that at extreme magnitudes the implementation stays finite where the reference is and agrees.
        """
        values = case
        scaled = [value * scale for value in values]
        assert_matches(
            apply_hilbert_trendline(scaled),
            hilbert_trendline_reference(scaled),
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
            abs_tol=input_scale(scaled) * EXACT_TOLERANCE_FACTOR,
        )

    @given(case=two_segment_missing_data(_WARMUP))
    def test_matches_reference_under_missing_data(self, case: list[float | None]) -> None:
        """
        Verifies that, for a finite prefix clearing the warm-up followed by a null / NaN / finite tail, the
        implementation matches the naive reference.

        Drawn as two segments (finite prefix longer than the warm-up, then a missing-data tail) so a defined trendline
        is emitted and then meets ``null`` / ``NaN`` — a single missing value anywhere in the warm-up latches the whole
        output to ``null``, so an all-mixed draw would almost always compare all-``null`` against all-``null`` and check
        nothing numeric.
        """
        values = case
        assert_matches(
            apply_hilbert_trendline(values),
            hilbert_trendline_reference(values),
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
            abs_tol=ABSOLUTE_TOLERANCE_EXACT,
        )
