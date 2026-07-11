"""
Tests for ``pomata.indicators.vortex`` — paired trend oscillators from summed vortex movement over summed true range.

``vortex`` is multi-input (``high`` / ``low`` / ``close``) and returns a single struct ``pl.Expr`` with the fields
``plus`` / ``minus``. The local ``apply_vortex`` helper materializes each line over a three-column ``Float64`` frame
into a dict of lists (via the shared ``materialize_struct``), so the shared ``assert_matches`` and the naive
``vortex_reference`` oracle (the certified ``true_range`` plus matched rolling sums) compare line by line.

It is a quotient of rolling sums -- residual-prone past a sane dynamic range (see ``CORRECTNESS.md``) -- and
scale-invariant (a ratio), so the large-magnitude
tier is vacuous and a scale-invariance tier takes its place; the flat-window ``0 / 0`` is pinned in the edge tier. The
ladder is otherwise canonical: contract, edge (warm-up / one-bar lag null / NaN / flat), correctness (oracle + golden),
properties (reference agreement incl. missing data, non-negativity, scale-invariance). Categories are split into
classes; cross-cutting categories use markers (see ``tests/README.md``).
"""

import math
from collections.abc import Sequence

import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st
from tests.indicators.oracles import vortex_reference
from tests.support import (
    ABSOLUTE_TOLERANCE_PROPERTY,
    CLOSE,
    GROUP_KEY,
    HIGH,
    LOW,
    RELATIVE_TOLERANCE_PROPERTY,
    WINDOW_MAX,
    assert_matches,
    assert_scale_homogeneous,
    coherent_hlc,
    coherent_hlc_with_missing,
    materialize_struct,
    split_triples,
)

from pomata.indicators import vortex

FIELDS = ("plus", "minus")

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- derived, not chosen; the rationale is the shared method in CORRECTNESS.md, the numbers are this
# indicator's. To add an indicator, set its three facts here; the property tier below is then the same shape as every
# other indicator's.
#   1. warmup  W(window) = window   (the rolling sums need ``window`` defined movements, and the first movement is null
#              -- it reads the previous bar -- so the first defined row is index ``window``)
#   2. memory  the oracle uses the same rolling sums and true range, so the property holds from the first defined row
#              (M = 0); each example carries D in [window, 2 * window] defined bars
#   3. domain  coherent_hlc(): coherent (high >= low, low <= close <= high) positive-finite bars; over them the summed
#              true range is positive, so the ratio is well-conditioned (the flat-window 0/0 is pinned in edge)
# Windows span 1 .. WINDOW_MAX. Repetitions N are the shared CI profile (tests/conftest.py).
# ----------------------------------------------------------------------------------------------------------------------


@st.composite
def _cases[T](draw: st.DrawFn, bars: st.SearchStrategy[T]) -> tuple[list[T], int]:
    """
    A (series, window) pair sized from the facts above: length = the ``window`` warm-up + a window of defined bars, so
    every example has output on both lines to check.
    """
    window = draw(st.integers(min_value=1, max_value=WINDOW_MAX))
    defined = draw(st.integers(min_value=window, max_value=2 * window))
    length = window + defined
    return draw(st.lists(bars, min_size=length, max_size=length)), window


def apply_vortex(
    high: Sequence[float | None],
    low: Sequence[float | None],
    close: Sequence[float | None],
    window: int,
) -> dict[str, list[float | None]]:
    """
    Materialize each line of ``vortex`` over a three-column frame, as a dict mirroring the oracle's output.
    """
    return materialize_struct(
        {HIGH: high, LOW: low, CLOSE: close},
        vortex(pl.col(HIGH), pl.col(LOW), pl.col(CLOSE), window),
        FIELDS,
    )


class TestVortexContract:
    """
    Type, struct schema, shape, and lazy/eager guarantees.
    """

    def test_over_partitions_independently(self) -> None:
        """
        Verifies that under ``.over`` neither the lag nor the window spans group boundaries.
        """
        frame = pl.DataFrame(
            {
                GROUP_KEY: ["a", "a", "a", "b", "b", "b"],
                HIGH: [2.0, 4.0, 6.0, 12.0, 14.0, 16.0],
                LOW: [1.0, 3.0, 4.0, 11.0, 13.0, 14.0],
                CLOSE: [1.5, 3.5, 5.0, 11.5, 13.5, 15.0],
            }
        )
        plus = vortex(pl.col(HIGH), pl.col(LOW), pl.col(CLOSE), 2).over(GROUP_KEY).struct.field("plus")
        result = frame.select(plus.alias("y"))["y"].to_list()
        # Each group warms up over window = 2; group b must not read group a's last bar through the lag.
        assert result[:2] == [None, None]
        assert result[3:5] == [None, None]
        assert result[2] is not None
        assert result[5] is not None

    def test_output_is_struct_with_named_fields(self) -> None:
        """
        Verifies that the output is a ``Float64`` struct with exactly the fields ``plus`` / ``minus``.
        """
        frame = pl.DataFrame({HIGH: [2.0, 4.0, 6.0], LOW: [1.0, 3.0, 4.0], CLOSE: [1.5, 3.5, 5.0]})
        dtype = frame.select(vortex(pl.col(HIGH), pl.col(LOW), pl.col(CLOSE), 2).alias("v")).schema["v"]
        assert isinstance(dtype, pl.Struct)
        assert [field.name for field in dtype.fields] == ["plus", "minus"]
        assert all(field.dtype == pl.Float64 for field in dtype.fields)


class TestVortexEdge:
    """
    Boundaries, warm-up, the one-bar lag, null / NaN, and the flat window.
    """

    def test_window_below_one_raises(self) -> None:
        """
        Verifies that ``window < 1`` raises ``ValueError``.
        """
        with pytest.raises(ValueError, match="window must be >= 1"):
            vortex(pl.col(HIGH), pl.col(LOW), pl.col(CLOSE), 0)

    def test_single_row(self) -> None:
        """
        Verifies that a one-row series with ``window > 1`` is all warm-up (one null) on both lines.
        """
        bands = apply_vortex([2.0], [1.0], [1.5], 2)
        for field in FIELDS:
            assert_matches(bands[field], [None])

    def test_all_null(self) -> None:
        """
        Verifies that an all-null input yields an all-null output on both lines.
        """
        bands = apply_vortex([None, None, None], [None, None, None], [None, None, None], 2)
        for field in FIELDS:
            assert_matches(bands[field], [None, None, None])

    def test_null_propagates_through_lag(self) -> None:
        """
        Verifies that a ``null`` flows through the one-bar lag and the rolling sums exactly as the reference.
        """
        high = [2.0, None, 6.0, 8.0, 10.0, 12.0, 13.0]
        low = [1.0, 3.0, 4.0, 6.0, 8.0, 10.0, 11.0]
        close = [1.5, 2.5, 5.0, 7.0, 9.0, 11.0, 12.0]
        bands = apply_vortex(high, low, close, 2)
        reference = vortex_reference(high, low, close, 2)
        for field in FIELDS:
            assert_matches(bands[field], reference[field])

    def test_nan_propagates_through_lag(self) -> None:
        """
        Verifies that a ``NaN`` flows through the one-bar lag and the rolling sums exactly as the reference.
        """
        high = [2.0, 4.0, 6.0, 8.0, math.nan, 12.0, 13.0]
        low = [1.0, 3.0, 4.0, 6.0, 8.0, 10.0, 11.0]
        close = [1.5, 2.5, 5.0, 7.0, 9.0, 11.0, 12.0]
        bands = apply_vortex(high, low, close, 2)
        reference = vortex_reference(high, low, close, 2)
        for field in FIELDS:
            assert_matches(bands[field], reference[field])

    def test_warmup_null_count(self) -> None:
        """
        Verifies that both lines are null for the first ``window`` rows (the lag nulls the first movement) and defined
        from the next.
        """
        high = [2.0, 4.0, 6.0, 5.0, 7.0]
        low = [1.0, 3.0, 4.0, 4.0, 5.0]
        close = [1.5, 3.5, 5.0, 4.5, 6.0]
        bands = apply_vortex(high, low, close, 2)
        for field in FIELDS:
            assert bands[field][:2] == [None, None]
            assert bands[field][2] is not None

    def test_window_exceeds_length(self) -> None:
        """
        Verifies that a window exceeding the series length yields an all-null output on both lines.
        """
        bands = apply_vortex([2.0, 4.0, 6.0], [1.0, 3.0, 4.0], [1.5, 3.5, 5.0], 5)
        for field in FIELDS:
            assert_matches(bands[field], [None, None, None])

    def test_flat_window_is_nan(self) -> None:
        """
        Verifies the flat window: a constant series has zero summed true range and zero summed movement, so both lines
        are the indeterminate ``0 / 0 == NaN`` after warm-up.
        """
        flat = [10.0] * 6
        bands = apply_vortex(flat, flat, flat, 2)
        for field in FIELDS:
            assert_matches(bands[field], [None, None, math.nan, math.nan, math.nan, math.nan])

    def test_flat_window_is_nan_at_large_magnitude(self) -> None:
        """
        Verifies the exact-flat guard is residual-free: a constant series at a large magnitude (where a streaming
        quotient of rolling sums could accumulate a non-zero float residue) still yields ``NaN`` on both lines, because
        the flat window is detected via the residual-free rolling maximum of the true range rather than the summed
        quotient.
        """
        flat = [1e9] * 6
        bands = apply_vortex(flat, flat, flat, 2)
        for field in FIELDS:
            assert_matches(bands[field], [None, None, math.nan, math.nan, math.nan, math.nan])


class TestVortexCorrectness:
    """
    Against the naive reference oracle and frozen golden-master values.
    """

    def test_matches_reference(self) -> None:
        """
        Verifies that both lines agree with the naive reference across several windows.
        """
        high = [2.0, 4.0, 6.0, 5.0, 7.0, 9.0, 8.0, 11.0]
        low = [1.0, 3.0, 4.0, 4.0, 5.0, 7.0, 6.0, 9.0]
        close = [1.5, 3.5, 5.0, 4.5, 6.0, 8.0, 7.0, 10.0]
        for window in (1, 2, 3, 4):
            bands = apply_vortex(high, low, close, window)
            reference = vortex_reference(high, low, close, window)
            for field in FIELDS:
                assert_matches(bands[field], reference[field])

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference: vortex(window=2) over a worked five-bar series.
        """
        bands = apply_vortex([2.0, 4.0, 6.0, 5.0, 7.0], [1.0, 3.0, 4.0, 4.0, 5.0], [1.5, 3.5, 5.0, 4.5, 6.0], 2)
        assert_matches(bands["plus"], [None, None, 1.2, 8.0 / 7.0, 8.0 / 7.0])
        assert_matches(bands["minus"], [None, None, 0.2, 4.0 / 7.0, 4.0 / 7.0])


class TestVortexProperties:
    """
    Invariants that must hold for all inputs (property-based).
    """

    @given(case=_cases(coherent_hlc()))
    def test_matches_reference_for_any_input(
        self,
        case: tuple[list[tuple[float, float, float]], int],
    ) -> None:
        """
        Verifies that, for any coherent OHLC series and window, both lines match the naive reference.
        """
        rows, window = case
        high, low, close = split_triples(rows)
        bands = apply_vortex(high, low, close, window)
        reference = vortex_reference(high, low, close, window)
        for field in FIELDS:
            assert_matches(
                bands[field],
                reference[field],
                rel_tol=RELATIVE_TOLERANCE_PROPERTY,
                abs_tol=ABSOLUTE_TOLERANCE_PROPERTY,
            )

    @given(case=_cases(coherent_hlc_with_missing()))
    def test_matches_reference_under_missing_data(
        self,
        case: tuple[list[tuple[float | None, float | None, float | None]], int],
    ) -> None:
        """
        Verifies that, for inputs freely mixing null / NaN / finite, both lines match the naive reference.
        """
        rows, window = case
        high, low, close = split_triples(rows)
        bands = apply_vortex(high, low, close, window)
        reference = vortex_reference(high, low, close, window)
        for field in FIELDS:
            assert_matches(
                bands[field],
                reference[field],
                rel_tol=RELATIVE_TOLERANCE_PROPERTY,
                abs_tol=ABSOLUTE_TOLERANCE_PROPERTY,
            )

    @given(
        case=_cases(coherent_hlc()),
        exponent=st.sampled_from([-4, -3, -2, -1, 1, 2, 3, 4]),
    )
    def test_scale_invariance(
        self,
        case: tuple[list[tuple[float, float, float]], int],
        exponent: int,
    ) -> None:
        """
        Verifies that ``vortex`` is scale-invariant: scaling every input value by a constant ``k`` leaves the output
        unchanged -- ``vortex(k * x) == vortex(x)``. ``k`` is a power of two, so the rescale is exact and adds no
        floating-point error.
        """
        k = 2.0**exponent
        rows, window = case
        high, low, close = split_triples(rows)
        base = apply_vortex(high, low, close, window)
        scaled = apply_vortex([v * k for v in high], [v * k for v in low], [v * k for v in close], window)
        for field in FIELDS:
            assert_scale_homogeneous(scaled[field], base[field], k=k, degree=0)

    @given(case=_cases(coherent_hlc()))
    def test_lines_are_non_negative(
        self,
        case: tuple[list[tuple[float, float, float]], int],
    ) -> None:
        """
        Verifies the true invariant: each vortex line is non-negative wherever defined and finite (a sum of absolute
        movements over a sum of non-negative true ranges).
        """
        rows, window = case
        high, low, close = split_triples(rows)
        bands = apply_vortex(high, low, close, window)
        for field in FIELDS:
            for value in bands[field]:
                if value is not None and not math.isnan(value):
                    assert value >= 0.0
