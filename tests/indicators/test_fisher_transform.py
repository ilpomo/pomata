"""
Tests for ``pomata.indicators.fisher_transform`` — Ehlers' channel-normalized, tail-stretched momentum oscillator.

``fisher_transform`` is multi-input (``high`` / ``low``) and returns a single struct ``pl.Expr`` with the fields
``fisher`` / ``signal``. The local ``apply_fisher`` helper materializes each line over a two-column ``Float64`` frame
into a dict of lists, so the shared ``assert_matches`` and the naive ``fisher_transform_reference`` oracle (an
independent plain-Python rebuild of the channel, the ``0.33 / 0.67`` smoothing, the clamp, and the ``0.5`` log
recursion) compare line by line.

The transform is scale-INVARIANT (the channel normalization cancels any positive rescaling), so the large-magnitude
tier is vacuous and a scale-invariance tier takes its place; the flat-window ``0 / 0`` is pinned in the edge tier. The
hard output bound is ``ln(1999)`` -- the ``0.5`` recursion doubles the clamped single-step ``0.5 * ln(1999)`` to its
steady state -- asserted without clipping. The ladder is otherwise canonical: contract, edge (warm-up / null / NaN /
flat / clamp), correctness (oracle + golden), properties (reference agreement incl. missing data, bound, scale
invariance, the signal-is-lagged-fisher identity). Categories are split into classes; cross-cutting categories use
markers (see ``tests/README.md``).
"""

import math
from collections.abc import Sequence

import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st
from polars.testing import assert_frame_equal
from tests.indicators.oracles import fisher_transform_reference
from tests.support import (
    ABSOLUTE_TOLERANCE_PROPERTY,
    GROUP_KEY,
    HIGH,
    LOW,
    RELATIVE_TOLERANCE_PROPERTY,
    WINDOW_MAX,
    assert_matches,
    assert_scale_homogeneous,
    coherent_hl,
    coherent_hl_with_missing,
    materialize_struct,
    split_pairs,
)

from pomata.indicators import fisher_transform

FIELDS = ("fisher", "signal")

# The hard output bound: with the position clamped to +/-0.999 the single-step log term is at most ln(1999), and the
# ``0.5 * Fisher[t-1]`` recursion lifts the steady state to 0.5 * ln(1999) / (1 - 0.5) = ln(1999) (a geometric series,
# approached from below). Asserted directly, never used to clip the output.
FISHER_BOUND = math.log(1999.0)

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- derived, not chosen; the rationale is the shared method in CORRECTNESS.md, the numbers are this
# indicator's. To add an indicator, set its three facts here; the property tier below is then the same shape as every
# other indicator's.
#   1. warmup  W(window) = window - 1   (the rolling channel needs ``window`` medians, so the first defined position --
#              and the first Fisher value -- is at index ``window - 1``)
#   2. memory  M = 0: the oracle runs the identical recursion from the identical zero seed, so over the same input it
#              agrees from the first defined row; each example carries a window-plus of defined bars
#   3. domain  coherent_hl(): coherent (high >= low) positive-finite bars; the channel is non-degenerate except a flat
#              window (and window == 1, which is flat by construction) -- both give the 0/0, pinned in edge
# Windows span 1 .. WINDOW_MAX. Repetitions N are the shared CI profile (tests/conftest.py).
# ----------------------------------------------------------------------------------------------------------------------


@st.composite
def _cases[T](draw: st.DrawFn, bars: st.SearchStrategy[T]) -> tuple[list[T], int]:
    """
    A (series, window) pair sized from the facts above: length = the channel warm-up plus a window of defined bars, so
    every example has output on both lines to check.
    """
    window = draw(st.integers(min_value=1, max_value=WINDOW_MAX))
    defined = draw(st.integers(min_value=window, max_value=2 * window))
    length = window + defined
    return draw(st.lists(bars, min_size=length, max_size=length)), window


def apply_fisher(
    high: Sequence[float | None],
    low: Sequence[float | None],
    window: int,
) -> dict[str, list[float | None]]:
    """
    Materialize each line of ``fisher_transform`` over a two-column frame, as a dict mirroring the oracle's output.
    """
    return materialize_struct(
        {HIGH: high, LOW: low},
        fisher_transform(pl.col(HIGH), pl.col(LOW), window),
        FIELDS,
    )


class TestFisherTransformContract:
    """
    Type, struct schema, shape, and lazy/eager guarantees.
    """

    def test_returns_expr(self) -> None:
        """
        Verifies that the factory returns a ``pl.Expr`` without touching a frame.
        """
        assert isinstance(fisher_transform(pl.col(HIGH), pl.col(LOW), 3), pl.Expr)

    def test_output_is_struct_with_named_fields(self) -> None:
        """
        Verifies that the output is a ``Float64`` struct with exactly the fields ``fisher`` / ``signal``.
        """
        frame = pl.DataFrame({HIGH: [2.0, 4.0, 6.0], LOW: [1.0, 3.0, 4.0]})
        dtype = frame.select(fisher_transform(pl.col(HIGH), pl.col(LOW), 2).alias("ft")).schema["ft"]
        assert isinstance(dtype, pl.Struct)
        assert [field.name for field in dtype.fields] == ["fisher", "signal"]
        assert all(field.dtype == pl.Float64 for field in dtype.fields)

    def test_preserves_length(self) -> None:
        """
        Verifies that the output has one struct per input row.
        """
        frame = pl.DataFrame({HIGH: [2.0, 4.0, 6.0], LOW: [1.0, 3.0, 4.0]})
        assert frame.select(fisher_transform(pl.col(HIGH), pl.col(LOW), 2).alias("ft")).height == frame.height

    def test_lazy_eager_parity(self) -> None:
        """
        Verifies that eager and lazy application produce identical materialized output.
        """
        frame = pl.DataFrame({HIGH: [2.0, 4.0, 6.0, 5.0], LOW: [1.0, 3.0, 4.0, 4.0]})
        expr = fisher_transform(pl.col(HIGH), pl.col(LOW), 2).alias("ft")
        assert_frame_equal(frame.select(expr), frame.lazy().select(expr).collect())

    def test_over_partitions_independently(self) -> None:
        """
        Verifies that under ``.over`` the recurrence and channel do not span group boundaries: a group's Fisher line
        equals that group computed on its own.
        """
        frame = pl.DataFrame(
            {
                GROUP_KEY: ["a", "a", "a", "a", "b", "b", "b", "b"],
                HIGH: [2.0, 4.0, 6.0, 5.0, 12.0, 14.0, 16.0, 15.0],
                LOW: [1.0, 3.0, 4.0, 4.0, 11.0, 13.0, 14.0, 14.0],
            }
        )
        expr = fisher_transform(pl.col(HIGH), pl.col(LOW), 2).over(GROUP_KEY).struct.field("fisher")
        panel = frame.select(expr.alias("y"))["y"].to_list()
        standalone = apply_fisher([12.0, 14.0, 16.0, 15.0], [11.0, 13.0, 14.0, 14.0], 2)["fisher"]
        assert_matches(panel[4:], standalone)


class TestFisherTransformEdge:
    """
    Boundaries, warm-up, null / NaN, the flat window, and the clamp.
    """

    def test_window_below_one_raises(self) -> None:
        """
        Verifies that ``window < 1`` raises ``ValueError``.
        """
        with pytest.raises(ValueError, match="window must be >= 1"):
            fisher_transform(pl.col(HIGH), pl.col(LOW), 0)

    def test_empty(self) -> None:
        """
        Verifies that an empty input yields an empty output on both lines.
        """
        bands = apply_fisher([], [], 2)
        for field in FIELDS:
            assert_matches(bands[field], [])

    def test_all_null(self) -> None:
        """
        Verifies that an all-null input yields an all-null output on both lines.
        """
        bands = apply_fisher([None, None, None], [None, None, None], 2)
        for field in FIELDS:
            assert_matches(bands[field], [None, None, None])

    def test_warmup_null_count(self) -> None:
        """
        Verifies that ``fisher`` is null for the first ``window - 1`` rows and defined from the next, and that
        ``signal`` lags it by one further row.
        """
        high = [2.0, 4.0, 6.0, 5.0, 7.0]
        low = [1.0, 3.0, 4.0, 4.0, 5.0]
        bands = apply_fisher(high, low, 3)
        assert bands["fisher"][:2] == [None, None]
        assert bands["fisher"][2] is not None
        assert bands["signal"][:3] == [None, None, None]
        assert bands["signal"][3] is not None

    def test_flat_window_is_nan(self) -> None:
        """
        Verifies the flat window: a constant series has ``max == min`` over every window, so the channel normalization
        is the indeterminate ``0 / 0 == NaN`` from the first defined row onward.
        """
        flat = [10.0] * 6
        bands = apply_fisher(flat, flat, 3)
        assert_matches(bands["fisher"], [None, None, math.nan, math.nan, math.nan, math.nan])
        assert_matches(bands["signal"], [None, None, None, math.nan, math.nan, math.nan])

    def test_clamp_keeps_fisher_finite(self) -> None:
        """
        Verifies the clamp boundary: a sustained uptrend pins the position at ``+1`` so the smoothed value crosses
        ``0.999``; the clamp holds the log argument inside its domain, leaving every Fisher value finite and within the
        ``ln(1999)`` bound (it climbs toward that steady state, never reaching the ``+1`` singularity).
        """
        high = [float(i) for i in range(1, 16)]
        low = [value - 0.5 for value in high]
        bands = apply_fisher(high, low, 3)
        defined = [value for value in bands["fisher"] if value is not None]
        assert defined  # the uptrend produces output
        for value in defined:
            assert not math.isnan(value)
            assert abs(value) <= FISHER_BOUND + 1e-9

    def test_null_and_nan_bridge(self) -> None:
        """
        Verifies that a ``null`` / ``NaN`` flows through the channel and the double recursion exactly as the reference
        (a transient gap is bridged, not latched).
        """
        high = [2.0, 4.0, None, 8.0, math.nan, 12.0, 13.0, 15.0]
        low = [1.0, 3.0, 4.0, 6.0, 8.0, 10.0, 11.0, 13.0]
        bands = apply_fisher(high, low, 2)
        reference = fisher_transform_reference(high, low, 2)
        for field in FIELDS:
            assert_matches(bands[field], reference[field])


class TestFisherTransformCorrectness:
    """
    Against the naive reference oracle and frozen golden-master values.
    """

    def test_matches_reference(self) -> None:
        """
        Verifies that both lines agree with the naive reference across several windows.
        """
        high = [2.0, 4.0, 6.0, 5.0, 7.0, 9.0, 8.0, 11.0]
        low = [1.0, 3.0, 4.0, 4.0, 5.0, 7.0, 6.0, 9.0]
        for window in (2, 3, 4, 5):
            bands = apply_fisher(high, low, window)
            reference = fisher_transform_reference(high, low, window)
            for field in FIELDS:
                assert_matches(bands[field], reference[field])

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference, hand-derived for ``window = 2`` over medians ``[1, 3, 2]`` (high ``[2, 4, 3]``,
        low ``[0, 2, 1]``):
          - row 1: position ``+1`` -> smoothed ``0.33`` -> Fisher ``0.5 ln(1.33 / 0.67) = 0.342799``;
          - row 2: position ``-1`` -> smoothed ``-0.1089`` -> Fisher ``0.5 ln(0.8911/1.1089) + 0.5(0.3428) = 0.0621``.
        """
        bands = apply_fisher([2.0, 4.0, 3.0], [0.0, 2.0, 1.0], 2)
        assert_matches([None if v is None else round(v, 4) for v in bands["fisher"]], [None, 0.3428, 0.0621])
        assert_matches([None if v is None else round(v, 4) for v in bands["signal"]], [None, None, 0.3428])


class TestFisherTransformProperties:
    """
    Invariants that must hold for all inputs (property-based).
    """

    @given(case=_cases(coherent_hl()))
    def test_matches_reference_for_any_input(
        self,
        case: tuple[list[tuple[float, float]], int],
    ) -> None:
        """
        Verifies that, for any coherent high/low series and window, both lines match the naive reference.
        """
        rows, window = case
        high, low = split_pairs(rows)
        bands = apply_fisher(high, low, window)
        reference = fisher_transform_reference(high, low, window)
        for field in FIELDS:
            assert_matches(
                bands[field],
                reference[field],
                rel_tol=RELATIVE_TOLERANCE_PROPERTY,
                abs_tol=ABSOLUTE_TOLERANCE_PROPERTY,
            )

    @given(case=_cases(coherent_hl()))
    def test_fisher_is_bounded(
        self,
        case: tuple[list[tuple[float, float]], int],
    ) -> None:
        """
        Verifies the hard bound: wherever defined and finite, both lines satisfy ``|value| <= ln(1999)`` -- asserted on
        the raw output, never by clipping it.
        """
        rows, window = case
        high, low = split_pairs(rows)
        bands = apply_fisher(high, low, window)
        for field in FIELDS:
            for value in bands[field]:
                if value is not None and not math.isnan(value):
                    assert abs(value) <= FISHER_BOUND + 1e-9

    @given(case=_cases(coherent_hl()))
    def test_signal_is_lagged_fisher(
        self,
        case: tuple[list[tuple[float, float]], int],
    ) -> None:
        """
        Verifies the defining identity: ``signal`` is exactly ``fisher`` shifted one row (the trigger line).
        """
        rows, window = case
        high, low = split_pairs(rows)
        bands = apply_fisher(high, low, window)
        assert_matches(bands["signal"], [None, *bands["fisher"][:-1]])

    @given(
        case=_cases(coherent_hl()),
        exponent=st.sampled_from([-4, -3, -2, -1, 1, 2, 3, 4]),
    )
    def test_scale_invariance(
        self,
        case: tuple[list[tuple[float, float]], int],
        exponent: int,
    ) -> None:
        """
        Verifies that the transform is scale-invariant under a positive common rescaling of high / low (the channel
        normalization cancels the scale). ``k`` is a power of two so the rescaling is lossless and the match is exact.
        """
        k = 2.0**exponent
        rows, window = case
        high, low = split_pairs(rows)
        base = apply_fisher(high, low, window)
        scaled = apply_fisher([v * k for v in high], [v * k for v in low], window)
        for field in FIELDS:
            assert_scale_homogeneous(scaled[field], base[field], k=k, degree=0)

    @given(case=_cases(coherent_hl_with_missing()))
    def test_matches_reference_under_missing_data(
        self,
        case: tuple[list[tuple[float | None, float | None]], int],
    ) -> None:
        """
        Verifies that, for inputs freely mixing null / NaN / finite, both lines match the naive reference.
        """
        rows, window = case
        high, low = split_pairs(rows)
        bands = apply_fisher(high, low, window)
        reference = fisher_transform_reference(high, low, window)
        for field in FIELDS:
            assert_matches(
                bands[field],
                reference[field],
                rel_tol=RELATIVE_TOLERANCE_PROPERTY,
                abs_tol=ABSOLUTE_TOLERANCE_PROPERTY,
            )
