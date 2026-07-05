"""
Tests for ``pomata.indicators.donchian_channels`` — the window's highest high and lowest low, with their midline.

``donchian_channels`` is multi-input (``high`` / ``low``) and returns a single struct ``pl.Expr`` with the fields
``lower`` / ``middle`` / ``upper``. The local ``apply_donchian_channels`` helper materializes each field over a
two-column ``Float64`` frame into a dict of lists, so the shared ``assert_matches`` and the naive
``donchian_channels_reference`` oracle (which returns the matching dict) compare band by band.

The ladder is the canonical one: contract (type / struct schema / shape / lazy-eager / ``.over`` independence), edge
(warm-up / window collapse / single-row / null / NaN / flat / malformed bar), correctness (vs the closed-form reference
and a frozen golden master), and properties (reference agreement incl. missing data, the band ordering, window-growth
monotonicity, degree-1 scale-homogeneity, and large-magnitude stability). The bands are window extremes and their mean
-- an exact transform -- so a fixed reference band applies (not an ``input_scale``-sized one). Categories are split into
classes; cross-cutting categories use markers (see ``tests/README.md``).
"""

import math
from collections.abc import Sequence

import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st
from tests.indicators.oracles import donchian_channels_reference
from tests.support import (
    ABSOLUTE_TOLERANCE_REFERENCE,
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

from pomata.indicators import donchian_channels

FIELDS = ("lower", "middle", "upper")

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- derived, not chosen; the rationale is the shared method in CORRECTNESS.md, the numbers are this
# indicator's. To add an indicator, set its three facts here; the property tier below is then the same shape as every
# other indicator's.
#   1. warmup  W(window) = window - 1   (every band is null until the window holds ``window`` non-null values)
#   2. memory  the oracle is windowed like pomata, so the property holds from the first defined row (M = 0); each
#              example carries D in [window, 2 * window] defined bars -- one window of output, never all warm-up
#   3. domain  coherent (high >= low) positive-finite bars over the test's regime; the bands take the window max of
#              ``high`` and min of ``low`` (no squaring), so no subnormal-square floor is needed
# Windows span 1 .. WINDOW_MAX. Repetitions N are the shared CI profile (tests/conftest.py); override per-test only if
# its parameter space is larger.
# ----------------------------------------------------------------------------------------------------------------------


@st.composite
def _cases[T](draw: st.DrawFn, bars: st.SearchStrategy[T]) -> tuple[list[T], int]:
    """
    A (series, window) pair sized from the facts above: ``window`` over its regimes, length = warm-up + a window of
    defined bars, so every example has output to check (never an all-warm-up series, the waste a ``window`` decoupled
    from the length would cause).
    """
    window = draw(st.integers(min_value=1, max_value=WINDOW_MAX))
    defined = draw(st.integers(min_value=window, max_value=2 * window))
    length = (window - 1) + defined
    return draw(st.lists(bars, min_size=length, max_size=length)), window


def apply_donchian_channels(
    high: Sequence[float | None],
    low: Sequence[float | None],
    window: int,
) -> dict[str, list[float | None]]:
    """
    Materialize each band of ``donchian_channels`` over a two-column frame, as a dict mirroring the oracle's output.
    """
    return materialize_struct(
        {HIGH: high, LOW: low},
        donchian_channels(pl.col(HIGH), pl.col(LOW), window),
        FIELDS,
    )


class TestDonchianChannelsContract:
    """
    Type, struct schema, shape, and lazy/eager guarantees.
    """

    def test_output_is_struct_with_named_fields(self) -> None:
        """
        Verifies that the output is a ``Float64`` struct with exactly the fields ``lower`` / ``middle`` / ``upper``.
        """
        frame = pl.DataFrame({HIGH: [11.0, 12.0, 13.0], LOW: [9.0, 10.0, 11.0]})
        dtype = frame.select(donchian_channels(pl.col(HIGH), pl.col(LOW), 2).alias("dc")).schema["dc"]
        assert isinstance(dtype, pl.Struct)
        assert [field.name for field in dtype.fields] == ["lower", "middle", "upper"]
        assert all(field.dtype == pl.Float64 for field in dtype.fields)

    def test_over_partitions_independently(self) -> None:
        """
        Verifies that under ``.over`` the window resets per group and never spans group boundaries.
        """
        frame = pl.DataFrame(
            {
                GROUP_KEY: ["a", "a", "a", "b", "b", "b"],
                HIGH: [11.0, 12.0, 13.0, 21.0, 22.0, 23.0],
                LOW: [9.0, 10.0, 11.0, 19.0, 20.0, 21.0],
            }
        )
        upper = donchian_channels(pl.col(HIGH), pl.col(LOW), 2).over(GROUP_KEY).struct.field("upper")
        result = frame.select(upper.alias("y"))["y"].to_list()
        assert_matches(result, [None, 12.0, 13.0, None, 22.0, 23.0])


class TestDonchianChannelsEdge:
    """
    Boundaries, warm-up, window collapse, null / NaN, and the flat and malformed bars.
    """

    def test_window_below_one_raises(self) -> None:
        """
        Verifies that ``window < 1`` raises ``ValueError``.
        """
        with pytest.raises(ValueError, match="window must be >= 1"):
            donchian_channels(pl.col(HIGH), pl.col(LOW), 0)

    def test_all_null(self) -> None:
        """
        Verifies that an all-null input yields an all-null output on every band.
        """
        bands = apply_donchian_channels([None, None, None], [None, None, None], 2)
        for field in FIELDS:
            assert_matches(bands[field], [None, None, None])

    def test_warmup_null_count(self) -> None:
        """
        Verifies that every band is null for the first ``window - 1`` rows and defined from the first full window.
        """
        bands = apply_donchian_channels([11.0, 12.0, 13.0, 14.0, 15.0], [9.0, 10.0, 11.0, 12.0, 13.0], 3)
        for field in FIELDS:
            assert bands[field][:2] == [None, None]
            assert bands[field][2] is not None

    def test_window_one_is_bar_extremes(self) -> None:
        """
        Verifies that ``window == 1`` gives the bar's own ``high`` / ``low`` and their per-bar mean.
        """
        bands = apply_donchian_channels([11.0, 12.0, 13.0], [9.0, 10.0, 11.0], 1)
        assert_matches(bands["upper"], [11.0, 12.0, 13.0])
        assert_matches(bands["lower"], [9.0, 10.0, 11.0])
        assert_matches(bands["middle"], [10.0, 11.0, 12.0])

    def test_single_row(self) -> None:
        """
        Verifies a one-element series: ``window == 1`` gives the bar's extremes, a larger window is all warm-up.
        """
        for field, value in zip(FIELDS, (9.0, 10.0, 11.0), strict=True):
            assert_matches(apply_donchian_channels([11.0], [9.0], 1)[field], [value])
            assert_matches(apply_donchian_channels([11.0], [9.0], 3)[field], [None])

    def test_window_exceeds_length(self) -> None:
        """
        Verifies that a window longer than the series yields an all-null result on every band.
        """
        bands = apply_donchian_channels([11.0, 12.0, 13.0], [9.0, 10.0, 11.0], 5)
        for field in FIELDS:
            assert_matches(bands[field], [None, None, None])

    def test_null_propagates_per_band(self) -> None:
        """
        Verifies that ``null`` propagates per band: a ``null`` in ``high`` nulls ``upper`` and ``middle`` while
        ``lower`` (reading the intact ``low``) stays defined.
        """
        bands = apply_donchian_channels([11.0, None, 13.0, 14.0], [9.0, 10.0, 11.0, 12.0], 2)
        assert_matches(bands["upper"], [None, None, None, 14.0])
        assert_matches(bands["lower"], [None, 9.0, 10.0, 11.0])
        assert_matches(bands["middle"], [None, None, None, 12.5])

    def test_nan_propagates_per_band(self) -> None:
        """
        Verifies that ``NaN`` propagates per band: a ``NaN`` in ``high`` makes ``upper`` and ``middle`` ``NaN`` while
        ``lower`` (reading the intact ``low``) stays finite.
        """
        bands = apply_donchian_channels([11.0, math.nan, 13.0, 14.0], [9.0, 10.0, 11.0, 12.0], 2)
        assert_matches(bands["upper"], [None, math.nan, math.nan, 14.0])
        assert_matches(bands["lower"], [None, 9.0, 10.0, 11.0])
        assert_matches(bands["middle"], [None, math.nan, math.nan, 12.5])

    def test_flat_window_collapses(self) -> None:
        """
        Verifies the flat window: where ``high`` and ``low`` hold one repeated value, all three bands equal it.
        """
        bands = apply_donchian_channels([5.0, 5.0, 5.0, 5.0], [5.0, 5.0, 5.0, 5.0], 2)
        for field in FIELDS:
            assert_matches(bands[field], [None, 5.0, 5.0, 5.0])

    def test_malformed_high_below_low_is_not_reordered(self) -> None:
        """
        Verifies that a malformed bar (``high < low``) flows through unchanged: the upper band sits below the lower band
        rather than being silently reordered.
        """
        bands = apply_donchian_channels([9.0, 9.0, 9.0], [11.0, 11.0, 11.0], 2)
        assert_matches(bands["upper"], [None, 9.0, 9.0])
        assert_matches(bands["lower"], [None, 11.0, 11.0])
        assert_matches(bands["middle"], [None, 10.0, 10.0])


class TestDonchianChannelsCorrectness:
    """
    Against the naive reference oracle and frozen golden-master values.
    """

    def test_matches_reference(self) -> None:
        """
        Verifies that every band agrees with the naive closed-form reference across several windows.
        """
        high = [11.0, 12.0, 13.0, 12.5, 14.0, 15.0, 14.5, 16.0]
        low = [9.0, 10.0, 11.0, 11.0, 12.0, 13.0, 12.5, 14.0]
        for window in (1, 2, 3, 4, 5):
            bands = apply_donchian_channels(high, low, window)
            reference = donchian_channels_reference(high, low, window)
            for field in FIELDS:
                assert_matches(bands[field], reference[field])

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference: channels(window=3) over the sample bars.
        """
        bands = apply_donchian_channels([11.0, 12.0, 13.0, 12.5, 14.0], [9.0, 10.0, 11.0, 11.0, 12.0], 3)
        assert_matches(bands["upper"], [None, None, 13.0, 13.0, 14.0])
        assert_matches(bands["lower"], [None, None, 9.0, 10.0, 11.0])
        assert_matches(bands["middle"], [None, None, 11.0, 11.5, 12.5])


class TestDonchianChannelsProperties:
    """
    Invariants that must hold for all inputs (property-based).
    """

    # NOTE: exact transform -- implementation and oracle compute identical arithmetic, residual is zero, so a fixed
    # reference band applies here (not input_scale-sized like the sum-based degree-1 kernels).
    @given(case=_cases(coherent_hl()))
    def test_matches_reference_for_any_input(
        self,
        case: tuple[list[tuple[float, float]], int],
    ) -> None:
        """
        Verifies that, for any aligned high/low series and window, every band matches the naive reference.
        """
        rows, window = case
        high, low = split_pairs(rows)
        bands = apply_donchian_channels(high, low, window)
        reference = donchian_channels_reference(high, low, window)
        for field in FIELDS:
            assert_matches(
                bands[field],
                reference[field],
                rel_tol=RELATIVE_TOLERANCE_PROPERTY,
                abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
            )

    @given(case=_cases(coherent_hl()))
    def test_bands_are_ordered(
        self,
        case: tuple[list[tuple[float, float]], int],
    ) -> None:
        """
        Verifies the true invariant on coherent bars: ``lower <= middle <= upper`` wherever defined (no clip needed --
        for ``high >= low`` the window max of ``high`` cannot fall below the window min of ``low``).
        """
        rows, window = case
        high, low = split_pairs(rows)
        bands = apply_donchian_channels(high, low, window)
        for lower, middle, upper in zip(bands["lower"], bands["middle"], bands["upper"], strict=True):
            if lower is None:
                assert middle is None
                assert upper is None
            else:
                assert middle is not None
                assert upper is not None
                assert lower <= middle <= upper

    @given(case=_cases(coherent_hl()))
    def test_monotone_under_window_growth(
        self,
        case: tuple[list[tuple[float, float]], int],
    ) -> None:
        """
        Verifies that widening the window cannot narrow the channel: a longer window admits more bars, so ``upper`` can
        only rise and ``lower`` can only fall, wherever both windows are defined.
        """
        rows, window = case
        high, low = split_pairs(rows)
        narrow = apply_donchian_channels(high, low, window)
        wide = apply_donchian_channels(high, low, window + 1)
        for index in range(len(rows)):
            narrow_upper = narrow["upper"][index]
            wide_upper = wide["upper"][index]
            if narrow_upper is None or wide_upper is None:
                continue
            narrow_lower = narrow["lower"][index]
            wide_lower = wide["lower"][index]
            assert narrow_lower is not None
            assert wide_lower is not None
            assert wide_upper >= narrow_upper
            assert wide_lower <= narrow_lower

    @given(
        case=_cases(coherent_hl()),
        exponent=st.sampled_from([-4, -3, -2, -1, 1, 2, 3, 4]),
    )
    def test_scale_homogeneity(
        self,
        case: tuple[list[tuple[float, float]], int],
        exponent: int,
    ) -> None:
        """
        Verifies that, for positive ``k``, every band is homogeneous of degree 1: ``band(k * h, k * l) == k * band``.
        ``k`` is a power of two so the rescaling is lossless and cannot perturb the windowed extremes.
        """
        k = 2.0**exponent
        rows, window = case
        high, low = split_pairs(rows)
        base = apply_donchian_channels(high, low, window)
        scaled = apply_donchian_channels([value * k for value in high], [value * k for value in low], window)
        for field in FIELDS:
            assert_scale_homogeneous(scaled[field], base[field], k=k, degree=1)

    @given(case=_cases(coherent_hl_with_missing()))
    def test_matches_reference_under_missing_data(
        self,
        case: tuple[list[tuple[float | None, float | None]], int],
    ) -> None:
        """
        Verifies that, for inputs freely mixing null / NaN / finite, every band matches the naive reference.
        """
        rows, window = case
        high, low = split_pairs(rows)
        bands = apply_donchian_channels(high, low, window)
        reference = donchian_channels_reference(high, low, window)
        for field in FIELDS:
            assert_matches(
                bands[field],
                reference[field],
                rel_tol=RELATIVE_TOLERANCE_PROPERTY,
                abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
            )

    @given(
        case=_cases(coherent_hl()),
        scale=st.sampled_from([1e-6, 1e6, 1e9]),
    )
    def test_matches_reference_at_large_magnitude(
        self,
        case: tuple[list[tuple[float, float]], int],
        scale: float,
    ) -> None:
        """
        Verifies that at extreme positive magnitudes every band stays finite where the reference is and agrees.
        """
        rows, window = case
        high = [row[0] * scale for row in rows]
        low = [row[1] * scale for row in rows]
        bands = apply_donchian_channels(high, low, window)
        reference = donchian_channels_reference(high, low, window)
        for field in FIELDS:
            assert_matches(
                bands[field],
                reference[field],
                rel_tol=RELATIVE_TOLERANCE_PROPERTY,
                abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
            )
