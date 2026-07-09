"""
Tests for ``pomata.indicators.keltner_channels`` — an EMA midline with ATR-scaled bands (the modern Raschke form).

``keltner_channels`` is multi-input (``high`` / ``low`` / ``close``) and returns a single struct ``pl.Expr`` with the
fields ``lower`` / ``middle`` / ``upper``. The local ``apply_keltner_channels`` helper materializes each field over a
three-column ``Float64`` frame into a dict of lists, so the shared ``assert_matches`` and the naive
``keltner_channels_reference`` oracle (a composition of the certified ``ema`` and ``atr`` references) compare band by
band.

The ladder is the canonical one: contract (type / struct schema / shape / lazy-eager / ``.over`` independence), edge
(warm-up / window collapse / single-row / null / NaN per band / flat series), correctness (vs the composed reference and
a frozen golden master), and properties (reference agreement incl. missing data, the public-API composition, band
symmetry, degree-1 scale-homogeneity, large-magnitude stability). The bands inherit the recursive ``ema`` / ``atr``
precision, so the property band is the recursive ``input_scale * EXACT_TOLERANCE_FACTOR``. Categories are split into
classes; cross-cutting categories use markers (see ``tests/README.md``).
"""

import math
from collections.abc import Sequence

import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st
from tests.indicators.oracles import keltner_channels_reference
from tests.support import (
    ABSOLUTE_TOLERANCE_EXACT,
    ABSOLUTE_TOLERANCE_REFERENCE,
    CLOSE,
    EXACT_TOLERANCE_FACTOR,
    GROUP_KEY,
    HIGH,
    LOW,
    RELATIVE_TOLERANCE_PROPERTY,
    RELATIVE_TOLERANCE_REFERENCE,
    RELATIVE_TOLERANCE_SCALE,
    WINDOW_MAX,
    assert_matches,
    assert_scale_homogeneous,
    coherent_hlc,
    coherent_hlc_with_missing,
    input_scale,
    materialize_struct,
    split_triples,
)

from pomata.indicators import atr, ema, keltner_channels

FIELDS = ("lower", "middle", "upper")

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- derived, not chosen; the rationale is the shared method in CORRECTNESS.md, the numbers are this
# indicator's. To add an indicator, set its three facts here; the property tier below is then the same shape as every
# other indicator's.
#   1. warmup  midline W(window) = window - 1; outer bands W = max(window, window_atr) - 1 (they also need the ATR)
#   2. memory  the oracle shares pomata's EMA / Wilder seeding, so the property holds from the first defined row
#              (M = 0); each example carries D in [maxw, 2 * maxw] defined bars (maxw = max(window, window_atr))
#   3. domain  coherent_hlc(): coherent (high >= low, low <= close <= high) positive-finite bars -- the ATR is only
#              non-negative on well-formed OHLC
# Windows span 1 .. WINDOW_MAX. Repetitions N are the shared CI profile (tests/conftest.py); override per-test only if
# its parameter space is larger.
# ----------------------------------------------------------------------------------------------------------------------


@st.composite
def _cases[T](draw: st.DrawFn, bars: st.SearchStrategy[T]) -> tuple[list[T], int, int]:
    """
    A (series, window, window_atr) triple sized from the facts above, length = the larger warm-up + a window of defined
    bars, so every example has output on every band to check.
    """
    window = draw(st.integers(min_value=1, max_value=WINDOW_MAX))
    window_atr = draw(st.integers(min_value=1, max_value=WINDOW_MAX))
    maxw = max(window, window_atr)
    defined = draw(st.integers(min_value=maxw, max_value=2 * maxw))
    length = (maxw - 1) + defined
    return draw(st.lists(bars, min_size=length, max_size=length)), window, window_atr


def apply_keltner_channels(
    high: Sequence[float | None],
    low: Sequence[float | None],
    close: Sequence[float | None],
    window: int,
    window_atr: int = 10,
    multiplier: float = 2.0,
) -> dict[str, list[float | None]]:
    """
    Materialize each band of ``keltner_channels`` over a three-column frame, as a dict mirroring the oracle's output.
    """
    return materialize_struct(
        {HIGH: high, LOW: low, CLOSE: close},
        keltner_channels(
            pl.col(HIGH), pl.col(LOW), pl.col(CLOSE), window=window, window_atr=window_atr, multiplier=multiplier
        ),
        FIELDS,
    )


class TestKeltnerChannelsContract:
    """
    Type, struct schema, shape, and lazy/eager guarantees.
    """

    def test_over_partitions_independently(self) -> None:
        """
        Verifies that under ``.over`` neither smoother spans group boundaries.
        """
        frame = pl.DataFrame(
            {
                GROUP_KEY: ["a", "a", "a", "b", "b", "b"],
                HIGH: [3.0, 4.0, 5.0, 13.0, 14.0, 15.0],
                LOW: [1.0, 2.0, 3.0, 11.0, 12.0, 13.0],
                CLOSE: [2.0, 3.0, 4.0, 12.0, 13.0, 14.0],
            }
        )
        middle = keltner_channels(pl.col(HIGH), pl.col(LOW), pl.col(CLOSE), window=2, window_atr=2).over(GROUP_KEY)
        result = frame.select(middle.struct.field("middle").alias("y"))["y"].to_list()
        # First row of each group is warm-up (window - 1 = 1 null); group b must not leak group a's EMA state.
        assert result[0] is None
        assert result[3] is None
        assert result[4] is not None
        assert result[4] > 10.0

    def test_output_is_struct_with_named_fields(self) -> None:
        """
        Verifies that the output is a ``Float64`` struct with exactly the fields ``lower`` / ``middle`` / ``upper``.
        """
        frame = pl.DataFrame({HIGH: [3.0, 4.0], LOW: [1.0, 2.0], CLOSE: [2.0, 3.0]})
        expr = keltner_channels(pl.col(HIGH), pl.col(LOW), pl.col(CLOSE), window=2, window_atr=2).alias("kc")
        dtype = frame.select(expr).schema["kc"]
        assert isinstance(dtype, pl.Struct)
        assert [field.name for field in dtype.fields] == ["lower", "middle", "upper"]
        assert all(field.dtype == pl.Float64 for field in dtype.fields)


class TestKeltnerChannelsEdge:
    """
    Boundaries, warm-up, validation, null / NaN per band, and the flat series.
    """

    def test_window_below_one_raises(self) -> None:
        """
        Verifies that ``window < 1`` raises ``ValueError``.
        """
        with pytest.raises(ValueError, match="window must be >= 1"):
            keltner_channels(pl.col(HIGH), pl.col(LOW), pl.col(CLOSE), window=0, window_atr=10)

    def test_window_atr_below_one_raises(self) -> None:
        """
        Verifies that ``window_atr < 1`` raises ``ValueError``.
        """
        with pytest.raises(ValueError, match="window_atr must be >= 1"):
            keltner_channels(pl.col(HIGH), pl.col(LOW), pl.col(CLOSE), window=3, window_atr=0)

    def test_invalid_multiplier_raises(self) -> None:
        """
        Verifies that a multiplier that is not a finite number ``> 0`` (zero, negative, ``NaN``, or ``±inf``) raises
        ``ValueError`` (a non-positive multiplier would silently swap or collapse the bands).
        """
        for invalid in (0.0, -1.0, math.nan, math.inf, -math.inf):
            with pytest.raises(ValueError, match="multiplier must be a finite number > 0"):
                keltner_channels(pl.col(HIGH), pl.col(LOW), pl.col(CLOSE), window=3, window_atr=10, multiplier=invalid)

    def test_single_row(self) -> None:
        """
        Verifies a one-element series: ``window == 1`` defines the midline; a larger window is all warm-up.
        """
        assert apply_keltner_channels([10.0], [8.0], [9.0], 1, window_atr=1)["middle"] == [9.0]
        assert apply_keltner_channels([10.0], [8.0], [9.0], 3, window_atr=3)["middle"] == [None]

    def test_all_null(self) -> None:
        """
        Verifies that an all-null input yields an all-null output on every band.
        """
        bands = apply_keltner_channels([None, None, None], [None, None, None], [None, None, None], 2, window_atr=2)
        for field in FIELDS:
            assert_matches(bands[field], [None, None, None])

    def test_null_bridged(self) -> None:
        """
        Verifies that an interior ``null`` in ``close`` is bridged by the recursive ``ema`` / ``atr`` legs: after the
        gap the midline recovers to a defined value, matching the reference.
        """
        high = [3.0, 4.0, 5.0, 6.0, 7.0]
        low = [1.0, 2.0, 3.0, 4.0, 5.0]
        close = [2.0, None, 4.0, 5.0, 6.0]
        bands = apply_keltner_channels(high, low, close, 2, window_atr=2)
        assert bands["middle"][-1] is not None
        reference = keltner_channels_reference(high, low, close, 2, 2, 2.0)
        for field in FIELDS:
            assert_matches(bands[field], reference[field])

    def test_nan_latches(self) -> None:
        """
        Verifies that a ``NaN`` in ``close`` propagates to every band through the recursive ``ema`` / ``atr`` legs,
        poisoning the midline forward (``null`` still takes precedence).
        """
        high = [3.0, 4.0, 5.0, 6.0]
        low = [1.0, 2.0, 3.0, 4.0]
        close = [2.0, math.nan, 4.0, 5.0]
        bands = apply_keltner_channels(high, low, close, 2, window_atr=2)
        assert_matches(bands["middle"], [None, math.nan, math.nan, math.nan])
        reference = keltner_channels_reference(high, low, close, 2, 2, 2.0)
        for field in ("lower", "upper"):
            assert_matches(bands[field], reference[field])

    def test_warmup_null_count(self) -> None:
        """
        Verifies that every band is null for the first ``window - 1`` rows and defined from the first full window.
        """
        bands = apply_keltner_channels(
            [3.0, 4.0, 5.0, 6.0, 7.0], [1.0, 2.0, 3.0, 4.0, 5.0], [2.0, 3.0, 4.0, 5.0, 6.0], 3, window_atr=3
        )
        for field in FIELDS:
            assert bands[field][:2] == [None, None]
            assert bands[field][2] is not None

    def test_window_exceeds_length(self) -> None:
        """
        Verifies that a window longer than the series yields an all-null result on every band.
        """
        bands = apply_keltner_channels([3.0, 4.0, 5.0], [1.0, 2.0, 3.0], [2.0, 3.0, 4.0], 5, window_atr=5)
        for field in FIELDS:
            assert_matches(bands[field], [None, None, None])

    def test_flat_series_collapses_to_ema(self) -> None:
        """
        Verifies the flat series: over a constant ``high == low == close`` run the ATR is ``0``, so all three bands
        collapse onto the EMA of ``close`` (here the constant itself).
        """
        flat = [10.0, 10.0, 10.0, 10.0, 10.0]
        bands = apply_keltner_channels(flat, flat, flat, 2, window_atr=2)
        expected = [None, 10.0, 10.0, 10.0, 10.0]
        for field in FIELDS:
            assert_matches(bands[field], expected)

    def test_missing_data_follows_the_legs(self) -> None:
        """
        Verifies that a missing value is handled by the recursive ``ema`` / ``atr`` legs (matching the composed
        reference), not by a channel-specific rule: a one-sided null ``high`` is absorbed by the true range (the
        low-close term survives), so the channel stays defined rather than nulling.
        """
        high = [10.0, None, 12.0, 13.0, 14.0]
        low = [8.0, 9.0, 10.0, 11.0, 12.0]
        close = [9.0, 10.0, 11.0, 12.0, 13.0]
        bands = apply_keltner_channels(high, low, close, 2, window_atr=2)
        reference = keltner_channels_reference(high, low, close, 2, 2, 2.0)
        for field in FIELDS:
            assert_matches(bands[field], reference[field])
        # Past the warm-up the channel stays defined despite the null high (the range absorbs the one-sided gap).
        for field in FIELDS:
            assert all(value is not None for value in bands[field][1:])


class TestKeltnerChannelsCorrectness:
    """
    Against the composed reference oracle and a frozen golden master.
    """

    def test_matches_reference(self) -> None:
        """
        Verifies agreement with the composed reference across several ``window`` / ``window_atr`` pairs.
        """
        high = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 7.5, 9.0]
        low = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 5.5, 7.0]
        close = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 6.5, 8.0]
        for window, window_atr in ((2, 2), (3, 2), (2, 3), (4, 4)):
            bands = apply_keltner_channels(high, low, close, window, window_atr=window_atr)
            reference = keltner_channels_reference(high, low, close, window, window_atr, 2.0)
            for field in FIELDS:
                assert_matches(
                    bands[field],
                    reference[field],
                    rel_tol=RELATIVE_TOLERANCE_PROPERTY,
                    abs_tol=input_scale(close) * EXACT_TOLERANCE_FACTOR,
                )

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference: bands(window=3, window_atr=3) over a non-flat five-bar OHLC series.
        """
        bands = apply_keltner_channels(
            [10.0, 12.0, 11.0, 13.0, 15.0], [8.0, 9.0, 9.5, 10.0, 12.0], [9.0, 11.0, 10.0, 12.0, 14.0], 3, window_atr=3
        )
        assert_matches(
            bands["lower"],
            [None, None, 5.666666666666667, 6.111111111111111, 7.2407407407407405],
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )
        assert_matches(
            bands["middle"],
            [None, None, 10.0, 11.0, 12.5],
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )
        assert_matches(
            bands["upper"],
            [None, None, 14.333333333333332, 15.88888888888889, 17.75925925925926],
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    def test_golden_master_flat(self) -> None:
        """
        Verifies the frozen reference on a flat series: bands(window=2, window_atr=2) == the EMA, the bands at zero ATR.
        """
        flat = [4.0, 4.0, 4.0, 4.0]
        bands = apply_keltner_channels(flat, flat, flat, 2, window_atr=2)
        for field in FIELDS:
            assert_matches(bands[field], [None, 4.0, 4.0, 4.0])

    def test_multiplier_scales_width(self) -> None:
        """
        Verifies the band half-width is linear in ``multiplier``: doubling it doubles the gap to the center.
        """
        high = [11.0, 12.0, 13.0, 12.0, 14.0]
        low = [9.0, 10.0, 11.0, 10.0, 12.0]
        close = [10.0, 11.0, 12.0, 11.0, 13.0]
        narrow = apply_keltner_channels(high, low, close, 3, window_atr=3, multiplier=1.0)
        wide = apply_keltner_channels(high, low, close, 3, window_atr=3, multiplier=2.0)
        for index in range(len(close)):
            center = narrow["middle"][index]
            if center is None:
                assert wide["upper"][index] is None
                continue
            narrow_upper = narrow["upper"][index]
            wide_upper = wide["upper"][index]
            assert narrow_upper is not None
            assert wide_upper is not None
            narrow_gap = narrow_upper - center
            wide_gap = wide_upper - center
            assert math.isclose(
                wide_gap, 2.0 * narrow_gap, rel_tol=RELATIVE_TOLERANCE_REFERENCE, abs_tol=ABSOLUTE_TOLERANCE_EXACT
            )


class TestKeltnerChannelsProperties:
    """
    Invariants that must hold for all inputs (property-based).
    """

    @given(case=_cases(coherent_hlc()))
    def test_matches_reference_for_any_input(
        self,
        case: tuple[list[tuple[float, float, float]], int, int],
    ) -> None:
        """
        Verifies that, for any coherent OHLC series and windows, every band matches the composed reference.
        """
        rows, window, window_atr = case
        high, low, close = split_triples(rows)
        bands = apply_keltner_channels(high, low, close, window, window_atr=window_atr)
        reference = keltner_channels_reference(high, low, close, window, window_atr, 2.0)
        for field in FIELDS:
            assert_matches(
                bands[field],
                reference[field],
                rel_tol=RELATIVE_TOLERANCE_PROPERTY,
                abs_tol=input_scale(close) * EXACT_TOLERANCE_FACTOR,
            )

    @given(case=_cases(coherent_hlc_with_missing()))
    def test_matches_reference_under_missing_data(
        self,
        case: tuple[list[tuple[float | None, float | None, float | None]], int, int],
    ) -> None:
        """
        Verifies that, for inputs freely mixing null / NaN / finite, every band matches the composed reference.
        """
        rows, window, window_atr = case
        high, low, close = split_triples(rows)
        bands = apply_keltner_channels(high, low, close, window, window_atr=window_atr)
        reference = keltner_channels_reference(high, low, close, window, window_atr, 2.0)
        for field in FIELDS:
            assert_matches(
                bands[field],
                reference[field],
                rel_tol=RELATIVE_TOLERANCE_PROPERTY,
                abs_tol=input_scale(close) * EXACT_TOLERANCE_FACTOR,
            )

    @given(
        case=_cases(coherent_hlc()),
        exponent=st.sampled_from([-4, -3, -2, -1, 1, 2, 3, 4]),
    )
    def test_scale_homogeneity(
        self,
        case: tuple[list[tuple[float, float, float]], int, int],
        exponent: int,
    ) -> None:
        """
        Verifies that ``keltner_channels`` is homogeneous of degree 1: scaling every input value by a constant ``k``
        scales the output by the same ``k`` -- ``keltner_channels(k * x) == k * keltner_channels(x)``. ``k`` is a
        power of two, so the rescale is exact and adds no floating-point error.
        """
        k = 2.0**exponent
        rows, window, window_atr = case
        high, low, close = split_triples(rows)
        base = apply_keltner_channels(high, low, close, window, window_atr=window_atr)
        scaled = apply_keltner_channels(
            [value * k for value in high],
            [value * k for value in low],
            [value * k for value in close],
            window,
            window_atr=window_atr,
        )
        for field in FIELDS:
            assert_scale_homogeneous(scaled[field], base[field], k=k, degree=1)

    @given(
        case=_cases(coherent_hlc()),
        scale=st.sampled_from([1e-6, 1e6, 1e9]),
    )
    def test_matches_reference_at_large_magnitude(
        self,
        case: tuple[list[tuple[float, float, float]], int, int],
        scale: float,
    ) -> None:
        """
        Verifies that at extreme positive magnitudes every band stays finite where the reference is and agrees.
        """
        rows, window, window_atr = case
        high = [row[0] * scale for row in rows]
        low = [row[1] * scale for row in rows]
        close = [row[2] * scale for row in rows]
        bands = apply_keltner_channels(high, low, close, window, window_atr=window_atr)
        reference = keltner_channels_reference(high, low, close, window, window_atr, 2.0)
        for field in FIELDS:
            assert_matches(
                bands[field],
                reference[field],
                rel_tol=RELATIVE_TOLERANCE_SCALE,
                abs_tol=input_scale(close) * EXACT_TOLERANCE_FACTOR,
            )

    @given(case=_cases(coherent_hlc()))
    def test_matches_public_api_composition(
        self,
        case: tuple[list[tuple[float, float, float]], int, int],
    ) -> None:
        """
        Verifies that the bands are exactly the public ``ema`` / ``atr`` composition: a second witness through the
        certified legs rather than the naive references.
        """
        rows, window, window_atr = case
        high, low, close = split_triples(rows)
        frame = pl.DataFrame(
            {
                HIGH: pl.Series(HIGH, high, dtype=pl.Float64),
                LOW: pl.Series(LOW, low, dtype=pl.Float64),
                CLOSE: pl.Series(CLOSE, close, dtype=pl.Float64),
            }
        )
        midline = ema(pl.col(CLOSE), window)
        half = 2.0 * atr(pl.col(HIGH), pl.col(LOW), pl.col(CLOSE), window_atr)
        expected = frame.select(
            midline.alias("middle"), (midline - half).alias("lower"), (midline + half).alias("upper")
        )
        bands = apply_keltner_channels(high, low, close, window, window_atr=window_atr)
        for field in FIELDS:
            assert_matches(bands[field], expected[field].to_list())
