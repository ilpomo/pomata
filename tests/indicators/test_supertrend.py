"""
Tests for ``pomata.indicators.supertrend`` — Seban's ATR-band trailing stop with a flip-driven trend direction.

``supertrend`` is multi-input (``high`` / ``low`` / ``close``) and returns a single struct ``pl.Expr`` with the fields
``line`` / ``direction``. The local ``apply_supertrend`` helper materializes each field over a three-column ``Float64``
frame into a dict of lists, so the shared ``assert_matches`` and the naive ``supertrend_reference`` oracle compare field
by field. The oracle is deliberately NOT a structural mirror -- it computes the final-band arrays in one pass and runs a
separate flip pass over them, where the implementation is a single interleaved state machine -- so agreement is evidence
rather than a shared shape; two hand-computed goldens (both seed directions, a non-default multiplier) anchor it.

``line`` is scale-dependent (degree 1, an ATR-scaled price level) so the large-magnitude tier applies; ``direction``
is scale-invariant (a flag from like-scaled crossings). The ladder is the canonical one: contract (type / struct schema
/ shape / lazy-eager / ``.over`` independence), edge (warm-up / validation / single-row / null / NaN bridge / flat),
correctness (vs the non-mirror reference and frozen goldens), and properties (reference agreement incl. missing data and
large magnitude, the trend-side band invariant, the ratchet hysteresis, scale behavior). The band inherits the recursive
ATR precision, so the property band is the recursive ``input_scale * EXACT_TOLERANCE_FACTOR``. Categories are split into
classes; cross-cutting categories use markers (see ``tests/README.md``).
"""

import math
from collections.abc import Sequence

import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st
from tests.indicators.oracles import supertrend_reference
from tests.support import (
    CLOSE,
    EXACT_TOLERANCE_FACTOR,
    GROUP_KEY,
    HIGH,
    LOW,
    RELATIVE_TOLERANCE_PROPERTY,
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

from pomata.indicators import supertrend

FIELDS = ("line", "direction")

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- derived, not chosen; the rationale is the shared method in CORRECTNESS.md, the numbers are this
# indicator's. To add an indicator, set its three facts here; the property tier below is then the same shape as every
# other indicator's.
#   1. warmup  W(window) = window - 1   (the ATR needs ``window`` true ranges, so the band -- and the recurrence -- is
#              first defined at index ``window - 1``)
#   2. memory  M = 0: the oracle builds its bands on the same Wilder-seeded ATR, so over the same input it agrees from
#              the first defined row; each example carries a window-plus of defined bars
#   3. domain  coherent_hlc(): coherent (high >= low, low <= close <= high) positive-finite bars -- the ATR is only
#              non-negative on well-formed OHLC; a flat window (ATR = 0, bands on the midpoint) is pinned in edge
# Windows span 1 .. WINDOW_MAX. Repetitions N are the shared CI profile (tests/conftest.py).
# ----------------------------------------------------------------------------------------------------------------------


@st.composite
def _cases[T](draw: st.DrawFn, bars: st.SearchStrategy[T]) -> tuple[list[T], int]:
    """
    A (series, window) pair sized from the facts above: length = the ATR warm-up plus a window of defined bars, so every
    example has output on both fields to check.
    """
    window = draw(st.integers(min_value=1, max_value=WINDOW_MAX))
    defined = draw(st.integers(min_value=window, max_value=2 * window))
    length = (window - 1) + defined
    return draw(st.lists(bars, min_size=length, max_size=length)), window


def apply_supertrend(
    high: Sequence[float | None],
    low: Sequence[float | None],
    close: Sequence[float | None],
    window: int,
    multiplier: float = 3.0,
) -> dict[str, list[float | None]]:
    """
    Materialize each field of ``supertrend`` over a three-column frame, as a dict mirroring the oracle's output.
    """
    return materialize_struct(
        {HIGH: high, LOW: low, CLOSE: close},
        supertrend(pl.col(HIGH), pl.col(LOW), pl.col(CLOSE), window, multiplier=multiplier),
        FIELDS,
    )


class TestSupertrendContract:
    """
    Type, struct schema, shape, and lazy/eager guarantees.
    """

    def test_output_is_struct_with_named_fields(self) -> None:
        """
        Verifies that the output is a ``Float64`` struct with exactly the fields ``line`` / ``direction``.
        """
        frame = pl.DataFrame({HIGH: [3.0, 4.0, 5.0], LOW: [1.0, 2.0, 3.0], CLOSE: [2.0, 3.0, 4.0]})
        dtype = frame.select(supertrend(pl.col(HIGH), pl.col(LOW), pl.col(CLOSE), 2).alias("s")).schema["s"]
        assert isinstance(dtype, pl.Struct)
        assert [field.name for field in dtype.fields] == ["line", "direction"]
        assert all(field.dtype == pl.Float64 for field in dtype.fields)

    def test_over_partitions_independently(self) -> None:
        """
        Verifies that under ``.over`` the recurrence and ATR do not span group boundaries: each group equals that group
        computed on its own.
        """
        frame = pl.DataFrame(
            {
                GROUP_KEY: ["a", "a", "a", "a", "b", "b", "b", "b"],
                HIGH: [10.0, 11.0, 12.0, 11.0, 20.0, 21.0, 22.0, 21.0],
                LOW: [9.0, 10.0, 11.0, 10.0, 19.0, 20.0, 21.0, 20.0],
                CLOSE: [9.5, 10.8, 11.8, 10.2, 19.5, 20.8, 21.8, 20.2],
            }
        )
        trend = supertrend(pl.col(HIGH), pl.col(LOW), pl.col(CLOSE), 2).over(GROUP_KEY)
        grouped = {field: frame.select(trend.struct.field(field).alias(field))[field].to_list() for field in FIELDS}
        group_a = apply_supertrend([10.0, 11.0, 12.0, 11.0], [9.0, 10.0, 11.0, 10.0], [9.5, 10.8, 11.8, 10.2], 2)
        group_b = apply_supertrend([20.0, 21.0, 22.0, 21.0], [19.0, 20.0, 21.0, 20.0], [19.5, 20.8, 21.8, 20.2], 2)
        for field in FIELDS:
            assert_matches(grouped[field], group_a[field] + group_b[field])


class TestSupertrendEdge:
    """
    Boundaries, warm-up, validation, single-row, null / NaN bridging, and the flat series.
    """

    def test_window_below_one_raises(self) -> None:
        """
        Verifies that ``window < 1`` raises ``ValueError``.
        """
        with pytest.raises(ValueError, match="window must be >= 1"):
            supertrend(pl.col(HIGH), pl.col(LOW), pl.col(CLOSE), 0)

    def test_invalid_multiplier_raises(self) -> None:
        """
        Verifies that a multiplier that is not a finite number ``> 0`` (zero, negative, ``NaN``, or ``±inf``) raises
        ``ValueError`` (it would collapse or invert the bands).
        """
        for invalid in (0.0, -1.0, math.nan, math.inf, -math.inf):
            with pytest.raises(ValueError, match="multiplier must be a finite number > 0"):
                supertrend(pl.col(HIGH), pl.col(LOW), pl.col(CLOSE), 3, multiplier=invalid)

    def test_all_null(self) -> None:
        """
        Verifies that an all-null input yields an all-null output on both fields.
        """
        bands = apply_supertrend([None, None, None], [None, None, None], [None, None, None], 2)
        for field in FIELDS:
            assert_matches(bands[field], [None, None, None])

    def test_warmup_null_count(self) -> None:
        """
        Verifies that both fields are null for the first ``window - 1`` rows (the ATR warm-up), defined from the next.
        """
        high = [10.0, 11.0, 12.0, 11.0, 13.0]
        low = [9.0, 10.0, 11.0, 10.0, 12.0]
        close = [9.5, 10.8, 11.8, 10.2, 12.8]
        bands = apply_supertrend(high, low, close, 3)
        for field in FIELDS:
            assert bands[field][:2] == [None, None]
            assert bands[field][2] is not None

    def test_single_row(self) -> None:
        """
        Verifies a one-element series: ``window == 1`` defines the bar (ATR = the range); a larger window is all
        warm-up.
        """
        assert apply_supertrend([10.0], [8.0], [9.0], 1)["direction"] == [1.0]
        assert apply_supertrend([10.0], [8.0], [9.0], 3)["direction"] == [None]

    def test_flat_series(self) -> None:
        """
        Verifies the flat series: a constant ``high == low == close`` run has zero ATR, so both bands collapse onto the
        midpoint; the seed reads ``close == lower`` as the bearish side and the line tracks the midpoint with direction
        ``-1`` (a flip needs a strict cross, which a flat series never makes).
        """
        flat = [5.0] * 5
        bands = apply_supertrend(flat, flat, flat, 2)
        assert_matches(bands["line"], [None, 5.0, 5.0, 5.0, 5.0])
        assert_matches(bands["direction"], [None, -1.0, -1.0, -1.0, -1.0])

    def test_null_and_nan_bridge(self) -> None:
        """
        Verifies that a ``null`` / ``NaN`` flows through the ATR and the recurrence as the reference (a transient gap is
        bridged by the running state and the last finite close, not latched).
        """
        high = [10.0, 11.0, None, 13.0, math.nan, 12.0, 14.0, 13.0]
        low = [9.0, 10.0, 11.0, 12.0, 13.0, 11.0, 13.0, 12.0]
        close = [9.5, 10.8, 11.0, 12.5, 13.5, 11.2, 13.8, 12.2]
        bands = apply_supertrend(high, low, close, 2)
        reference = supertrend_reference(high, low, close, 2, 3.0)
        for field in FIELDS:
            assert_matches(bands[field], reference[field])


class TestSupertrendCorrectness:
    """
    Against the non-mirror reference oracle and frozen, hand-derived golden masters.
    """

    def test_matches_reference(self) -> None:
        """
        Verifies that both fields agree with the non-mirror reference across several windows and multipliers.
        """
        high = [10.0, 11.0, 12.0, 13.0, 14.0, 13.0, 12.0, 11.0]
        low = [9.0, 10.0, 11.0, 12.0, 13.0, 12.0, 11.0, 10.0]
        close = [9.5, 10.5, 11.5, 12.5, 13.5, 12.0, 11.0, 10.2]
        for window, multiplier in ((1, 3.0), (2, 2.0), (3, 3.0), (4, 1.5)):
            bands = apply_supertrend(high, low, close, window, multiplier=multiplier)
            reference = supertrend_reference(high, low, close, window, multiplier)
            for field in FIELDS:
                assert_matches(
                    bands[field],
                    reference[field],
                    rel_tol=RELATIVE_TOLERANCE_PROPERTY,
                    abs_tol=input_scale(close) * EXACT_TOLERANCE_FACTOR,
                )

    def test_golden_master_uptrend_then_flip(self) -> None:
        """
        Verifies the frozen reference, hand-derived for ``window = 3``, ``multiplier = 2.0`` over a rising series that
        pulls back into a flip: the line ratchets up while ``direction == +1`` (holding at ``10.6481`` through the
        pullback), then flips to the upper band ``12.9005`` with ``direction == -1`` when the close breaks below it.
        """
        high = [10.0, 11.0, 12.0, 13.0, 14.0, 13.0, 12.0, 11.0]
        low = [9.0, 10.0, 11.0, 12.0, 13.0, 12.0, 11.0, 10.0]
        close = [9.5, 10.5, 11.5, 12.5, 13.5, 12.0, 11.0, 10.2]
        bands = apply_supertrend(high, low, close, 3, multiplier=2.0)
        assert_matches(
            [None if v is None else round(v, 4) for v in bands["line"]],
            [None, None, 8.8333, 9.7222, 10.6481, 10.6481, 10.6481, 12.9005],
        )
        assert_matches(bands["direction"], [None, None, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0])

    def test_golden_master_downtrend_seed_nondefault_multiplier(self) -> None:
        """
        Verifies the frozen reference, hand-derived for ``window = 2``, ``multiplier = 1.0`` (non-default) over a series
        that seeds long, flips short on the next bar as the close breaks the lower band, then flips back long: every
        branch -- seed, ``+1 -> -1``, ``-1 -> +1`` -- is anchored deterministically (the oracle is not trusted to be
        independent here).
        """
        high = [20.0, 19.0, 18.0, 17.0, 18.0, 19.0, 20.0, 21.0]
        low = [19.0, 18.0, 17.0, 16.0, 17.0, 18.0, 19.0, 20.0]
        close = [19.2, 18.2, 17.2, 16.2, 17.8, 18.8, 19.8, 20.8]
        bands = apply_supertrend(high, low, close, 2, multiplier=1.0)
        assert_matches(
            [None if v is None else round(v, 4) for v in bands["line"]],
            [None, 17.4, 18.65, 17.675, 16.0125, 17.1562, 18.2281, 19.2641],
        )
        assert_matches(bands["direction"], [None, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0])


class TestSupertrendProperties:
    """
    Invariants that must hold for all inputs (property-based).
    """

    @given(case=_cases(coherent_hlc()))
    def test_matches_reference_for_any_input(
        self,
        case: tuple[list[tuple[float, float, float]], int],
    ) -> None:
        """
        Verifies that, for any coherent OHLC series and window, both fields match the non-mirror reference.
        """
        rows, window = case
        high, low, close = split_triples(rows)
        bands = apply_supertrend(high, low, close, window)
        reference = supertrend_reference(high, low, close, window, 3.0)
        for field in FIELDS:
            assert_matches(
                bands[field],
                reference[field],
                rel_tol=RELATIVE_TOLERANCE_PROPERTY,
                abs_tol=input_scale(close) * EXACT_TOLERANCE_FACTOR,
            )

    @given(case=_cases(coherent_hlc()))
    def test_line_sits_on_the_trend_side_of_price(
        self,
        case: tuple[list[tuple[float, float, float]], int],
    ) -> None:
        """
        Verifies the defining band invariant: in an up-trend (``direction == +1``) the line sits at or below the close,
        in a down-trend (``-1``) at or above it -- a true property of the definition, asserted on the raw output.
        """
        rows, window = case
        high, low, close = split_triples(rows)
        bands = apply_supertrend(high, low, close, window)
        margin = input_scale(close) * EXACT_TOLERANCE_FACTOR
        for line_value, direction_value, close_value in zip(bands["line"], bands["direction"], close, strict=True):
            if line_value is None or math.isnan(line_value):
                continue
            assert close_value is not None
            if direction_value == 1.0:
                assert line_value <= close_value + margin
            else:
                assert line_value >= close_value - margin

    @given(case=_cases(coherent_hlc()))
    def test_line_ratchets_within_a_trend(
        self,
        case: tuple[list[tuple[float, float, float]], int],
    ) -> None:
        """
        Verifies the hysteresis: across consecutive defined bars that keep the same direction the line is monotone --
        non-decreasing while ``+1`` (lower band only rises), non-increasing while ``-1`` (upper band only falls).
        """
        rows, window = case
        high, low, close = split_triples(rows)
        bands = apply_supertrend(high, low, close, window)
        margin = input_scale(close) * EXACT_TOLERANCE_FACTOR
        previous_line: float | None = None
        previous_direction: float | None = None
        for line_value, direction_value in zip(bands["line"], bands["direction"], strict=True):
            if line_value is None or math.isnan(line_value):
                previous_line = None
                previous_direction = None
                continue
            if previous_line is not None and direction_value == previous_direction:
                if direction_value == 1.0:
                    assert line_value >= previous_line - margin
                else:
                    assert line_value <= previous_line + margin
            previous_line = line_value
            previous_direction = direction_value

    @given(case=_cases(coherent_hlc_with_missing()))
    def test_matches_reference_under_missing_data(
        self,
        case: tuple[list[tuple[float | None, float | None, float | None]], int],
    ) -> None:
        """
        Verifies that, for inputs freely mixing null / NaN / finite, both fields match the non-mirror reference.
        """
        rows, window = case
        high, low, close = split_triples(rows)
        bands = apply_supertrend(high, low, close, window)
        reference = supertrend_reference(high, low, close, window, 3.0)
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
        case: tuple[list[tuple[float, float, float]], int],
        exponent: int,
    ) -> None:
        """
        Verifies that ``supertrend`` is homogeneous of degree 1 in its ``line`` and invariant in its ``direction``:
        scaling every input value by a constant ``k`` scales the ``line`` by the same ``k`` and leaves the ``direction``
        unchanged. ``k`` is a power of two, so the rescale is exact and adds no floating-point error.
        """
        k = 2.0**exponent
        rows, window = case
        high, low, close = split_triples(rows)
        base = apply_supertrend(high, low, close, window)
        scaled = apply_supertrend([v * k for v in high], [v * k for v in low], [v * k for v in close], window)
        assert_scale_homogeneous(scaled["line"], base["line"], k=k, degree=1)
        assert_matches(scaled["direction"], base["direction"])

    @given(
        case=_cases(coherent_hlc()),
        scale=st.sampled_from([1e-6, 1e6, 1e9]),
    )
    def test_matches_reference_at_large_magnitude(
        self,
        case: tuple[list[tuple[float, float, float]], int],
        scale: float,
    ) -> None:
        """
        Verifies that at extreme magnitudes the ``line`` stays finite where the reference is and agrees, and the
        ``direction`` is unchanged.
        """
        rows, window = case
        high = [row[0] * scale for row in rows]
        low = [row[1] * scale for row in rows]
        close = [row[2] * scale for row in rows]
        bands = apply_supertrend(high, low, close, window)
        reference = supertrend_reference(high, low, close, window, 3.0)
        for field in FIELDS:
            assert_matches(
                bands[field],
                reference[field],
                rel_tol=RELATIVE_TOLERANCE_SCALE,
                abs_tol=input_scale(close) * EXACT_TOLERANCE_FACTOR,
            )
