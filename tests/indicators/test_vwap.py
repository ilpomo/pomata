"""
Tests for ``pomata.indicators.vwap`` — the running volume-weighted mean of the typical price.

``vwap`` is multi-input (``high`` / ``low`` / ``close`` / ``volume``), single-output, and cumulative (anchored to the
partition start, not windowed). Tests use a local ``apply_vwap`` helper over a four-column ``Float64`` frame;
``assert_matches`` and the naive ``vwap_reference`` oracle (the certified ``price_typical`` + matched cumulative sums)
are shared.

The ladder is the canonical one: contract (type / shape / lazy-eager / ``.over`` anchoring), edge (zero-volume head /
interior zero / negative volume / single-row / null / NaN), correctness (vs the cumulative reference and a frozen golden
master), and properties (reference agreement incl. missing data, the convex-combination bound, per-session
independence, degree-1 price homogeneity, degree-0 volume invariance, large-magnitude stability). Categories are split
into classes; cross-cutting categories use markers (see ``tests/README.md``).
"""

import math
from collections.abc import Sequence

import polars as pl
from hypothesis import given
from hypothesis import strategies as st
from polars.testing import assert_frame_equal
from tests.indicators.oracles import vwap_reference
from tests.support import (
    CLOSE,
    EXACT_TOLERANCE_FACTOR,
    HIGH,
    LOW,
    RELATIVE_TOLERANCE_PROPERTY,
    RELATIVE_TOLERANCE_SCALE,
    VOLUME,
    assert_matches,
    assert_scale_homogeneous,
    coherent_hlcv,
    coherent_hlcv_with_missing,
    input_scale,
    materialize,
    split_quads,
)

from pomata.indicators import vwap

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- derived, not chosen; the rationale is the shared method in CORRECTNESS.md, the numbers are this
# indicator's. To add an indicator, set its three facts here; the property tier below is then the same shape as every
# other indicator's.
#   1. warmup  none -- VWAP is defined from row 0 (NaN only while the cumulative volume is still zero)
#   2. memory  the oracle is cumulative like pomata, so the property holds from row 0 (M = 0); each example carries a
#              run of defined bars, never empty
#   3. domain  coherent_hlcv(): coherent (high >= low, low <= close <= high) positive-finite bars with volume >= 1, so
#              the cumulative volume is always positive and the convex-combination bound holds
# Series span 2 .. SERIES_MAX bars (no window). Repetitions N are the shared CI profile (tests/conftest.py).
# ----------------------------------------------------------------------------------------------------------------------
SERIES_MAX = 32


@st.composite
def _cases[T](draw: st.DrawFn, bars: st.SearchStrategy[T]) -> list[T]:
    """
    A run of bars sized from the facts above: cumulative, so any non-empty run has output to check from row 0.
    """
    length = draw(st.integers(min_value=2, max_value=SERIES_MAX))
    return draw(st.lists(bars, min_size=length, max_size=length))


def apply_vwap(
    high: Sequence[float | None],
    low: Sequence[float | None],
    close: Sequence[float | None],
    volume: Sequence[float | None],
) -> list[float | None]:
    """
    Materialize ``vwap`` over a four-column ``Float64`` frame built from the aligned high / low / close / volume lists.
    """
    return materialize(
        {HIGH: high, LOW: low, CLOSE: close, VOLUME: volume},
        vwap(pl.col(HIGH), pl.col(LOW), pl.col(CLOSE), pl.col(VOLUME)),
    )


class TestVwapContract:
    """
    Type, shape, lazy/eager, and ``.over`` anchoring guarantees.
    """

    def test_returns_expr(self) -> None:
        """
        Verifies that the factory returns a ``pl.Expr`` without touching a frame.
        """
        assert isinstance(vwap(pl.col(HIGH), pl.col(LOW), pl.col(CLOSE), pl.col(VOLUME)), pl.Expr)

    def test_preserves_length_and_dtype(self) -> None:
        """
        Verifies that the output has one value per input row and is ``Float64``.
        """
        frame = pl.DataFrame({HIGH: [2.0, 4.0], LOW: [0.0, 2.0], CLOSE: [1.0, 3.0], VOLUME: [10.0, 20.0]})
        result = frame.select(vwap(pl.col(HIGH), pl.col(LOW), pl.col(CLOSE), pl.col(VOLUME)).alias("y"))
        assert result.height == frame.height
        assert result.schema["y"] == pl.Float64

    def test_lazy_eager_parity(self) -> None:
        """
        Verifies that eager and lazy application produce identical materialized output.
        """
        frame = pl.DataFrame(
            {HIGH: [2.0, 4.0, 6.0], LOW: [0.0, 2.0, 4.0], CLOSE: [1.0, 3.0, 5.0], VOLUME: [10.0, 20.0, 30.0]}
        )
        expr = vwap(pl.col(HIGH), pl.col(LOW), pl.col(CLOSE), pl.col(VOLUME)).alias("y")
        assert_frame_equal(frame.select(expr), frame.lazy().select(expr).collect())

    def test_over_anchors_per_session(self) -> None:
        """
        Verifies that under ``.over`` the VWAP restarts per session and never accumulates across the boundary.
        """
        frame = pl.DataFrame(
            {
                "session": ["a", "a", "b", "b"],
                HIGH: [2.0, 4.0, 12.0, 14.0],
                LOW: [0.0, 2.0, 10.0, 12.0],
                CLOSE: [1.0, 3.0, 11.0, 13.0],
                VOLUME: [10.0, 20.0, 10.0, 20.0],
            }
        )
        expr = vwap(pl.col(HIGH), pl.col(LOW), pl.col(CLOSE), pl.col(VOLUME)).over("session")
        result = frame.select(expr.alias("y"))["y"].to_list()
        assert_matches(result, [1.0, 70.0 / 30.0, 11.0, 370.0 / 30.0])


class TestVwapEdge:
    """
    Zero / negative volume, single-row, null and NaN handling.
    """

    def test_empty(self) -> None:
        """
        Verifies that an empty input yields an empty output.
        """
        assert_matches(apply_vwap([], [], [], []), [])

    def test_all_null(self) -> None:
        """
        Verifies that an all-null input yields an all-null output (every contribution and the cumulative volume null).
        """
        assert_matches(apply_vwap([None, None], [None, None], [None, None], [None, None]), [None, None])

    def test_zero_volume_head_is_nan(self) -> None:
        """
        Verifies the ``0 / 0`` head: a leading zero-volume bar reads ``NaN`` until positive volume accrues.
        """
        result = apply_vwap([2.0, 4.0], [0.0, 2.0], [1.0, 3.0], [0.0, 20.0])
        head = result[0]
        assert head is not None
        assert math.isnan(head)
        assert result[1] == 3.0

    def test_interior_zero_volume_carries_forward(self) -> None:
        """
        Verifies an interior zero-volume bar: it adds nothing, so the prefix sums and thus the VWAP carry across it
        (no subtract-on-exit residual, since the sums are cumulative not windowed).
        """
        high = [2.0, 4.0, 6.0]
        low = [0.0, 2.0, 4.0]
        close = [1.0, 3.0, 5.0]
        volume = [10.0, 0.0, 30.0]
        assert_matches(apply_vwap(high, low, close, volume), vwap_reference(high, low, close, volume))

    def test_negative_volume_flows_through(self) -> None:
        """
        Verifies that negative volume is summed as-is (documented as out-of-domain, no guard): the output matches the
        reference computed on the same negative weights.
        """
        high = [2.0, 4.0, 6.0]
        low = [0.0, 2.0, 4.0]
        close = [1.0, 3.0, 5.0]
        volume = [10.0, -20.0, 30.0]
        assert_matches(apply_vwap(high, low, close, volume), vwap_reference(high, low, close, volume))

    def test_single_row(self) -> None:
        """
        Verifies a one-element series: VWAP is the bar's typical price (positive volume) or ``NaN`` (zero volume).
        """
        assert_matches(apply_vwap([2.0], [0.0], [1.0], [10.0]), [1.0])
        zero = apply_vwap([2.0], [0.0], [1.0], [0.0])[0]
        assert zero is not None
        assert math.isnan(zero)

    def test_null_and_nan_follow_the_cumulative_sums(self) -> None:
        """
        Verifies that a ``null`` (carried across) and a ``NaN`` (poisoning the rest) flow through exactly as the
        reference cumulative sums say.
        """
        high = [2.0, None, 6.0, 8.0, math.nan, 12.0]
        low = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
        close = [1.0, 3.0, 5.0, 7.0, 9.0, 11.0]
        volume = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
        assert_matches(apply_vwap(high, low, close, volume), vwap_reference(high, low, close, volume))


class TestVwapCorrectness:
    """
    Against the cumulative reference oracle and a frozen golden master.
    """

    def test_matches_reference(self) -> None:
        """
        Verifies agreement with the cumulative reference on a worked series.
        """
        high = [2.0, 4.0, 6.0, 5.0, 7.0, 9.0]
        low = [0.0, 2.0, 4.0, 3.0, 5.0, 7.0]
        close = [1.0, 3.0, 5.0, 4.0, 6.0, 8.0]
        volume = [10.0, 20.0, 30.0, 15.0, 25.0, 35.0]
        assert_matches(apply_vwap(high, low, close, volume), vwap_reference(high, low, close, volume))

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference: VWAP over a +2 typical ramp weighted 10 / 20 / 30.
        """
        result = apply_vwap([2.0, 4.0, 6.0], [0.0, 2.0, 4.0], [1.0, 3.0, 5.0], [10.0, 20.0, 30.0])
        assert_matches(result, [1.0, 70.0 / 30.0, 220.0 / 60.0])


class TestVwapProperties:
    """
    Invariants that must hold for all inputs (property-based).
    """

    @given(rows=_cases(coherent_hlcv()))
    def test_matches_reference_for_any_input(
        self,
        rows: list[tuple[float, float, float, float]],
    ) -> None:
        """
        Verifies that, for any coherent OHLCV run, the output matches the cumulative reference.
        """
        high, low, close, volume = split_quads(rows)
        assert_matches(
            apply_vwap(high, low, close, volume),
            vwap_reference(high, low, close, volume),
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=input_scale(close) * EXACT_TOLERANCE_FACTOR,
        )

    @given(rows=_cases(coherent_hlcv()))
    def test_within_typical_range(
        self,
        rows: list[tuple[float, float, float, float]],
    ) -> None:
        """
        Verifies the convex-combination bound: with positive volume the VWAP lies between the minimum and maximum
        typical price seen so far (a true invariant -- it is a weighted average of those typicals).
        """
        high, low, close, volume = split_quads(rows)
        result = apply_vwap(high, low, close, volume)
        typical = [(h + low_value + c) / 3.0 for h, low_value, c in zip(high, low, close, strict=True)]
        for index, value in enumerate(result):
            assert value is not None
            prefix = typical[: index + 1]
            assert min(prefix) - 1e-9 <= value <= max(prefix) + 1e-9

    @given(rows=_cases(coherent_hlcv()))
    def test_per_session_independence(
        self,
        rows: list[tuple[float, float, float, float]],
    ) -> None:
        """
        Verifies the anchoring semantics: two sessions concatenated and run under ``.over`` reproduce each session
        computed alone, concatenated.
        """
        high, low, close, volume = split_quads(rows)
        alone = apply_vwap(high, low, close, volume)
        frame = pl.DataFrame(
            {
                "session": ["a"] * len(rows) + ["b"] * len(rows),
                HIGH: pl.Series(high + high, dtype=pl.Float64),
                LOW: pl.Series(low + low, dtype=pl.Float64),
                CLOSE: pl.Series(close + close, dtype=pl.Float64),
                VOLUME: pl.Series(volume + volume, dtype=pl.Float64),
            }
        )
        over = frame.select(vwap(pl.col(HIGH), pl.col(LOW), pl.col(CLOSE), pl.col(VOLUME)).over("session").alias("y"))[
            "y"
        ].to_list()
        assert_matches(over, alone + alone)

    @given(
        rows=_cases(coherent_hlcv()),
        exponent=st.sampled_from([-4, -3, -2, -1, 1, 2, 3, 4]),
    )
    def test_price_homogeneity(
        self,
        rows: list[tuple[float, float, float, float]],
        exponent: int,
    ) -> None:
        """
        Verifies degree-1 homogeneity in price: ``VWAP(k * prices, volume) == k * VWAP``. ``k`` is a power of two so the
        rescaling is lossless.
        """
        k = 2.0**exponent
        high, low, close, volume = split_quads(rows)
        base = apply_vwap(high, low, close, volume)
        scaled = apply_vwap([v * k for v in high], [v * k for v in low], [v * k for v in close], volume)
        assert_scale_homogeneous(scaled, base, k=k, degree=1)

    @given(
        rows=_cases(coherent_hlcv()),
        exponent=st.sampled_from([-4, -2, 1, 3, 6]),
    )
    def test_volume_invariance(
        self,
        rows: list[tuple[float, float, float, float]],
        exponent: int,
    ) -> None:
        """
        Verifies degree-0 invariance in volume: rescaling volume by any positive ``k`` leaves VWAP bit-identical (the
        weight cancels in the ratio). ``k`` is a power of two so the rescaling is lossless.
        """
        k = 2.0**exponent
        high, low, close, volume = split_quads(rows)
        base = apply_vwap(high, low, close, volume)
        rescaled = apply_vwap(high, low, close, [v * k for v in volume])
        assert_scale_homogeneous(rescaled, base, k=k, degree=0)

    @given(rows=_cases(coherent_hlcv_with_missing()))
    def test_matches_reference_under_missing_data(
        self,
        rows: list[tuple[float | None, float | None, float | None, float | None]],
    ) -> None:
        """
        Verifies that, for inputs freely mixing null / NaN / finite, the output matches the cumulative reference.
        """
        high, low, close, volume = split_quads(rows)
        assert_matches(
            apply_vwap(high, low, close, volume),
            vwap_reference(high, low, close, volume),
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=input_scale(close) * EXACT_TOLERANCE_FACTOR,
        )

    @given(
        rows=_cases(coherent_hlcv()),
        scale=st.sampled_from([1e-6, 1e6, 1e9]),
    )
    def test_matches_reference_at_large_magnitude(
        self,
        rows: list[tuple[float, float, float, float]],
        scale: float,
    ) -> None:
        """
        Verifies that at extreme magnitudes in both price and volume the output stays finite where the reference is and
        agrees (the long cumulative sums are the conditioning concern).
        """
        high = [r[0] * scale for r in rows]
        low = [r[1] * scale for r in rows]
        close = [r[2] * scale for r in rows]
        volume = [r[3] * scale for r in rows]
        assert_matches(
            apply_vwap(high, low, close, volume),
            vwap_reference(high, low, close, volume),
            rel_tol=RELATIVE_TOLERANCE_SCALE,
            abs_tol=input_scale(close) * EXACT_TOLERANCE_FACTOR,
        )
