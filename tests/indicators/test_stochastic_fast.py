"""
Tests for ``pomata.indicators.stochastic_fast`` — the Fast Stochastic Oscillator (%K and %D).

``stochastic_fast`` is multi-input (high, low, close) and multi-output: it returns a single struct ``pl.Expr`` with the
fields ``k`` / ``d``. The local ``apply_stochastic_fast`` helper materializes each field over a three-column ``Float64``
frame into a dict of lists, so the shared ``assert_matches`` and the naive ``stochastic_fast_reference`` oracle (which
returns the matching dict) compare line by line. Both lines are scale-invariant and bounded in ``[0, 100]`` for
well-formed bars.

The ladder is the canonical one: contract (type / struct schema / shape / lazy-eager / ``.over`` independence), edge
(window floors / warm-up / flat range / null / NaN), correctness (vs the closed-form reference and a frozen golden
master), and properties (reference agreement incl. missing data, scale-invariance, boundedness). Categories are split
into classes; cross-cutting categories use markers (see ``tests/README.md``).
"""

import math
from collections.abc import Sequence

import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st
from tests.indicators.oracles import stochastic_fast_reference
from tests.support import (
    ABSOLUTE_TOLERANCE_PROPERTY,
    BOUND_MARGIN,
    CLOSE,
    GROUP_KEY,
    HIGH,
    LOW,
    RELATIVE_TOLERANCE_PROPERTY,
    assert_matches,
    assert_scale_homogeneous,
    coherent_hlc,
    coherent_hlc_with_missing,
    materialize_struct,
    split_triples,
)

from pomata.indicators import stochastic_fast

FIELDS = ("k", "d")

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- derived, not chosen; the rationale is the shared method in CORRECTNESS.md, the numbers are this
# indicator's. To add an indicator, set its three facts here; the property tier below is then the same shape as every
# other indicator's.
#   1. warmup  W(window_k, window_d) = window_k + window_d - 2   (%K warms up over ``window_k - 1`` rows and %D over a
#              further ``window_d - 1`` of the sma of %K)
#   2. memory  the oracle shares pomata's seeding, so the property holds from the first defined row (M = 0); each
#              example carries D in [W + 1, W + 1 + span] defined bars -- a window of output, never all warm-up
#   3. domain  coherent_hlc(): coherent (high >= low, low <= close <= high) positive-finite bars -- both lines are only
#              well-defined and bounded in ``[0, 100]`` on well-formed bars; windows span 1 .. WINDOW_MAX
# Both lines are scale-INVARIANT bounded ratios (O(1) in ``[0, 100]``), so the scale tier uses an ABSOLUTE tolerance,
# never ``input_scale``-sized, and the large-magnitude tier is vacuous (the common factor cancels) and absent.
# Repetitions N are the shared CI profile (tests/conftest.py); override per-test only if its parameter space is larger.
# ----------------------------------------------------------------------------------------------------------------------
WINDOW_MAX = 12


@st.composite
def _cases[T](draw: st.DrawFn, bars: st.SearchStrategy[T]) -> tuple[list[T], int, int]:
    """
    A (series, window_k, window_d) triple sized from the facts above: each window over its regimes, length = warm-up + a
    window of defined bars, so every example has output to check (never an all-warm-up series, the waste a window
    decoupled from the length would cause).
    """
    window_k = draw(st.integers(min_value=1, max_value=WINDOW_MAX))
    window_d = draw(st.integers(min_value=1, max_value=WINDOW_MAX))
    warmup = window_k + window_d - 2
    defined = draw(st.integers(min_value=1, max_value=WINDOW_MAX))
    length = warmup + defined
    return draw(st.lists(bars, min_size=length, max_size=length)), window_k, window_d


def apply_stochastic_fast(
    high: Sequence[float | None],
    low: Sequence[float | None],
    close: Sequence[float | None],
    window_k: int,
    window_d: int,
) -> dict[str, list[float | None]]:
    """
    Materialize each line of ``stochastic_fast`` over a three-column ``Float64`` frame, as a dict mirroring the oracle.
    """
    return materialize_struct(
        {HIGH: high, LOW: low, CLOSE: close},
        stochastic_fast(pl.col(HIGH), pl.col(LOW), pl.col(CLOSE), window_k=window_k, window_d=window_d),
        FIELDS,
    )


class TestStochasticFastContract:
    """
    Type, struct schema, shape, and lazy/eager guarantees.
    """

    def test_output_is_struct_with_named_fields(self) -> None:
        """
        Verifies that the output is a ``Float64`` struct with exactly the fields ``k`` / ``d``.
        """
        frame = pl.DataFrame({HIGH: [10.0, 11.0, 12.0], LOW: [9.0, 10.0, 11.0], CLOSE: [9.5, 10.5, 11.5]})
        dtype = frame.select(
            stochastic_fast(pl.col(HIGH), pl.col(LOW), pl.col(CLOSE), window_k=2, window_d=2).alias("s")
        ).schema["s"]
        assert isinstance(dtype, pl.Struct)
        assert [field.name for field in dtype.fields] == ["k", "d"]
        assert all(field.dtype == pl.Float64 for field in dtype.fields)

    def test_over_partitions_independently(self) -> None:
        """
        Verifies that under ``.over`` the windows reset per group and never span boundaries.
        """
        frame = pl.DataFrame(
            {
                GROUP_KEY: ["a"] * 4 + ["b"] * 4,
                HIGH: [10.0, 11.0, 12.0, 11.5, 20.0, 21.0, 22.0, 21.5],
                LOW: [9.0, 10.0, 11.0, 10.5, 19.0, 20.0, 21.0, 20.5],
                CLOSE: [9.5, 10.5, 11.5, 11.0, 19.5, 20.5, 21.5, 21.0],
            }
        )
        oscillator = stochastic_fast(pl.col(HIGH), pl.col(LOW), pl.col(CLOSE), window_k=2, window_d=2).over(GROUP_KEY)
        grouped = {
            field: frame.select(oscillator.struct.field(field).alias(field))[field].to_list() for field in FIELDS
        }
        group_a = apply_stochastic_fast(
            [10.0, 11.0, 12.0, 11.5], [9.0, 10.0, 11.0, 10.5], [9.5, 10.5, 11.5, 11.0], 2, 2
        )
        group_b = apply_stochastic_fast(
            [20.0, 21.0, 22.0, 21.5], [19.0, 20.0, 21.0, 20.5], [19.5, 20.5, 21.5, 21.0], 2, 2
        )
        for field in FIELDS:
            assert_matches(grouped[field], group_a[field] + group_b[field])


class TestStochasticFastEdge:
    """
    Boundaries, warm-up, flat range, and null / NaN handling.
    """

    def test_window_k_below_one_raises(self) -> None:
        """
        Verifies that ``window_k < 1`` raises ``ValueError``.
        """
        with pytest.raises(ValueError, match="window_k must be >= 1"):
            stochastic_fast(pl.col(HIGH), pl.col(LOW), pl.col(CLOSE), window_k=0, window_d=3)

    def test_window_d_below_one_raises(self) -> None:
        """
        Verifies that ``window_d < 1`` raises ``ValueError``.
        """
        with pytest.raises(ValueError, match="window_d must be >= 1"):
            stochastic_fast(pl.col(HIGH), pl.col(LOW), pl.col(CLOSE), window_k=3, window_d=0)

    def test_all_null(self) -> None:
        """
        Verifies that an all-null input yields an all-null output on both lines.
        """
        result = apply_stochastic_fast(
            [None, None, None], [None, None, None], [None, None, None], window_k=2, window_d=2
        )
        assert_matches(result["k"], [None, None, None])
        assert_matches(result["d"], [None, None, None])

    def test_single_row(self) -> None:
        """
        Verifies that a one-row series is all warm-up on both lines (the longest window exceeds the length).
        """
        result = apply_stochastic_fast([11.0], [9.0], [10.0], window_k=3, window_d=2)
        assert_matches(result["k"], [None])
        assert_matches(result["d"], [None])

    def test_window_exceeds_length(self) -> None:
        """
        Verifies that a short series whose longest window exceeds the length is all warm-up on both lines.
        """
        result = apply_stochastic_fast(
            [11.0, 12.0, 13.0], [9.0, 10.0, 11.0], [10.0, 11.0, 12.0], window_k=5, window_d=2
        )
        assert_matches(result["k"], [None, None, None])
        assert_matches(result["d"], [None, None, None])

    def test_warmup_null_count(self) -> None:
        """
        Verifies that ``k`` warms up over ``window_k - 1`` rows and ``d`` over a further ``window_d - 1``.
        """
        high = [10.0, 11.0, 12.0, 11.5, 13.0, 12.5]
        low = [9.0, 10.0, 11.0, 10.5, 12.0, 11.5]
        close = [9.5, 10.5, 11.5, 11.0, 12.5, 12.0]
        result = apply_stochastic_fast(high, low, close, window_k=3, window_d=2)
        assert result["k"][:2] == [None, None]
        assert result["k"][2] is not None
        assert result["d"][:3] == [None, None, None]
        assert result["d"][3] is not None

    def test_flat_window_is_nan(self) -> None:
        """
        Verifies that a flat look-back (highest high equals lowest low) yields ``NaN`` on ``k`` (``0 / 0``).
        """
        result = apply_stochastic_fast(
            [10.0, 10.0, 10.0], [10.0, 10.0, 10.0], [10.0, 10.0, 10.0], window_k=2, window_d=1
        )
        assert result["k"][0] is None
        assert result["k"][1] is not None
        assert math.isnan(result["k"][1])

    def test_null_propagates(self) -> None:
        """
        Verifies that a null propagates (matching the naive reference).
        """
        high = [10.0, 11.0, 12.0, None, 13.0, 13.0, 14.0, 13.5]
        low = [9.0, 10.0, 11.0, 10.5, 12.0, 11.5, 13.0, 12.5]
        close = [9.5, 10.5, 11.5, 11.0, 12.5, 12.0, 13.5, 13.0]
        applied = apply_stochastic_fast(high, low, close, window_k=2, window_d=2)
        reference = stochastic_fast_reference(high, low, close, 2, 2)
        for field in FIELDS:
            assert_matches(applied[field], reference[field])

    def test_nan_propagates(self) -> None:
        """
        Verifies that a NaN propagates (matching the naive reference).
        """
        high = [10.0, 11.0, 12.0, 12.0, 13.0, math.nan, 14.0, 13.5]
        low = [9.0, 10.0, 11.0, 10.5, 12.0, 11.5, 13.0, 12.5]
        close = [9.5, 10.5, 11.5, 11.0, 12.5, 12.0, 13.5, 13.0]
        applied = apply_stochastic_fast(high, low, close, window_k=2, window_d=2)
        reference = stochastic_fast_reference(high, low, close, 2, 2)
        for field in FIELDS:
            assert_matches(applied[field], reference[field])


class TestStochasticFastCorrectness:
    """
    Against the naive reference oracle and frozen golden-master values.
    """

    def test_matches_reference(self) -> None:
        """
        Verifies agreement with the naive closed-form reference across several window pairs.
        """
        high = [10.0, 11.0, 12.0, 11.5, 13.0, 12.5, 14.0, 13.5, 15.0, 14.5]
        low = [9.0, 10.0, 11.0, 10.5, 12.0, 11.5, 13.0, 12.5, 14.0, 13.5]
        close = [9.5, 10.5, 11.5, 11.0, 12.5, 12.0, 13.5, 13.0, 14.5, 14.0]
        for window_k, window_d in ((1, 1), (3, 2), (5, 3), (5, 1)):
            applied = apply_stochastic_fast(high, low, close, window_k=window_k, window_d=window_d)
            reference = stochastic_fast_reference(high, low, close, window_k, window_d)
            for field in FIELDS:
                assert_matches(
                    applied[field],
                    reference[field],
                    rel_tol=RELATIVE_TOLERANCE_PROPERTY,
                    abs_tol=ABSOLUTE_TOLERANCE_PROPERTY,
                )

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference: stochastic_fast(window_k=5, window_d=3) over the sample series.
        """
        high = [10.0, 11.0, 12.0, 11.5, 13.0, 12.5, 14.0, 13.5, 15.0, 14.5]
        low = [9.0, 10.0, 11.0, 10.5, 12.0, 11.5, 13.0, 12.5, 14.0, 13.5]
        close = [9.5, 10.5, 11.5, 11.0, 12.5, 12.0, 13.5, 13.0, 14.5, 14.0]
        result = apply_stochastic_fast(high, low, close, window_k=5, window_d=3)
        assert_matches(
            [None if v is None else round(v, 4) for v in result["k"]],
            [None, None, None, None, 87.5, 66.6667, 85.7143, 71.4286, 85.7143, 71.4286],
        )
        assert_matches(
            [None if v is None else round(v, 4) for v in result["d"]],
            [None, None, None, None, None, None, 79.9603, 74.6032, 80.9524, 76.1905],
        )


class TestStochasticFastProperties:
    """
    Invariants that must hold for all inputs (property-based).
    """

    @given(case=_cases(coherent_hlc()))
    def test_matches_reference_for_any_input(
        self,
        case: tuple[list[tuple[float, float, float]], int, int],
    ) -> None:
        """
        Verifies that, for any positive series and windows, the implementation matches the naive reference.
        """
        rows, window_k, window_d = case
        high, low, close = split_triples(rows)
        applied = apply_stochastic_fast(high, low, close, window_k=window_k, window_d=window_d)
        reference = stochastic_fast_reference(high, low, close, window_k, window_d)
        for field in FIELDS:
            assert_matches(
                applied[field],
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
        case: tuple[list[tuple[float, float, float]], int, int],
        exponent: int,
    ) -> None:
        """
        Verifies that both lines are scale-invariant under a positive common rescaling of high / low / close. ``k`` is a
        power of two so the rescaling is lossless: the lines pick the extremes of the look-back, and an arbitrary factor
        can round two near-tied bars to the same value and flip which one wins, changing the result.
        """
        k = 2.0**exponent
        rows, window_k, window_d = case
        high, low, close = split_triples(rows)
        base = apply_stochastic_fast(high, low, close, window_k=window_k, window_d=window_d)
        scaled = apply_stochastic_fast(
            [value * k for value in high],
            [value * k for value in low],
            [value * k for value in close],
            window_k,
            window_d,
        )
        for field in FIELDS:
            assert_scale_homogeneous(scaled[field], base[field], k=k, degree=0)

    @given(case=_cases(coherent_hlc()))
    def test_bounded(
        self,
        case: tuple[list[tuple[float, float, float]], int, int],
    ) -> None:
        """
        Verifies that every defined value lies within ``[0, 100]`` for well-formed OHLC bars.
        """
        rows, window_k, window_d = case
        high, low, close = split_triples(rows)
        applied = apply_stochastic_fast(high, low, close, window_k=window_k, window_d=window_d)
        for field in FIELDS:
            for value in applied[field]:
                if value is not None and not math.isnan(value):
                    assert -BOUND_MARGIN <= value <= 100.0 + BOUND_MARGIN

    @given(case=_cases(coherent_hlc_with_missing()))
    def test_matches_reference_under_missing_data(
        self,
        case: tuple[list[tuple[float | None, float | None, float | None]], int, int],
    ) -> None:
        """
        Verifies that, for positive inputs freely mixing null / NaN, the implementation matches the naive reference.
        """
        rows, window_k, window_d = case
        high, low, close = split_triples(rows)
        applied = apply_stochastic_fast(high, low, close, window_k=window_k, window_d=window_d)
        reference = stochastic_fast_reference(high, low, close, window_k, window_d)
        for field in FIELDS:
            assert_matches(applied[field], reference[field])
