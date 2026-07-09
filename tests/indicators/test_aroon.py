"""
Tests for ``pomata.indicators.aroon`` — the Aroon up / down trend indicator.

``aroon`` is multi-input (high, low) and returns a single struct ``pl.Expr`` with the fields ``up`` / ``down``. The
local ``apply_aroon`` helper materializes each field over a two-column ``Float64`` frame into a dict of lists, so the
shared ``assert_matches`` and the naive ``aroon_reference`` oracle (a plain arg-max, independent of the implementation's
``min_horizontal`` trick) compare field by field. Each line depends only on the *position* of the window extreme, so it
is bounded in ``[0, 100]`` and scale-invariant — so it carries scale-invariance and boundedness properties in place of
the homogeneity / large-magnitude tests used for scale-dependent indicators.

The ladder is the canonical one: contract, edge (warm-up / current-extreme / ties / null / NaN), correctness (vs the
closed-form reference and a frozen golden master), and properties (reference agreement incl. missing data,
scale-invariance, boundedness). Categories are split into classes; cross-cutting categories use markers (see
``tests/README.md``).
"""

import math
from collections.abc import Sequence

import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st
from tests.indicators.oracles import aroon_reference
from tests.support import (
    BOUND_MARGIN,
    GROUP_KEY,
    HIGH,
    LOW,
    assert_matches,
    assert_scale_homogeneous,
    coherent_hl,
    coherent_hl_with_missing,
    materialize_struct,
    split_pairs,
)

from pomata.indicators import aroon

FIELDS = ("up", "down")

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- derived, not chosen; the rationale is the shared method in CORRECTNESS.md, the numbers are this
# indicator's. To add an indicator, set its three facts here; the property tier below is then the same shape as every
# other indicator's.
#   1. warmup  W(window) = window   (both lines are null for the first ``window`` rows: a full ``window + 1``-bar
#              look-back must exist before the position of the window extreme is defined)
#   2. memory  the oracle shares pomata's seeding, so the property holds from the first defined row (M = 0); each
#              example carries D in [window, 2 * window] defined bars -- one window of output, never all warm-up
#   3. domain  the agreement / boundedness tiers draw ``tied_hl`` (small discrete highs / lows so frequent ties exercise
#              the most-recent-extreme rule); the scale tier draws ``coherent_hl`` and the missing-data tier
#              ``coherent_hl_with_missing``. Windows span 1 .. WINDOW_MAX
# Aroon is a scale-INVARIANT bounded line (O(1) in ``[0, 100]``, a position-of-extreme percentage), so the scale tier
# uses an ABSOLUTE tolerance, never ``input_scale``-sized, and the large-magnitude tier is vacuous (the common factor
# cancels) and absent. Repetitions N are the shared CI profile (tests/conftest.py); override per-test only if its
# parameter space is larger.
# ----------------------------------------------------------------------------------------------------------------------
WINDOW_MAX = 15


@st.composite
def _cases[T](draw: st.DrawFn, bars: st.SearchStrategy[T]) -> tuple[list[T], int]:
    """
    A (series, window) pair sized from the facts above: ``window`` over its regimes, length = warm-up + a window of
    defined bars, so every example has output to check (never an all-warm-up series, the waste a ``window`` decoupled
    from the length would cause).
    """
    window = draw(st.integers(min_value=1, max_value=WINDOW_MAX))
    defined = draw(st.integers(min_value=window, max_value=2 * window))
    length = window + defined
    return draw(st.lists(bars, min_size=length, max_size=length)), window


@st.composite
def tied_hl(draw: st.DrawFn) -> tuple[float, float]:
    """
    A ``(high, low)`` bar drawn from a small discrete grid, so frequent ties exercise the most-recent-extreme rule.
    """
    grid = [1.0, 2.0, 3.0, 4.0, 5.0]
    return draw(st.sampled_from(grid)), draw(st.sampled_from(grid))


def apply_aroon(
    high: Sequence[float | None],
    low: Sequence[float | None],
    window: int,
) -> dict[str, list[float | None]]:
    """
    Materialize each field of ``aroon`` over a two-column frame, as a dict mirroring the oracle's output.
    """
    return materialize_struct(
        {HIGH: high, LOW: low},
        aroon(pl.col(HIGH), pl.col(LOW), window),
        FIELDS,
    )


class TestAroonContract:
    """
    Type, struct schema, shape, and lazy/eager guarantees.
    """

    def test_over_partitions_independently(self) -> None:
        """
        Verifies that under ``.over`` the rolling extremes reset per group: the partitioned line equals the per-group
        calls.
        """
        frame = pl.DataFrame(
            {
                GROUP_KEY: ["a"] * 4 + ["b"] * 4,
                HIGH: [10.0, 11.0, 12.0, 11.0, 20.0, 22.0, 21.0, 23.0],
                LOW: [9.0, 10.0, 11.0, 10.0, 19.0, 21.0, 20.0, 22.0],
            }
        )
        up = aroon(pl.col(HIGH), pl.col(LOW), 2).over(GROUP_KEY).struct.field("up")
        grouped = frame.select(up.alias("y"))["y"].to_list()
        group_a = apply_aroon([10.0, 11.0, 12.0, 11.0], [9.0, 10.0, 11.0, 10.0], 2)["up"]
        group_b = apply_aroon([20.0, 22.0, 21.0, 23.0], [19.0, 21.0, 20.0, 22.0], 2)["up"]
        assert_matches(grouped, group_a + group_b)

    def test_output_is_struct_with_named_fields(self) -> None:
        """
        Verifies that the output is a ``Float64`` struct with exactly the fields ``up`` / ``down``.
        """
        frame = pl.DataFrame({HIGH: [3.0, 2.0, 4.0], LOW: [1.0, 0.0, 2.0]})
        dtype = frame.select(aroon(pl.col(HIGH), pl.col(LOW), 2).alias("a")).schema["a"]
        assert isinstance(dtype, pl.Struct)
        assert [field.name for field in dtype.fields] == ["up", "down"]
        assert all(field.dtype == pl.Float64 for field in dtype.fields)


class TestAroonEdge:
    """
    Boundaries, warm-up, current-extreme, ties, and null / NaN handling.
    """

    def test_window_below_one_raises(self) -> None:
        """
        Verifies that ``window < 1`` raises ``ValueError``.
        """
        with pytest.raises(ValueError, match="window must be >= 1"):
            aroon(pl.col(HIGH), pl.col(LOW), 0)

    def test_single_row(self) -> None:
        """
        Verifies behavior on a one-element series: the lone bar is always warm-up on every field.
        """
        bands = apply_aroon([5.0], [4.0], 2)
        for field in FIELDS:
            assert_matches(bands[field], [None])

    def test_all_null(self) -> None:
        """
        Verifies that all-null inputs yield all null on every field.
        """
        bands = apply_aroon([None, None, None, None], [None, None, None, None], 2)
        for field in FIELDS:
            assert_matches(bands[field], [None, None, None, None])

    def test_null_in_window_is_null(self) -> None:
        """
        Verifies that a ``null`` anywhere in the look-back yields ``null`` on the affected line.
        """
        values_high = [10.0, 11.0, None, 13.0, 14.0, 15.0]
        values_low = [9.0, 10.0, 11.0, 12.0, 13.0, 14.0]
        bands = apply_aroon(values_high, values_low, 2)
        reference = aroon_reference(values_high, values_low, 2)
        for field in FIELDS:
            assert_matches(bands[field], reference[field])

    def test_nan_propagates(self) -> None:
        """
        Verifies that a ``NaN`` in the look-back yields ``NaN`` on the affected line (not treated as an extreme).
        """
        values_high = [10.0, 11.0, 12.0, math.nan, 14.0, 15.0]
        values_low = [9.0, 10.0, 11.0, 12.0, 13.0, 14.0]
        bands = apply_aroon(values_high, values_low, 2)
        reference = aroon_reference(values_high, values_low, 2)
        for field in FIELDS:
            assert_matches(bands[field], reference[field])

    def test_warmup_null_count(self) -> None:
        """
        Verifies that both lines are null for the first ``window`` rows and defined once a full look-back exists.
        """
        bands = apply_aroon([1.0, 2.0, 3.0, 4.0, 5.0], [0.0, 1.0, 2.0, 3.0, 4.0], 2)
        for field in FIELDS:
            assert bands[field][:2] == [None, None]
            assert bands[field][2] is not None

    def test_window_exceeds_length(self) -> None:
        """
        Verifies that when ``window`` exceeds the series length every field is null (no full look-back exists).
        """
        bands = apply_aroon([1.0, 2.0, 3.0], [0.0, 1.0, 2.0], 5)
        for field in FIELDS:
            assert_matches(bands[field], [None, None, None])

    def test_current_extreme_reads_100(self) -> None:
        """
        Verifies that when the current bar holds the look-back high (low) the up (down) line reads ``100``.
        """
        bands = apply_aroon([1.0, 2.0, 3.0], [3.0, 2.0, 1.0], 2)
        assert_matches(bands["up"], [None, None, 100.0])
        assert_matches(bands["down"], [None, None, 100.0])

    def test_ties_use_most_recent_extreme(self) -> None:
        """
        Verifies that a repeated extreme resolves to the most recent occurrence (here the high ``5`` one bar back).
        """
        bands = apply_aroon([5.0, 5.0, 3.0], [1.0, 2.0, 3.0], 2)
        assert_matches(bands["up"], [None, None, 50.0])


class TestAroonCorrectness:
    """
    Against the naive reference oracle and frozen golden-master values.
    """

    def test_matches_reference(self) -> None:
        """
        Verifies that both lines agree with the naive closed-form reference across several windows.
        """
        high = [10.0, 11.0, 12.0, 11.0, 13.0, 12.0, 14.0, 13.0, 15.0, 14.0]
        low = [9.0, 10.0, 11.0, 10.0, 12.0, 11.0, 13.0, 12.0, 14.0, 13.0]
        for window in (1, 2, 3, 5):
            bands = apply_aroon(high, low, window)
            reference = aroon_reference(high, low, window)
            for field in FIELDS:
                assert_matches(bands[field], reference[field])

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference: aroon(window=3) over the sample series.
        """
        high = [10.0, 11.0, 12.0, 11.0, 13.0, 12.0, 14.0, 13.0]
        low = [9.0, 10.0, 11.0, 10.0, 12.0, 11.0, 13.0, 12.0]
        bands = apply_aroon(high, low, 3)
        assert_matches(
            [None if v is None else round(v, 4) for v in bands["up"]],
            [None, None, None, 66.6667, 100.0, 66.6667, 100.0, 66.6667],
        )
        assert_matches(
            [None if v is None else round(v, 4) for v in bands["down"]],
            [None, None, None, 0.0, 66.6667, 33.3333, 0.0, 33.3333],
        )


class TestAroonProperties:
    """
    Invariants that must hold for all inputs (property-based).
    """

    @given(case=_cases(tied_hl()))
    def test_matches_reference_for_any_input(
        self,
        case: tuple[list[tuple[float, float]], int],
    ) -> None:
        """
        Verifies that, for any series and window (small integers force frequent ties), both lines match the reference.
        """
        rows, window = case
        high, low = split_pairs(rows)
        bands = apply_aroon(high, low, window)
        reference = aroon_reference(high, low, window)
        for field in FIELDS:
            assert_matches(bands[field], reference[field])

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
        bands = apply_aroon(high, low, window)
        reference = aroon_reference(high, low, window)
        for field in FIELDS:
            assert_matches(bands[field], reference[field])

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
        Verifies that ``aroon`` is scale-invariant: scaling every input value by a constant ``k`` leaves the output
        unchanged -- ``aroon(k * x) == aroon(x)``. ``k`` is a power of two, so the rescale is exact and adds no
        floating-point error.
        """
        k = 2.0**exponent
        rows, window = case
        high, low = split_pairs(rows)
        base = apply_aroon(high, low, window)
        scaled = apply_aroon([value * k for value in high], [value * k for value in low], window)
        for field in FIELDS:
            assert_scale_homogeneous(scaled[field], base[field], k=k, degree=0)

    @given(case=_cases(tied_hl()))
    def test_bounded(
        self,
        case: tuple[list[tuple[float, float]], int],
    ) -> None:
        """
        Verifies that every defined value of both lines lies within ``[0, 100]``.
        """
        rows, window = case
        high, low = split_pairs(rows)
        bands = apply_aroon(high, low, window)
        for field in FIELDS:
            for value in bands[field]:
                if value is not None and not math.isnan(value):
                    assert 0.0 - BOUND_MARGIN <= value <= 100.0 + BOUND_MARGIN
