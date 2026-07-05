"""
Tests for ``pomata.indicators.true_range`` — Wilder's single-bar True Range.

``true_range`` is multi-input (``high`` / ``low`` / ``close``) and windowless, so tests use a local ``apply_true_range``
helper to materialize the factory over a three-column ``Float64`` frame rather than the shared single-column
``apply_expr``; ``assert_matches`` and the naive ``true_range_reference`` oracle are shared across the suite.

The ladder is the canonical one, adapted to a windowless indicator: contract (type / shape / lazy-eager / ``.over``
parity), edge (first-row fallback / empty / single-row / null / NaN), correctness (vs the closed-form reference and a
frozen golden master), and properties (reference agreement for any input, non-negativity on valid OHLC, and
scale-homogeneity). Categories are split into classes; cross-cutting categories elsewhere use markers (see
``tests/README.md``).
"""

import math
from collections.abc import Sequence

import polars as pl
from hypothesis import given
from hypothesis import strategies as st
from tests.indicators.oracles import true_range_reference
from tests.support import (
    ABSOLUTE_TOLERANCE_REFERENCE,
    CLOSE,
    GROUP_KEY,
    HIGH,
    LOW,
    RELATIVE_TOLERANCE_PROPERTY,
    RELATIVE_TOLERANCE_REFERENCE,
    assert_matches,
    assert_scale_homogeneous,
    coherent_hlc,
    coherent_hlc_with_missing,
    materialize,
    split_triples,
)

from pomata.indicators import true_range

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- derived, not chosen; the rationale is the shared method in CORRECTNESS.md, the numbers are this
# indicator's. To add an indicator, set its three facts here; the property tier below is then the same shape as every
# other indicator's.
#   1. warmup  W = 0   (windowless: every row is defined from row 0, which falls back to ``high - low``)
#   2. memory  the oracle shares pomata's per-row recomputation, so the property holds from row 0 (M = 0); with W = 0
#              there is no warm-up to outlast, so a case is simply a series of bars -- every row is output
#   3. domain  coherent_hlc(): coherent (high >= low, low <= close <= high) positive-finite bars -- True Range is only
#              non-negative on well-formed OHLC; SERIES_MAX bars span several total sizes
# True Range has no window parameter, so ``_cases`` draws only the series (no window to couple, hence no ``window`` in
# the unpacked pair). Repetitions N are the shared CI profile (tests/conftest.py); override per-test only if its
# parameter space is larger.
# ----------------------------------------------------------------------------------------------------------------------
SERIES_MAX = 50


@st.composite
def _cases[T](draw: st.DrawFn, bars: st.SearchStrategy[T]) -> list[T]:
    """
    A series of bars sized from the facts above. True Range is windowless (W = 0), so -- unlike the windowed
    indicators' ``(series, window)`` pair -- a case is just the series: every row is output, never warm-up.
    """
    # NOTE: windowless -- returns the bare series (no window to couple length to); the W + D coupling of the windowed
    # ``_cases`` is vacuous here because W = 0 and every drawn row is already a defined output.
    return draw(st.lists(bars, min_size=0, max_size=SERIES_MAX))


def apply_true_range(
    high: Sequence[float | None],
    low: Sequence[float | None],
    close: Sequence[float | None],
) -> list[float | None]:
    """
    Materialize ``true_range`` over a three-column ``Float64`` frame built from the aligned ``high`` / ``low`` /
    ``close`` lists.
    """
    return materialize({HIGH: high, LOW: low, CLOSE: close}, true_range(pl.col(HIGH), pl.col(LOW), pl.col(CLOSE)))


class TestTrueRangeContract:
    """
    Type, shape, and lazy/eager guarantees.
    """

    def test_over_partitions_independently(self) -> None:
        """
        Verifies that under ``.over`` the previous-close shift resets per group and never spans group boundaries.
        """
        frame = pl.DataFrame(
            {
                GROUP_KEY: ["a", "a", "a", "b", "b", "b"],
                HIGH: [10.0, 12.0, 11.0, 20.0, 22.0, 21.0],
                LOW: [9.0, 10.5, 10.0, 19.0, 20.5, 20.0],
                CLOSE: [9.5, 11.0, 10.5, 19.5, 21.0, 20.5],
            }
        )
        result = frame.select(true_range(pl.col(HIGH), pl.col(LOW), pl.col(CLOSE)).over(GROUP_KEY).alias("y"))[
            "y"
        ].to_list()
        assert_matches(result, [1.0, 2.5, 1.0, 1.0, 2.5, 1.0])


class TestTrueRangeEdge:
    """
    Boundaries and null / NaN handling.
    """

    def test_first_row_is_high_minus_low(self) -> None:
        """
        Verifies the StockCharts/Wilder convention: with no previous close the first row is ``high - low``.
        """
        assert_matches(apply_true_range([10.0, 12.0], [9.0, 10.5], [9.5, 11.0]), [1.0, 2.5])

    def test_single_row(self) -> None:
        """
        Verifies that a one-row series resolves to ``high - low`` (no previous close exists).
        """
        assert_matches(apply_true_range([10.0], [9.0], [9.5]), [1.0])

    def test_constant_series_is_zero(self) -> None:
        """
        Verifies that a flat series with ``high == low == close`` everywhere has zero True Range on every row.
        """
        assert_matches(apply_true_range([5.0, 5.0, 5.0], [5.0, 5.0, 5.0], [5.0, 5.0, 5.0]), [0.0, 0.0, 0.0])

    def test_high_equals_low_uses_gap_terms(self) -> None:
        """
        Verifies that when ``high == low`` (zero bar spread) the gap-to-previous-close terms drive the range: with
        ``close == [10, 8, 12]`` row 0 has no previous close (TR = 0), row 1's previous close is ``10`` (still 0), and
        row 2's previous close is ``8`` so the gap ``|10 - 8| = 2`` surfaces. Cross-checked against the oracle so a
        future arithmetic slip cannot pass review.
        """
        high = [10.0, 10.0, 10.0]
        low = [10.0, 10.0, 10.0]
        close = [10.0, 8.0, 12.0]
        expected = true_range_reference(high, low, close)
        assert_matches(expected, [0.0, 0.0, 2.0])
        assert_matches(apply_true_range(high, low, close), expected)

    def test_previous_close_null_falls_back(self) -> None:
        """
        Verifies that a ``null`` previous close drops the two gap terms and the row falls back to ``high - low``.
        """
        assert_matches(
            apply_true_range([10.0, 12.0, 11.0, 13.0], [9.0, 10.5, 10.0, 11.0], [9.5, None, 10.5, 12.0]),
            [1.0, 2.5, 1.0, 2.5],
        )

    def test_null_high_drops_candidate(self) -> None:
        """
        Verifies that a ``null`` ``high`` drops the candidates it appears in and the row resolves from the survivors.
        """
        assert_matches(
            apply_true_range([10.0, None, 11.0], [9.0, 10.5, 10.0], [9.5, 11.0, 10.5]),
            [1.0, 1.0, 1.0],
        )

    def test_all_inputs_null_at_row_yields_null(self) -> None:
        """
        Verifies that a row is ``null`` only when every candidate drops (``high``, ``low``, and previous close null).
        """
        assert_matches(
            apply_true_range([10.0, None, 11.0], [9.0, None, 10.0], [9.5, None, 10.5]),
            [1.0, None, 1.0],
        )

    def test_all_null_columns(self) -> None:
        """
        Verifies that with all-``null`` ``high`` and ``low`` every row is ``null`` regardless of ``close``.
        """
        assert_matches(apply_true_range([None, None, None], [None, None, None], [9.5, 10.0, 10.5]), [None, None, None])

    def test_nan_in_high_dominates_its_row(self) -> None:
        """
        Verifies that a ``NaN`` ``high`` makes its row ``NaN`` (``NaN`` dominates the max) but does not latch.
        """
        assert_matches(
            apply_true_range([10.0, math.nan, 11.0, 13.0], [9.0, 10.5, 10.0, 11.0], [9.5, 11.0, 10.5, 12.0]),
            [1.0, math.nan, 1.0, 2.5],
        )

    def test_nan_in_close_contaminates_next_row_only(self) -> None:
        """
        Verifies that a ``NaN`` ``close`` is finite at its own row but poisons the next row's gap terms (``NaN``).
        """
        assert_matches(
            apply_true_range([10.0, 12.0, 11.0, 13.0], [9.0, 10.5, 10.0, 11.0], [9.5, math.nan, 10.5, 12.0]),
            [1.0, 2.5, math.nan, 2.5],
        )

    def test_null_takes_precedence_over_nan_when_alone(self) -> None:
        """
        Verifies the null-vs-NaN distinction: a dropped (``null``) candidate is skipped, a ``NaN`` candidate is not.
        """
        result_null = apply_true_range([10.0, None], [9.0, None], [9.5, None])
        result_nan = apply_true_range([10.0, math.nan], [9.0, math.nan], [9.5, math.nan])
        assert_matches(result_null, [1.0, None])
        assert_matches(result_nan, [1.0, math.nan])


class TestTrueRangeCorrectness:
    """
    Against the naive reference oracle and frozen golden-master values.
    """

    def test_matches_reference(self) -> None:
        """
        Verifies agreement with the naive closed-form reference over a representative OHLC series.
        """
        high = [10.0, 12.0, 11.0, 13.0, 14.0, 13.5, 15.0, 14.0]
        low = [9.0, 10.5, 10.0, 11.0, 12.5, 12.0, 13.5, 13.0]
        close = [9.5, 11.0, 10.5, 12.0, 13.0, 12.5, 14.5, 13.5]
        assert_matches(
            apply_true_range(high, low, close),
            true_range_reference(high, low, close),
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference: True Range over a five-bar OHLC series.
        """
        assert_matches(
            apply_true_range(
                [10.0, 12.0, 11.5, 13.0, 12.5],
                [9.0, 10.5, 10.0, 11.0, 11.5],
                [9.5, 11.0, 10.5, 12.5, 12.0],
            ),
            [1.0, 2.5, 1.5, 2.5, 1.0],
        )


class TestTrueRangeProperties:
    """
    Invariants that must hold for all inputs (property-based).
    """

    # NOTE: exact transform -- implementation and oracle compute identical arithmetic, residual is zero, so a fixed
    # reference band applies here (not input_scale-sized like the sum-based degree-1 kernels).
    @given(case=_cases(coherent_hlc()))
    def test_matches_reference_for_any_input(
        self,
        case: list[tuple[float, float, float]],
    ) -> None:
        """
        Verifies that, for any aligned ``high`` / ``low`` / ``close`` series, the implementation matches the naive
        reference.
        """
        rows = case
        high, low, close = split_triples(rows)
        assert_matches(
            apply_true_range(high, low, close),
            true_range_reference(high, low, close),
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    @given(case=_cases(coherent_hlc()))
    def test_non_negative_on_valid_ohlc(
        self,
        case: list[tuple[float, float, float]],
    ) -> None:
        """
        Verifies that True Range is non-negative on every bar of a coherent OHLC series (the valid-OHLC regime).
        """
        rows = case
        high, low, close = split_triples(rows)
        result = apply_true_range(high, low, close)
        for value in result:
            assert value is not None
            assert value >= 0.0

    @given(
        case=_cases(coherent_hlc()),
        exponent=st.sampled_from([-4, -3, -2, -1, 1, 2, 3, 4]),
    )
    def test_scale_homogeneity(
        self,
        case: list[tuple[float, float, float]],
        exponent: int,
    ) -> None:
        """
        Verifies that ``true_range`` is homogeneous of degree 1: scaling every input value by a constant ``k``
        scales the output by the same ``k`` -- ``true_range(k * x) == k * true_range(x)``. ``k`` is a power of two,
        so the rescale is exact and adds no floating-point error.
        """
        k = 2.0**exponent
        rows = case
        high, low, close = split_triples(rows)
        result_base = apply_true_range(high, low, close)
        result_scaled = apply_true_range(
            [value * k for value in high],
            [value * k for value in low],
            [value * k for value in close],
        )
        assert_scale_homogeneous(result_scaled, result_base, k=k, degree=1)

    @given(case=_cases(coherent_hlc_with_missing()))
    def test_matches_reference_under_missing_data(
        self,
        case: list[tuple[float | None, float | None, float | None]],
    ) -> None:
        """
        Verifies that, for inputs freely mixing null / NaN / finite, the implementation matches the naive reference.
        """
        rows = case
        high = [high_value for high_value, _, _ in rows]
        low = [low_value for _, low_value, _ in rows]
        close = [close_value for _, _, close_value in rows]
        assert_matches(
            apply_true_range(high, low, close),
            true_range_reference(high, low, close),
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    @given(
        case=_cases(coherent_hlc()),
        scale=st.sampled_from([1e-6, 1e6, 1e9]),
    )
    def test_matches_reference_at_large_magnitude(
        self,
        case: list[tuple[float, float, float]],
        scale: float,
    ) -> None:
        """
        Verifies that at extreme positive magnitudes the implementation stays finite where the reference is and agrees.
        """
        rows = case
        high = [high_value * scale for high_value, _, _ in rows]
        low = [low_value * scale for _, low_value, _ in rows]
        close = [close_value * scale for _, _, close_value in rows]
        assert_matches(
            apply_true_range(high, low, close),
            true_range_reference(high, low, close),
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )
