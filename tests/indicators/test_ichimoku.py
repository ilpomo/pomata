"""
Tests for ``pomata.indicators.ichimoku`` — the four rolling high-low midpoints of Ichimoku Kinkō Hyō, zero-displaced.

``ichimoku`` is multi-input (``high`` / ``low``) and returns a single struct ``pl.Expr`` with the fields ``tenkan`` /
``kijun`` / ``senkou_a`` / ``senkou_b``. The local ``apply_ichimoku`` helper materializes each line over a two-column
``Float64`` frame into a dict of lists, so the shared ``assert_matches`` and the naive ``ichimoku_reference`` oracle (an
independent loop over window midpoints) compare line by line.

Every line is a windowed midpoint -- scale-dependent (degree 1), so the large-magnitude tier applies. The defining
guarantee is NO LOOKAHEAD: because each line is aligned to its computation row with zero displacement, a prefix of the
series and the full series must agree on every shared row -- a metamorphic test placed in the contract tier so it gates.
The ladder is otherwise canonical: contract (type / struct schema / shape / lazy-eager / ``.over`` independence /
no-lookahead), edge (per-line warm-up / validation / equal-window degeneracy / flat / null), correctness (oracle +
frozen golden), properties (reference agreement incl. missing data and large magnitude, window-range containment,
degree-1 homogeneity).
Categories are split into classes; cross-cutting categories use markers (see ``tests/README.md``).
"""

import math
from collections.abc import Sequence

import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st
from tests.indicators.oracles import ichimoku_reference
from tests.support import (
    EXACT_TOLERANCE_FACTOR,
    GROUP_KEY,
    HIGH,
    LOW,
    RELATIVE_TOLERANCE_PROPERTY,
    RELATIVE_TOLERANCE_SCALE,
    assert_matches,
    assert_scale_homogeneous,
    coherent_hl,
    coherent_hl_with_missing,
    input_scale,
    materialize_struct,
    split_pairs,
)

from pomata.indicators import ichimoku

FIELDS = ("tenkan", "kijun", "senkou_a", "senkou_b")

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- derived, not chosen; the rationale is the shared method in CORRECTNESS.md, the numbers are this
# indicator's. To add an indicator, set its three facts here; the property tier below is then the same shape as every
# other indicator's.
#   1. warmup  W = window_senkou - 1   (the longest line, senkou_b, is first defined at index ``window_senkou - 1``;
#              tenkan / kijun / senkou_a warm up sooner)
#   2. memory  M = 0: the oracle recomputes the same windowed midpoints, so over the same input it agrees from the first
#              defined row; each example carries a window-plus of defined bars on every line
#   3. domain  coherent_hl(): coherent (high >= low) positive-finite bars
# Windows span 1 .. WINDOW_MAX, drawn ordered (tenkan <= kijun <= senkou). Repetitions N are the shared CI profile.
# ----------------------------------------------------------------------------------------------------------------------
WINDOW_MAX = 12


@st.composite
def _cases[T](draw: st.DrawFn, bars: st.SearchStrategy[T]) -> tuple[list[T], int, int, int]:
    """
    A (series, window_tenkan, window_kijun, window_senkou) tuple sized from the facts above, with the windows drawn in
    non-decreasing order and the length the longest warm-up plus a window of defined bars.
    """
    window_tenkan = draw(st.integers(min_value=1, max_value=WINDOW_MAX))
    window_kijun = draw(st.integers(min_value=window_tenkan, max_value=WINDOW_MAX))
    window_senkou = draw(st.integers(min_value=window_kijun, max_value=WINDOW_MAX))
    defined = draw(st.integers(min_value=window_senkou, max_value=2 * window_senkou))
    length = (window_senkou - 1) + defined
    return draw(st.lists(bars, min_size=length, max_size=length)), window_tenkan, window_kijun, window_senkou


def apply_ichimoku(
    high: Sequence[float | None],
    low: Sequence[float | None],
    window_tenkan: int,
    window_kijun: int,
    window_senkou: int,
) -> dict[str, list[float | None]]:
    """
    Materialize each line of ``ichimoku`` over a two-column frame, as a dict mirroring the oracle's output.
    """
    return materialize_struct(
        {HIGH: high, LOW: low},
        ichimoku(
            pl.col(HIGH),
            pl.col(LOW),
            window_tenkan=window_tenkan,
            window_kijun=window_kijun,
            window_senkou=window_senkou,
        ),
        FIELDS,
    )


class TestIchimokuContract:
    """
    Type, struct schema, shape, lazy/eager guarantees, and the no-lookahead property.
    """

    def test_output_is_struct_with_named_fields(self) -> None:
        """
        Verifies that the output is a ``Float64`` struct with exactly the fields ``tenkan`` / ``kijun`` / ``senkou_a`` /
        ``senkou_b``.
        """
        frame = pl.DataFrame({HIGH: [3.0, 4.0, 5.0], LOW: [1.0, 2.0, 3.0]})
        dtype = frame.select(
            ichimoku(pl.col(HIGH), pl.col(LOW), window_tenkan=1, window_kijun=2, window_senkou=3).alias("ic")
        ).schema["ic"]
        assert isinstance(dtype, pl.Struct)
        assert [field.name for field in dtype.fields] == ["tenkan", "kijun", "senkou_a", "senkou_b"]
        assert all(field.dtype == pl.Float64 for field in dtype.fields)

    def test_over_partitions_independently(self) -> None:
        """
        Verifies that under ``.over`` no window spans group boundaries.
        """
        frame = pl.DataFrame(
            {
                GROUP_KEY: ["a", "a", "a", "b", "b", "b"],
                HIGH: [3.0, 4.0, 5.0, 13.0, 14.0, 15.0],
                LOW: [1.0, 2.0, 3.0, 11.0, 12.0, 13.0],
            }
        )
        kijun = (
            ichimoku(pl.col(HIGH), pl.col(LOW), window_tenkan=1, window_kijun=2, window_senkou=3)
            .over(GROUP_KEY)
            .struct.field("kijun")
        )
        result = frame.select(kijun.alias("y"))["y"].to_list()
        # kijun warms up over window = 2; group b must not read group a's last bar.
        assert result[0] is None
        assert result[3] is None
        assert result[4] is not None
        assert result[4] > 10.0

    @given(case=_cases(coherent_hl()), data=st.data())
    def test_no_lookahead_prefix_matches_full(
        self,
        case: tuple[list[tuple[float, float]], int, int, int],
        data: st.DataObject,
    ) -> None:
        """
        Verifies the defining guarantee: computing on a prefix of the series gives identical values to the full series
        on every shared row -- so no line reads a future bar (the displacement is zero). This is the metamorphic test
        that distinguishes a backtest-safe Ichimoku from a forward-shifted one.
        """
        rows, window_tenkan, window_kijun, window_senkou = case
        high, low = split_pairs(rows)
        prefix_length = data.draw(st.integers(min_value=1, max_value=len(rows)))
        full = apply_ichimoku(high, low, window_tenkan, window_kijun, window_senkou)
        prefix = apply_ichimoku(high[:prefix_length], low[:prefix_length], window_tenkan, window_kijun, window_senkou)
        for field in FIELDS:
            assert_matches(prefix[field], full[field][:prefix_length])


class TestIchimokuEdge:
    """
    Validation, per-line warm-up, the equal-window degeneracy, the flat window, and null / NaN per input.
    """

    def test_window_tenkan_below_one_raises(self) -> None:
        """
        Verifies that ``window_tenkan < 1`` raises ``ValueError``.
        """
        with pytest.raises(ValueError, match="window_tenkan must be >= 1"):
            ichimoku(pl.col(HIGH), pl.col(LOW), window_tenkan=0, window_kijun=2, window_senkou=3)

    def test_window_kijun_below_one_raises(self) -> None:
        """
        Verifies that ``window_kijun < 1`` raises ``ValueError`` (caught before the ordering check).
        """
        with pytest.raises(ValueError, match="window_kijun must be >= 1"):
            ichimoku(pl.col(HIGH), pl.col(LOW), window_tenkan=1, window_kijun=0, window_senkou=3)

    def test_window_senkou_below_one_raises(self) -> None:
        """
        Verifies that a ``window_senkou < 1`` (forcing all three below one) raises ``ValueError``.
        """
        with pytest.raises(ValueError, match="window_senkou must be >= 1"):
            ichimoku(pl.col(HIGH), pl.col(LOW), window_tenkan=1, window_kijun=1, window_senkou=0)

    def test_unordered_windows_raise(self) -> None:
        """
        Verifies that windows out of non-decreasing order raise ``ValueError`` (both adjacent inversions).
        """
        with pytest.raises(ValueError, match="windows must be ordered window_tenkan <= window_kijun <= window_senkou"):
            ichimoku(pl.col(HIGH), pl.col(LOW), window_tenkan=5, window_kijun=3, window_senkou=7)
        with pytest.raises(ValueError, match="windows must be ordered window_tenkan <= window_kijun <= window_senkou"):
            ichimoku(pl.col(HIGH), pl.col(LOW), window_tenkan=2, window_kijun=9, window_senkou=7)

    def test_all_null(self) -> None:
        """
        Verifies that an all-null input yields an all-null output on every line.
        """
        lines = apply_ichimoku([None, None, None], [None, None, None], 1, 2, 3)
        for field in FIELDS:
            assert_matches(lines[field], [None, None, None])

    def test_per_line_warmup_counts(self) -> None:
        """
        Verifies the four distinct warm-ups: ``tenkan`` after ``window_tenkan - 1`` nulls, ``kijun`` and ``senkou_a``
        after ``window_kijun - 1`` (the latter needs both), ``senkou_b`` after ``window_senkou - 1``.
        """
        high = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
        low = [9.0, 10.0, 11.0, 12.0, 13.0, 14.0]
        lines = apply_ichimoku(high, low, 2, 3, 4)
        assert lines["tenkan"][0] is None
        assert lines["tenkan"][1] is not None
        assert lines["kijun"][1] is None
        assert lines["kijun"][2] is not None
        assert lines["senkou_a"][1] is None
        assert lines["senkou_a"][2] is not None
        assert lines["senkou_b"][2] is None
        assert lines["senkou_b"][3] is not None

    def test_equal_windows_collapse_lines(self) -> None:
        """
        Verifies the degenerate equal-window case: with ``window_tenkan == window_kijun`` the two lines coincide, so
        ``senkou_a`` (their midpoint) equals ``tenkan`` exactly.
        """
        high = [10.0, 12.0, 11.0, 13.0, 14.0]
        low = [8.0, 9.0, 10.0, 11.0, 12.0]
        lines = apply_ichimoku(high, low, 2, 2, 3)
        assert_matches(lines["kijun"], lines["tenkan"])
        assert_matches(lines["senkou_a"], lines["tenkan"])

    def test_single_row(self) -> None:
        """
        Verifies a one-element series: all windows ``== 1`` give the bar's midprice on every line, a larger window is
        all warm-up.
        """
        for field in FIELDS:
            assert_matches(apply_ichimoku([11.0], [9.0], 1, 1, 1)[field], [10.0])
            assert_matches(apply_ichimoku([11.0], [9.0], 2, 3, 4)[field], [None])

    def test_window_exceeds_length(self) -> None:
        """
        Verifies that windows longer than the series yield an all-null result on every line.
        """
        lines = apply_ichimoku([10.0, 11.0, 12.0], [9.0, 10.0, 11.0], 4, 5, 6)
        for field in FIELDS:
            assert_matches(lines[field], [None, None, None])

    def test_flat_window_equals_price(self) -> None:
        """
        Verifies the flat window: over a constant series the high and low extremes coincide, so every line equals the
        price once warmed up.
        """
        flat = [7.0] * 6
        lines = apply_ichimoku(flat, flat, 2, 3, 4)
        assert_matches(lines["tenkan"], [None, 7.0, 7.0, 7.0, 7.0, 7.0])
        assert_matches(lines["senkou_b"], [None, None, None, 7.0, 7.0, 7.0])

    def test_nan_propagates(self) -> None:
        """
        Verifies that a ``NaN`` in ``high`` or ``low`` flows through the rolling extremes exactly as the reference,
        nanning the windows it reaches, then recovering.
        """
        high = [10.0, math.nan, 12.0, 13.0, 14.0, 15.0, 16.0]
        low = [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
        lines = apply_ichimoku(high, low, 2, 3, 4)
        reference = ichimoku_reference(high, low, 2, 3, 4)
        for field in FIELDS:
            assert_matches(lines[field], reference[field])

    def test_null_in_high_vs_low(self) -> None:
        """
        Verifies that a ``null`` / ``NaN`` in ``high`` only, and in ``low`` only, both flow through the rolling extremes
        exactly as the reference (the inputs enter different legs of each midpoint).
        """
        high = [10.0, None, 12.0, 13.0, math.nan, 15.0, 16.0]
        low = [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
        lines = apply_ichimoku(high, low, 2, 3, 4)
        reference = ichimoku_reference(high, low, 2, 3, 4)
        for field in FIELDS:
            assert_matches(lines[field], reference[field])


class TestIchimokuCorrectness:
    """
    Against the naive reference oracle and a frozen golden master.
    """

    def test_matches_reference(self) -> None:
        """
        Verifies that every line agrees with the naive reference across several window triples.
        """
        high = [10.0, 12.0, 11.0, 13.0, 14.0, 12.0, 15.0, 13.0, 16.0, 14.0]
        low = [8.0, 9.0, 10.0, 11.0, 12.0, 10.0, 12.0, 11.0, 13.0, 12.0]
        for windows in ((1, 1, 1), (2, 3, 4), (2, 2, 5), (3, 4, 6)):
            lines = apply_ichimoku(high, low, *windows)
            reference = ichimoku_reference(high, low, *windows)
            for field in FIELDS:
                assert_matches(lines[field], reference[field])

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference, hand-derived for windows ``2 / 3 / 4`` over an eight-bar series: e.g. ``kijun``
        at row 2 is ``(max(10, 12, 11) + min(8, 9, 10)) / 2 = (12 + 8) / 2 = 10``, and ``senkou_a`` at row 2 is
        ``(tenkan + kijun) / 2 = (10.5 + 10) / 2 = 10.25``.
        """
        high = [10.0, 12.0, 11.0, 13.0, 14.0, 12.0, 15.0, 13.0]
        low = [8.0, 9.0, 10.0, 11.0, 12.0, 10.0, 12.0, 11.0]
        lines = apply_ichimoku(high, low, 2, 3, 4)
        assert_matches(lines["tenkan"], [None, 10.0, 10.5, 11.5, 12.5, 12.0, 12.5, 13.0])
        assert_matches(lines["kijun"], [None, None, 10.0, 11.0, 12.0, 12.0, 12.5, 12.5])
        assert_matches(lines["senkou_a"], [None, None, 10.25, 11.25, 12.25, 12.0, 12.5, 12.75])
        assert_matches(lines["senkou_b"], [None, None, None, 10.5, 11.5, 12.0, 12.5, 12.5])


class TestIchimokuProperties:
    """
    Invariants that must hold for all inputs (property-based).
    """

    @given(case=_cases(coherent_hl()))
    def test_matches_reference_for_any_input(
        self,
        case: tuple[list[tuple[float, float]], int, int, int],
    ) -> None:
        """
        Verifies that, for any coherent high/low series and ordered windows, every line matches the naive reference.
        """
        rows, window_tenkan, window_kijun, window_senkou = case
        high, low = split_pairs(rows)
        lines = apply_ichimoku(high, low, window_tenkan, window_kijun, window_senkou)
        reference = ichimoku_reference(high, low, window_tenkan, window_kijun, window_senkou)
        for field in FIELDS:
            assert_matches(
                lines[field],
                reference[field],
                rel_tol=RELATIVE_TOLERANCE_PROPERTY,
                abs_tol=input_scale(high) * EXACT_TOLERANCE_FACTOR,
            )

    @given(case=_cases(coherent_hl()))
    def test_lines_lie_within_their_window_range(
        self,
        case: tuple[list[tuple[float, float]], int, int, int],
    ) -> None:
        """
        Verifies the containment invariant: each line lies within ``[rolling_min(low, w), rolling_max(high, w)]`` of its
        own window (a midpoint cannot leave the range it averages); ``senkou_a`` uses the wider ``kijun`` range.
        """
        rows, window_tenkan, window_kijun, window_senkou = case
        high, low = split_pairs(rows)
        lines = apply_ichimoku(high, low, window_tenkan, window_kijun, window_senkou)
        frame = pl.DataFrame({HIGH: pl.Series(high, dtype=pl.Float64), LOW: pl.Series(low, dtype=pl.Float64)})
        windows = {"tenkan": window_tenkan, "kijun": window_kijun, "senkou_a": window_kijun, "senkou_b": window_senkou}
        for field, window in windows.items():
            bounds = frame.select(
                pl.col(LOW).rolling_min(window).alias("floor"),
                pl.col(HIGH).rolling_max(window).alias("ceiling"),
            )
            floor = bounds["floor"].to_list()
            ceiling = bounds["ceiling"].to_list()
            for value, low_bound, high_bound in zip(lines[field], floor, ceiling, strict=True):
                if value is None or math.isnan(value):
                    continue
                assert low_bound is not None
                assert high_bound is not None
                assert low_bound - 1e-9 <= value <= high_bound + 1e-9

    @given(
        case=_cases(coherent_hl()),
        exponent=st.sampled_from([-4, -3, -2, -1, 1, 2, 3, 4]),
    )
    def test_scale_homogeneity(
        self,
        case: tuple[list[tuple[float, float]], int, int, int],
        exponent: int,
    ) -> None:
        """
        Verifies that ``ichimoku`` is homogeneous of degree 1: scaling every input value by a constant ``k`` scales
        the output by the same ``k`` -- ``ichimoku(k * x) == k * ichimoku(x)``. ``k`` is a power of two, so the
        rescale is exact and adds no floating-point error.
        """
        k = 2.0**exponent
        rows, window_tenkan, window_kijun, window_senkou = case
        high, low = split_pairs(rows)
        base = apply_ichimoku(high, low, window_tenkan, window_kijun, window_senkou)
        scaled = apply_ichimoku([v * k for v in high], [v * k for v in low], window_tenkan, window_kijun, window_senkou)
        for field in FIELDS:
            assert_scale_homogeneous(scaled[field], base[field], k=k, degree=1)

    @given(case=_cases(coherent_hl_with_missing()))
    def test_matches_reference_under_missing_data(
        self,
        case: tuple[list[tuple[float | None, float | None]], int, int, int],
    ) -> None:
        """
        Verifies that, for inputs freely mixing null / NaN / finite, every line matches the naive reference.
        """
        rows, window_tenkan, window_kijun, window_senkou = case
        high, low = split_pairs(rows)
        lines = apply_ichimoku(high, low, window_tenkan, window_kijun, window_senkou)
        reference = ichimoku_reference(high, low, window_tenkan, window_kijun, window_senkou)
        for field in FIELDS:
            assert_matches(
                lines[field],
                reference[field],
                rel_tol=RELATIVE_TOLERANCE_PROPERTY,
                abs_tol=input_scale(high) * EXACT_TOLERANCE_FACTOR,
            )

    @given(
        case=_cases(coherent_hl()),
        scale=st.sampled_from([1e-6, 1e6, 1e9]),
    )
    def test_matches_reference_at_large_magnitude(
        self,
        case: tuple[list[tuple[float, float]], int, int, int],
        scale: float,
    ) -> None:
        """
        Verifies that at extreme positive magnitudes every line stays finite where the reference is and agrees.
        """
        rows, window_tenkan, window_kijun, window_senkou = case
        high = [row[0] * scale for row in rows]
        low = [row[1] * scale for row in rows]
        lines = apply_ichimoku(high, low, window_tenkan, window_kijun, window_senkou)
        reference = ichimoku_reference(high, low, window_tenkan, window_kijun, window_senkou)
        for field in FIELDS:
            assert_matches(
                lines[field],
                reference[field],
                rel_tol=RELATIVE_TOLERANCE_SCALE,
                abs_tol=input_scale(high) * EXACT_TOLERANCE_FACTOR,
            )
