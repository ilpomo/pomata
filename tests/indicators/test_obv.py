"""
Tests for ``pomata.indicators.obv`` — On-Balance Volume.

``obv`` is multi-input (close and volume) and has no window, so tests build the two-column frame inline rather than
using the single-input ``apply_expr`` helper; ``assert_matches`` and the naive ``obv_reference`` oracle are shared.

The ladder is the canonical one: contract (type / shape / lazy-eager / ``.over`` parity), edge (empty / single-row /
no-warm-up / null / NaN), correctness (vs the closed-form reference and a frozen golden master), and properties
(reference agreement, volume scale-homogeneity, price-shift invariance, and the per-bar step bound).
"""

import math
from collections.abc import Sequence

import polars as pl
from hypothesis import given
from hypothesis import strategies as st
from tests.indicators.oracles import obv_reference
from tests.support import (
    ABSOLUTE_TOLERANCE_REFERENCE,
    CLOSE,
    EXACT_TOLERANCE_FACTOR,
    GROUP_KEY,
    RELATIVE_TOLERANCE_PROPERTY,
    RELATIVE_TOLERANCE_SCALE,
    VOLUME,
    assert_matches,
    assert_scale_homogeneous,
    input_scale,
    materialize,
    missing_data_floats,
    split_pairs,
)

from pomata.indicators import obv

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- derived, not chosen; the rationale is the shared method in CORRECTNESS.md, the numbers are this
# indicator's. To add an indicator, set its three facts here; the property tier below is then the same shape as every
# other indicator's.
#   1. warmup  W = 0   (windowless cumulative line: every row is defined, starting at ``0`` on the first row, which has
#              no predecessor and so contributes a zero direction)
#   2. memory  the oracle shares pomata's running cumulative sum, so the property holds from row 0 (M = 0); with W = 0
#              there is no warm-up to outlast, so a case is simply a series of bars -- every row is output
#   3. domain  finite (close, volume) pairs drawn per tier over its regime (any-input / scale / shift / large-mag),
#              volume strictly positive except the step tier (non-negative, to reach the zero-volume bar); the
#              missing-data tier draws pairs that freely mix null / NaN / finite. SERIES_MAX bars span several sizes
# OBV is homogeneous of degree 1 in volume (each bar adds +/- its whole volume) and invariant to an additive price
# shift, so it carries a degree-1 volume-scale-homogeneity property and a large-magnitude tier. obv has no window
# parameter, so ``_cases`` draws only the series (no window to couple). Repetitions N are the shared CI profile
# (tests/conftest.py); override per-test only if its parameter space is larger.
# ----------------------------------------------------------------------------------------------------------------------
SERIES_MAX = 50


@st.composite
def _cases[T](draw: st.DrawFn, bars: st.SearchStrategy[T]) -> list[T]:
    """
    A series of bars sized from the facts above. obv is windowless (W = 0), so -- unlike the windowed indicators'
    ``(series, window)`` pair -- a case is just the series: every row is output, never warm-up.
    """
    # NOTE: windowless -- returns the bare series (no window to couple length to); the W + D coupling of the windowed
    # ``_cases`` is vacuous here because W = 0 and every drawn row is already a defined output.
    return draw(st.lists(bars, min_size=1, max_size=SERIES_MAX))


def apply_obv(
    close_values: Sequence[float | None],
    volume_values: Sequence[float | None],
) -> list[float | None]:
    """
    Materialize ``obv`` over a two-column ``Float64`` frame built from ``close_values`` and ``volume_values``.

    Mirrors the single-input ``apply_expr`` helper for this multi-input indicator: it names the two columns, applies the
    factory to ``pl.col`` references, and returns the output as a plain Python list for ``assert_matches``.
    """
    return materialize({CLOSE: close_values, VOLUME: volume_values}, obv(pl.col(CLOSE), pl.col(VOLUME)))


class TestObvContract:
    """
    Type, shape, and lazy/eager guarantees.
    """

    def test_over_partitions_independently(self) -> None:
        """
        Verifies that under ``.over`` both the diff and the cumulative sum reset per group and never span boundaries.
        """
        close_a = [10.0, 12.0, 11.0, 11.0]
        volume_a = [100.0, 200.0, 150.0, 80.0]
        close_b = [5.0, 4.0, 6.0, 6.0]
        volume_b = [50.0, 60.0, 70.0, 30.0]
        frame = pl.DataFrame(
            {
                GROUP_KEY: ["a"] * 4 + ["b"] * 4,
                CLOSE: close_a + close_b,
                VOLUME: volume_a + volume_b,
            }
        )
        result_over = frame.select(obv(pl.col(CLOSE), pl.col(VOLUME)).over(GROUP_KEY).alias("y"))["y"].to_list()
        assert_matches(result_over, apply_obv(close_a, volume_a) + apply_obv(close_b, volume_b))


class TestObvEdge:
    """
    Boundaries, no-warm-up, and null / NaN handling.
    """

    def test_single_row_starts_at_zero(self) -> None:
        """
        Verifies that a one-element series yields ``0`` (the first bar has no predecessor, so its direction is ``0``).
        """
        assert_matches(apply_obv([42.0], [10.0]), [0.0])

    def test_starts_at_zero(self) -> None:
        """
        Verifies that the series starts at ``0`` regardless of the first volume.
        """
        result = apply_obv([10.0, 12.0, 11.0], [100.0, 200.0, 150.0])
        assert result[0] == 0.0

    def test_no_warmup(self) -> None:
        """
        Verifies that there is no warm-up: every row is defined (no leading nulls).
        """
        result = apply_obv([10.0, 12.0, 11.0, 11.0, 13.0], [100.0, 200.0, 150.0, 80.0, 300.0])
        assert all(value is not None for value in result)

    def test_constant_price_is_flat(self) -> None:
        """
        Verifies that a constant price never moves the total (every direction is ``0``).
        """
        assert_matches(apply_obv([5.0, 5.0, 5.0, 5.0], [10.0, 20.0, 30.0, 40.0]), [0.0, 0.0, 0.0, 0.0])

    def test_zero_volume_is_flat(self) -> None:
        """
        Verifies that all-zero volume leaves the total at ``0`` regardless of price direction.
        """
        assert_matches(apply_obv([10.0, 12.0, 11.0], [0.0, 0.0, 0.0]), [0.0, 0.0, 0.0])

    def test_negative_prices_sign_correct(self) -> None:
        """
        Verifies that the direction tracks the sign of the change even for negative prices.
        """
        assert_matches(apply_obv([-5.0, -3.0, -7.0], [10.0, 20.0, 30.0]), [0.0, 20.0, -10.0])

    def test_all_null_price_is_flat(self) -> None:
        """
        Verifies that an all-null price gives a zero direction everywhere, so the total stays at ``0``.
        """
        assert_matches(apply_obv([None, None, None], [1.0, 2.0, 3.0]), [0.0, 0.0, 0.0])

    def test_all_null_volume_is_null(self) -> None:
        """
        Verifies that an all-null volume makes every contribution ``null``.
        """
        assert_matches(apply_obv([10.0, 12.0, 11.0], [None, None, None]), [None, None, None])

    def test_null_price_zeroes_two_bars(self) -> None:
        """
        Verifies that a ``null`` close zeroes the direction at its own row and at the following row (both diffs null).
        """
        assert_matches(
            apply_obv([10.0, 12.0, None, 14.0, 13.0], [100.0, 200.0, 300.0, 400.0, 500.0]),
            obv_reference([10.0, 12.0, None, 14.0, 13.0], [100.0, 200.0, 300.0, 400.0, 500.0]),
        )

    def test_null_volume_is_null_at_row_then_continues(self) -> None:
        """
        Verifies that a ``null`` volume yields ``null`` at exactly that row while the cumulative sum carries on.
        """
        assert_matches(
            apply_obv([10.0, 12.0, 11.0, 14.0, 13.0], [100.0, 200.0, None, 400.0, 500.0]),
            [0.0, 200.0, None, 600.0, 100.0],
        )

    def test_nan_price_latches(self) -> None:
        """
        Verifies that a ``NaN`` close poisons the direction and latches the running total to ``NaN`` thereafter.
        """
        assert_matches(
            apply_obv([1.0, math.nan, 3.0, 4.0, 5.0], [10.0, 20.0, 30.0, 40.0, 50.0]),
            [0.0, math.nan, math.nan, math.nan, math.nan],
        )

    def test_nan_volume_latches(self) -> None:
        """
        Verifies that a ``NaN`` volume poisons its contribution and latches the running total to ``NaN`` thereafter.
        """
        assert_matches(
            apply_obv([10.0, 12.0, 11.0, 13.0, 9.0], [100.0, math.nan, 200.0, 300.0, 400.0]),
            [0.0, math.nan, math.nan, math.nan, math.nan],
        )

    def test_nan_volume_on_flat_bar_still_latches(self) -> None:
        """
        Verifies that a ``NaN`` volume latches even on a flat bar, since ``0 * NaN`` is ``NaN`` under IEEE-754.
        """
        assert_matches(
            apply_obv([5.0, 5.0, 5.0], [10.0, math.nan, 30.0]),
            [0.0, math.nan, math.nan],
        )

    def test_null_volume_on_first_bar_is_null(self) -> None:
        """
        Verifies that a ``null`` volume on the first bar yields ``null`` there (``0 * null`` is ``null``).
        """
        assert_matches(
            apply_obv([10.0, 12.0, 11.0], [None, 200.0, 300.0]),
            obv_reference([10.0, 12.0, 11.0], [None, 200.0, 300.0]),
        )


class TestObvCorrectness:
    """
    Against the naive reference oracle and frozen golden-master values.
    """

    def test_matches_reference(self) -> None:
        """
        Verifies agreement with the naive closed-form reference on a mixed series.
        """
        close_values = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]
        volume_values = [10.0, 50.0, 20.0, 40.0, 30.0, 70.0, 60.0, 80.0]
        assert_matches(
            apply_obv(close_values, volume_values),
            obv_reference(close_values, volume_values),
        )

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference: OBV over close [10, 12, 11, 11, 13, 9, 9, 14] with volume
        [100, 200, 150, 80, 300, 250, 90, 400].
        """
        assert_matches(
            apply_obv(
                [10.0, 12.0, 11.0, 11.0, 13.0, 9.0, 9.0, 14.0],
                [100.0, 200.0, 150.0, 80.0, 300.0, 250.0, 90.0, 400.0],
            ),
            [0.0, 200.0, 50.0, 50.0, 350.0, 100.0, 100.0, 500.0],
        )


class TestObvProperties:
    """
    Invariants that must hold for all inputs (property-based).
    """

    @given(
        case=_cases(
            st.tuples(
                st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
                st.floats(min_value=1.0, max_value=1e3, allow_nan=False, allow_infinity=False),
            )
        ),
    )
    def test_matches_reference_for_any_input(
        self,
        case: list[tuple[float, float]],
    ) -> None:
        """
        Verifies that, for any close series and non-negative volume series, the implementation matches the naive
        reference. Volume is drawn strictly positive (negative volume is out of spec).
        """
        close_values, volume_values = split_pairs(case)
        assert_matches(
            apply_obv(close_values, volume_values),
            obv_reference(close_values, volume_values),
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=input_scale(volume_values) * EXACT_TOLERANCE_FACTOR,
        )

    @given(
        case=_cases(
            st.tuples(
                st.sampled_from(["value", "null", "nan"]),
                st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False),
                st.sampled_from(["value", "null", "nan"]),
                st.floats(min_value=0.0, max_value=1e3, allow_nan=False, allow_infinity=False),
            ),
        ),
    )
    def test_matches_reference_with_nulls_and_nans(
        self,
        case: list[tuple[str, float, str, float]],
    ) -> None:
        """
        Verifies that the implementation matches the naive reference even with ``None`` and ``NaN`` injected into both
        the close and the volume, exercising the null pass-through and the NaN-latch propagation in the property layer.
        """
        choices = case
        close_values: list[float | None] = []
        volume_values: list[float | None] = []
        for close_kind, close_number, volume_kind, volume_number in choices:
            close_values.append(None if close_kind == "null" else (math.nan if close_kind == "nan" else close_number))
            volume_values.append(
                None if volume_kind == "null" else (math.nan if volume_kind == "nan" else volume_number)
            )
        assert_matches(
            apply_obv(close_values, volume_values),
            obv_reference(close_values, volume_values),
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=input_scale(volume_values) * EXACT_TOLERANCE_FACTOR,
        )

    @given(
        case=_cases(
            st.tuples(
                st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False),
                st.floats(min_value=1.0, max_value=1e3, allow_nan=False, allow_infinity=False),
            )
        ),
        exponent=st.sampled_from([-4, -3, -2, -1, 1, 2, 3, 4]),
        sign=st.sampled_from([-1.0, 1.0]),
    )
    def test_volume_scale_homogeneity(
        self,
        case: list[tuple[float, float]],
        exponent: int,
        sign: float,
    ) -> None:
        """
        Verifies that ``obv`` is homogeneous of degree 1 in volume: scaling the volume by a constant ``k`` scales
        the output by the same ``k``, while the prices are untouched. ``k`` is a power of two, so the rescale is
        exact and adds no floating-point error.
        """
        k = sign * 2.0**exponent
        close_values, volume_values = split_pairs(case)
        result_base = apply_obv(close_values, volume_values)
        result_scaled = apply_obv(close_values, [value * k for value in volume_values])
        assert_scale_homogeneous(result_scaled, result_base, k=k, degree=1)

    @given(
        case=_cases(
            st.tuples(
                st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False),
                st.floats(min_value=1.0, max_value=1e3, allow_nan=False, allow_infinity=False),
            )
        ),
        shift=st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False),
    )
    def test_additive_shift_invariance(
        self,
        case: list[tuple[float, float]],
        shift: float,
    ) -> None:
        """
        Verifies that ``obv`` is invariant to a common additive shift: adding the same constant to every input value
        leaves the output unchanged, because the shift cancels.
        """
        close_values, volume_values = split_pairs(case)
        close_values = [round(value, 4) for value in close_values]  # realistic price precision: diffs survive the shift
        result_base = apply_obv(close_values, volume_values)
        result_shifted = apply_obv([value + shift for value in close_values], volume_values)
        tolerance = input_scale(volume_values) * EXACT_TOLERANCE_FACTOR
        assert_matches(result_shifted, result_base, rel_tol=RELATIVE_TOLERANCE_SCALE, abs_tol=tolerance)

    @given(
        case=_cases(
            st.tuples(
                st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
                st.floats(min_value=0.0, max_value=1e3, allow_nan=False, allow_infinity=False),
            )
        ),
    )
    def test_step_bounded_by_volume(
        self,
        case: list[tuple[float, float]],
    ) -> None:
        """
        Verifies that each bar-to-bar change in OBV equals exactly ``0`` or ``+/- volume`` of that bar.
        """
        close_values, volume_values = split_pairs(case)
        result = apply_obv(close_values, volume_values)
        previous_total = 0.0
        for index, value in enumerate(result):
            assert value is not None
            step = value - previous_total
            assert math.isclose(step, 0.0, abs_tol=ABSOLUTE_TOLERANCE_REFERENCE) or math.isclose(
                abs(step),
                volume_values[index],
                rel_tol=RELATIVE_TOLERANCE_PROPERTY,
                abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
            )
            previous_total = value

    @given(
        case=_cases(
            st.tuples(
                missing_data_floats(),
                missing_data_floats(),
            ),
        ),
    )
    def test_matches_reference_under_missing_data(
        self,
        case: list[tuple[float | None, float | None]],
    ) -> None:
        """
        Verifies that, for inputs freely mixing null / NaN / finite, the implementation matches the naive reference.
        """
        rows = case
        close = [close_value for close_value, _ in rows]
        volume = [volume_value for _, volume_value in rows]
        assert_matches(
            apply_obv(close, volume),
            obv_reference(close, volume),
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=input_scale(volume) * EXACT_TOLERANCE_FACTOR,
        )

    @given(
        case=_cases(
            st.tuples(
                st.floats(min_value=1e-3, max_value=1.0, allow_nan=False, allow_infinity=False),
                st.floats(min_value=1e-3, max_value=1.0, allow_nan=False, allow_infinity=False),
            ),
        ),
        scale=st.sampled_from([1e-6, 1e6, 1e9]),
    )
    def test_matches_reference_at_large_magnitude(
        self,
        case: list[tuple[float, float]],
        scale: float,
    ) -> None:
        """
        Verifies that at extreme positive magnitudes the implementation stays finite where the reference is and agrees.
        """
        rows = case
        close = [close_value * scale for close_value, _ in rows]
        volume = [volume_value * scale for _, volume_value in rows]
        assert_matches(
            apply_obv(close, volume),
            obv_reference(close, volume),
            rel_tol=RELATIVE_TOLERANCE_SCALE,
            abs_tol=input_scale(volume) * EXACT_TOLERANCE_FACTOR,
        )
