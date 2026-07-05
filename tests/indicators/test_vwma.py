"""
Tests for ``pomata.indicators.vwma`` — the Volume-Weighted Moving Average (VWMA).

``vwma`` is multi-input (``price`` and ``volume``), so this module cannot reuse the single-input ``apply_expr`` helper
and instead defines a local ``apply_vwma`` that builds the two-column ``Float64`` frame inline. The shared
``assert_matches`` comparator and the naive ``vwma_reference`` oracle are reused unchanged.

The ladder is the canonical one: contract (type / shape / lazy-eager / ``.over`` parity), edge (warm-up / window
boundaries / single-row / null / NaN / zero-volume), correctness (vs the closed-form reference and a frozen golden
master), and properties (reference agreement for any input, price scale-homogeneity, volume scale-invariance, and the
convex-combination range bound). Categories are split into classes; cross-cutting categories elsewhere use markers (see
``tests/README.md``).
"""

import math
from collections.abc import Sequence

import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st
from tests.indicators.oracles import vwma_reference
from tests.support import (
    ABSOLUTE_TOLERANCE_REFERENCE,
    CLOSE,
    EXACT_TOLERANCE_FACTOR,
    GROUP_KEY,
    RELATIVE_TOLERANCE_PROPERTY,
    RELATIVE_TOLERANCE_REFERENCE,
    RELATIVE_TOLERANCE_SCALE,
    VOLUME,
    WINDOW_MAX,
    assert_matches,
    assert_scale_homogeneous,
    input_scale,
    materialize,
    missing_data_floats,
    split_pairs,
)

from pomata.indicators import vwma

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- derived, not chosen; the rationale is the shared method in CORRECTNESS.md, the numbers are this
# indicator's. To add an indicator, set its three facts here; the property tier below is then the same shape as every
# other indicator's.
#   1. warmup  W(window) = window - 1   (the window must hold ``window`` non-null bars before a result is emitted)
#   2. memory  the oracle shares pomata's windowed seeding, so the property holds from the first defined row (M = 0);
#              each example carries D in [window, 2 * window] defined bars -- one window of output, never all warm-up
#   3. domain  finite (price, volume) pairs drawn per tier over its regime (any-input / scale / range / large-mag),
#              volume strictly positive (a near-zero total volume invites the ``0 / 0`` boundary, which the edge tests
#              pin directly); the missing-data tier draws pairs that freely mix null / NaN / finite. Windows span 1 ..
#              WINDOW_MAX
# VWMA is homogeneous of degree 1 in price (a convex combination of the window's prices) and invariant to a positive
# common rescaling of volume, so it carries a degree-1 price-scale-homogeneity property and a large-magnitude tier.
# Repetitions N are the shared CI profile (tests/conftest.py); override per-test only if its parameter space is larger.
# ----------------------------------------------------------------------------------------------------------------------


def _missing_data_volume() -> st.SearchStrategy[float | None]:
    """
    The volume counterpart of :func:`missing_data_floats` for the missing-data tier, constrained non-negative.

    Volume is non-negative by definition, and every other vwma tier draws it strictly positive; mixing in negative
    volumes would invite a near-canceling window whose total is a sub-ULP residual, where the implementation and the
    oracle legitimately disagree on the ``+/-inf`` versus finite classification — a floating-point artifact, not a bug.
    Drawing volume from ``[0, 1e6]`` (still freely interleaved with ``null`` and ``NaN``) keeps the genuine zero-volume
    ``0 / 0`` degenerate exercised while ruling out that out-of-spec cancellation.
    """
    return st.one_of(
        st.none(),
        st.just(math.nan),
        st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False),
    )


@st.composite
def _cases[T](draw: st.DrawFn, bars: st.SearchStrategy[T]) -> tuple[list[T], int]:
    """
    A (series, window) pair sized from the facts above: ``window`` over its regimes, length = warm-up + a window of
    defined bars, so every example has output to check (never an all-warm-up series, the waste a ``window`` decoupled
    from the length would cause). The property tiers draw (price, volume) pairs here.
    """
    window = draw(st.integers(min_value=1, max_value=WINDOW_MAX))
    defined = draw(st.integers(min_value=window, max_value=2 * window))
    length = (window - 1) + defined
    return draw(st.lists(bars, min_size=length, max_size=length)), window


def apply_vwma(
    price_values: Sequence[float | None],
    volume_values: Sequence[float | None],
    window: int,
) -> list[float | None]:
    """
    Materialize ``vwma`` over a two-column ``Float64`` frame built from ``price_values`` and ``volume_values``.

    Mirrors the single-input ``apply_expr`` helper for this multi-input indicator: it names the two columns, applies the
    factory to ``pl.col`` references, and returns the output as a plain Python list for ``assert_matches``.

    Args:
        price_values: The price observations (may contain ``None`` and ``float('nan')``).
        volume_values: The traded-volume observations (may contain ``None`` and ``float('nan')``); same length as
            ``price_values``.
        window: Number of observations in the moving window. Must be ``>= 1``.

    Returns:
        The materialized VWMA as a Python list of the same length as the inputs, with ``None`` for ``null`` entries.
    """
    return materialize({CLOSE: price_values, VOLUME: volume_values}, vwma(pl.col(CLOSE), pl.col(VOLUME), window))


class TestVwmaContract:
    """
    Type, shape, and lazy/eager guarantees.
    """

    def test_over_partitions_independently(self) -> None:
        """
        Verifies that under ``.over`` the window resets per group and never spans group boundaries.

        With equal volumes per row the VWMA reduces to the SMA of price, so the expected per-group values are the
        running pairwise means within each group.
        """
        frame = pl.DataFrame(
            {
                GROUP_KEY: ["a", "a", "a", "b", "b", "b"],
                CLOSE: [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
                VOLUME: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            }
        )
        result = frame.select(vwma(pl.col(CLOSE), pl.col(VOLUME), 2).over(GROUP_KEY).alias("y"))["y"].to_list()
        assert_matches(result, [None, 1.5, 2.5, None, 15.0, 25.0])


class TestVwmaEdge:
    """
    Boundaries, warm-up, and null / NaN handling.
    """

    def test_window_below_one_raises(self) -> None:
        """
        Verifies that ``window < 1`` raises ``ValueError``.
        """
        with pytest.raises(ValueError, match="window must be >= 1"):
            vwma(pl.col(CLOSE), pl.col(VOLUME), 0)

    def test_warmup_null_count(self) -> None:
        """
        Verifies that the first ``window - 1`` rows are null (warm-up) and the first full window is defined.
        """
        result = apply_vwma([10.0, 11.0, 12.0, 13.0, 14.0], [100.0, 200.0, 300.0, 400.0, 500.0], 3)
        assert result[:2] == [None, None]
        assert result[2] is not None

    def test_all_null(self) -> None:
        """
        Verifies that all-null price and volume series yield an all-null output.
        """
        assert_matches(apply_vwma([None, None, None, None, None], [None, None, None, None, None], 3), [None] * 5)

    def test_window_one_is_identity(self) -> None:
        """
        Verifies that ``window == 1`` with non-zero volume reproduces the price.
        """
        assert_matches(apply_vwma([10.0, 11.0, 12.0], [5.0, 6.0, 7.0], 1), [10.0, 11.0, 12.0])

    def test_window_equals_length(self) -> None:
        """
        Verifies the single defined value when ``window`` equals the series length.
        """
        assert_matches(
            apply_vwma([2.0, 4.0, 6.0], [50.0, 50.0, 50.0], 3),
            vwma_reference([2.0, 4.0, 6.0], [50.0, 50.0, 50.0], 3),
        )

    def test_window_exceeds_length(self) -> None:
        """
        Verifies that a window exceeding the series length yields an all-null output.
        """
        assert_matches(apply_vwma([10.0, 11.0, 12.0], [100.0, 200.0, 300.0], 5), [None, None, None])

    def test_single_row(self) -> None:
        """
        Verifies behavior on a one-element series.
        """
        assert_matches(apply_vwma([42.0], [10.0], 1), [42.0])
        assert_matches(apply_vwma([42.0], [10.0], 3), [None])

    def test_equal_volume_reduces_to_sma(self) -> None:
        """
        Verifies that a constant volume in the window reduces the VWMA to the SMA of price.
        """
        assert_matches(
            apply_vwma([2.0, 4.0, 6.0, 8.0, 10.0], [50.0, 50.0, 50.0, 50.0, 50.0], 3),
            [None, None, 4.0, 6.0, 8.0],
        )

    def test_null_in_price_propagates(self) -> None:
        """
        Verifies that a ``null`` in the price window yields ``null``.
        """
        assert_matches(
            apply_vwma([10.0, None, 12.0, 13.0, 14.0], [100.0, 200.0, 300.0, 400.0, 500.0], 2),
            [None, None, None, 12.571428571428571, 13.555555555555555],
        )

    def test_null_in_volume_propagates(self) -> None:
        """
        Verifies that a ``null`` in the volume window yields ``null``.
        """
        assert_matches(
            apply_vwma([10.0, 11.0, 12.0, 13.0, 14.0], [100.0, None, 300.0, 400.0, 500.0], 2),
            [None, None, None, 12.571428571428571, 13.555555555555555],
        )

    def test_interior_null_propagates(self) -> None:
        """
        Verifies the interior-null case: a ``null`` in price taints exactly the windows that contain it.
        """
        price_values = [2.0, 4.0, None, 8.0, 10.0, 12.0]
        volume_values = [100.0, 200.0, 300.0, 400.0, 500.0, 600.0]
        assert_matches(
            apply_vwma(price_values, volume_values, 2),
            vwma_reference(price_values, volume_values, 2),
        )

    def test_nan_in_price_propagates(self) -> None:
        """
        Verifies that a ``NaN`` in the price window yields ``NaN`` (no ``null`` present).
        """
        assert_matches(
            apply_vwma([10.0, math.nan, 12.0, 13.0], [100.0, 200.0, 300.0, 400.0], 2),
            [None, math.nan, math.nan, 12.571428571428571],
        )

    def test_nan_in_volume_propagates(self) -> None:
        """
        Verifies that a ``NaN`` in the volume window yields ``NaN`` (no ``null`` present).
        """
        assert_matches(
            apply_vwma([10.0, 11.0, 12.0, 13.0], [100.0, math.nan, 300.0, 400.0], 2),
            [None, math.nan, math.nan, 12.571428571428571],
        )

    def test_null_takes_precedence_over_nan(self) -> None:
        """
        Verifies that a window containing both a ``null`` and a ``NaN`` yields ``null`` (``null`` precedence).
        """
        price_values = [10.0, None, 12.0, 13.0]
        volume_values = [100.0, 200.0, math.nan, 400.0]
        assert_matches(
            apply_vwma(price_values, volume_values, 3),
            vwma_reference(price_values, volume_values, 3),
        )

    def test_zero_total_volume_is_nan(self) -> None:
        """
        Verifies that a window whose total volume is zero yields ``NaN`` (IEEE-754 ``0 / 0``).
        """
        assert_matches(apply_vwma([10.0, 11.0, 12.0], [0.0, 0.0, 0.0], 2), [None, math.nan, math.nan])

    def test_zero_volume_window_after_movement_is_nan(self) -> None:
        """
        Verifies that an all-zero-volume window that follows non-zero-volume bars yields ``NaN``, not the ``+/-inf`` a
        sub-ULP residual left in the rolling-sum numerator (by Polars' subtract-on-exit accumulator) would otherwise
        produce. With price ``[10, 50, 90, 1000, 2000, 3000]`` and volume ``[0.1, 1.1, 1.1, 0, 0, 0]`` the window at the
        final row spans the three zero-volume bars, so its result is the ``0 / 0`` degenerate ``NaN``; the two defined
        rows before it carry the ordinary volume-weighted mean.
        """
        assert_matches(
            apply_vwma([10.0, 50.0, 90.0, 1000.0, 2000.0, 3000.0], [0.1, 1.1, 1.1, 0.0, 0.0, 0.0], 3),
            [None, None, 67.3913043478261, 70.0, 90.0, math.nan],
        )


class TestVwmaCorrectness:
    """
    Against the naive reference oracle and frozen golden-master values.
    """

    def test_matches_reference(self) -> None:
        """
        Verifies agreement with the naive closed-form reference across several windows.
        """
        price_values = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]
        volume_values = [10.0, 50.0, 20.0, 40.0, 30.0, 70.0, 60.0, 80.0]
        for window in (1, 2, 3, 4, 5):
            assert_matches(
                apply_vwma(price_values, volume_values, window),
                vwma_reference(price_values, volume_values, window),
            )

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference: VWMA(window=3) over price [10, 11, 12, 13, 14] with volume
        [100, 200, 300, 400, 500] == [None, None, 11.333..., 12.222..., 13.166...].
        """
        assert_matches(
            apply_vwma([10.0, 11.0, 12.0, 13.0, 14.0], [100.0, 200.0, 300.0, 400.0, 500.0], 3),
            [None, None, 11.333333333333334, 12.222222222222221, 13.166666666666666],
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )


class TestVwmaProperties:
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
        case: tuple[list[tuple[float, float]], int],
    ) -> None:
        """
        Verifies that, for any price series, strictly positive volume series, and window, the implementation matches the
        naive reference. Volume is drawn strictly positive (negative volume is out of spec and invites
        float-cancellation flakiness, and a near-zero total invites the ``0 / 0`` boundary).
        """
        rows, window = case
        price_values, volume_values = split_pairs(rows)
        assert_matches(
            apply_vwma(price_values, volume_values, window),
            vwma_reference(price_values, volume_values, window),
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=input_scale(price_values) * EXACT_TOLERANCE_FACTOR,
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
    def test_price_scale_homogeneity(
        self,
        case: tuple[list[tuple[float, float]], int],
        exponent: int,
        sign: float,
    ) -> None:
        """
        Verifies that VWMA is homogeneous of degree 1 in price: ``vwma(k * p, v) == k * vwma(p, v)``. ``k`` is a signed
        power of two so the rescaling is lossless and the shared assertion's ``|k|``-scaled floor never underflows.
        """
        k = sign * 2.0**exponent
        rows, window = case
        price_values, volume_values = split_pairs(rows)
        result_base = apply_vwma(price_values, volume_values, window)
        result_scaled = apply_vwma([value * k for value in price_values], volume_values, window)
        assert_scale_homogeneous(result_scaled, result_base, k=k, degree=1)

    @given(
        case=_cases(
            st.tuples(
                st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False),
                st.floats(min_value=1.0, max_value=1e3, allow_nan=False, allow_infinity=False),
            )
        ),
        exponent=st.sampled_from([-4, -3, -2, -1, 1, 2, 3, 4]),
    )
    def test_volume_scale_invariance(
        self,
        case: tuple[list[tuple[float, float]], int],
        exponent: int,
    ) -> None:
        """
        Verifies that VWMA is invariant to a positive global rescaling of volume: ``vwma(p, c * v) == vwma(p, v)``.
        ``c`` is a power of two so the rescaling is lossless and the shared assertion's floor never underflows.
        """
        c = 2.0**exponent
        rows, window = case
        price_values, volume_values = split_pairs(rows)
        result_base = apply_vwma(price_values, volume_values, window)
        result_scaled = apply_vwma(price_values, [value * c for value in volume_values], window)
        assert_scale_homogeneous(result_scaled, result_base, k=c, degree=0)

    @given(
        case=_cases(
            st.tuples(
                st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False),
                st.floats(min_value=1.0, max_value=1e3, allow_nan=False, allow_infinity=False),
            )
        ),
    )
    def test_within_price_window_range(
        self,
        case: tuple[list[tuple[float, float]], int],
    ) -> None:
        """
        Verifies that with non-negative volume the VWMA is a convex combination of the window's prices and so lies
        within ``[min, max]`` of the price window.
        """
        rows, window = case
        price_values, volume_values = split_pairs(rows)
        result = apply_vwma(price_values, volume_values, window)
        for index, value in enumerate(result):
            if value is None:
                continue
            price_window = price_values[index + 1 - window : index + 1]
            assert min(price_window) - 1e-6 <= value <= max(price_window) + 1e-6

    @given(
        case=_cases(
            st.tuples(
                missing_data_floats(),
                _missing_data_volume(),
            ),
        ),
    )
    def test_matches_reference_under_missing_data(
        self,
        case: tuple[list[tuple[float | None, float | None]], int],
    ) -> None:
        """
        Verifies that, for inputs freely mixing null / NaN / finite, the implementation matches the naive reference.
        Price is drawn freely; volume is non-negative (see ``_missing_data_volume``), the domain the other tiers use.
        """
        rows, window = case
        price = [price_value for price_value, _ in rows]
        volume = [volume_value for _, volume_value in rows]
        assert_matches(
            apply_vwma(price, volume, window),
            vwma_reference(price, volume, window),
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=input_scale(price) * EXACT_TOLERANCE_FACTOR,
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
        case: tuple[list[tuple[float, float]], int],
        scale: float,
    ) -> None:
        """
        Verifies that at extreme positive magnitudes the implementation stays finite where the reference is and agrees.
        """
        rows, window = case
        price = [price_value * scale for price_value, _ in rows]
        volume = [volume_value * scale for _, volume_value in rows]
        assert_matches(
            apply_vwma(price, volume, window),
            vwma_reference(price, volume, window),
            rel_tol=RELATIVE_TOLERANCE_SCALE,
            abs_tol=input_scale(price) * EXACT_TOLERANCE_FACTOR,
        )
