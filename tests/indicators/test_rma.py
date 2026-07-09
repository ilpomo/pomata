"""
Tests for ``pomata.indicators.rma`` — Wilder Moving Average (SMMA / Wilder smoothing / Modified MA).

Categories are split into classes; cross-cutting categories elsewhere use markers (see ``tests/README.md``).
"""

import math

import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st
from tests.indicators.oracles import rma_reference
from tests.support import (
    ABSOLUTE_TOLERANCE_REFERENCE,
    COLUMN_X,
    EXACT_TOLERANCE_FACTOR,
    GROUP_KEY,
    RELATIVE_TOLERANCE_PROPERTY,
    RELATIVE_TOLERANCE_REFERENCE,
    RELATIVE_TOLERANCE_SCALE,
    SUBNORMAL_FLOOR,
    WINDOW_MAX,
    apply_expr,
    assert_matches,
    assert_scale_homogeneous,
    input_scale,
    missing_data_floats,
    subnormal_safe_floats,
)

from pomata.indicators import rma

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- derived, not chosen; the rationale is the shared method in CORRECTNESS.md, the numbers are this
# indicator's. To add an indicator, set its three facts here; the property tier below is then the same shape as every
# other indicator's.
#   1. warmup  W(window) = window - 1   (the recursion is gated by ``min_samples = window`` and seeded at the first
#              non-null observation, so it emits only once ``window`` non-null observations have been counted)
#   2. memory  the oracle shares pomata's recursive Wilder seeding, so the property holds from the first defined row
#              (M = 0); each example carries D in [window, 2 * window] defined values -- one window of output, never all
#              warm-up
#   3. domain  subnormal_safe_floats(bound): finite values floored away from the subnormal range where a recursive
#              mean's ``input_scale * EXACT_TOLERANCE_FACTOR`` abs-tol collapses to 0.0 and it rounds one ULP from the
#              oracle (see the helper); the bound is widened per test below
# Windows span ``window_min`` .. WINDOW_MAX. Repetitions N are the shared CI profile (tests/conftest.py); override
# per-test only if its parameter space is larger.
# ----------------------------------------------------------------------------------------------------------------------


@st.composite
def _cases[T](
    draw: st.DrawFn,
    values: st.SearchStrategy[T],
    window_min: int = 1,
) -> tuple[list[T], int]:
    """
    A (series, window) pair sized from the facts above: ``window`` over its regimes, length = warm-up + a window of
    defined values, so every example has output to check (never an all-warm-up series, the waste a ``window`` decoupled
    from the length would cause).
    """
    window = draw(st.integers(min_value=window_min, max_value=WINDOW_MAX))
    defined = draw(st.integers(min_value=window, max_value=2 * window))
    length = (window - 1) + defined
    return draw(st.lists(values, min_size=length, max_size=length)), window


class TestRmaContract:
    """
    Type, shape, and lazy/eager guarantees.
    """

    def test_over_partitions_independently(self) -> None:
        """
        Verifies that under ``.over`` the recursion resets per group and never spans group boundaries.
        """
        frame = pl.DataFrame({GROUP_KEY: ["a", "a", "a", "b", "b", "b"], COLUMN_X: [1.0, 2.0, 3.0, 10.0, 20.0, 30.0]})
        result = frame.select(rma(pl.col(COLUMN_X), 2).over(GROUP_KEY).alias("y"))["y"].to_list()
        assert_matches(result, [None, 1.5, 2.25, None, 15.0, 22.5])

    def test_window_one_is_float64_on_int_input(self) -> None:
        """
        Verifies that the ``window == 1`` identity short-circuit still yields ``Float64`` on an ``Int64`` input, so the
        output dtype is uniform with every ``window >= 2`` path.
        """
        frame = pl.DataFrame({COLUMN_X: pl.Series(COLUMN_X, [1, 2, 3], dtype=pl.Int64)})
        result = frame.select(rma(pl.col(COLUMN_X), 1).alias("y"))
        assert result.schema["y"] == pl.Float64


class TestRmaEdge:
    """
    Boundaries, warm-up, and null / NaN handling.
    """

    def test_window_below_one_raises(self) -> None:
        """
        Verifies that ``window < 1`` raises ``ValueError``.
        """
        with pytest.raises(ValueError, match="window must be >= 1"):
            rma(pl.col(COLUMN_X), 0)

    def test_single_row(self) -> None:
        """
        Verifies behavior on a one-element series.
        """
        assert_matches(apply_expr([42.0], rma(pl.col(COLUMN_X), 1)), [42.0])
        assert_matches(apply_expr([42.0], rma(pl.col(COLUMN_X), 3)), [None])

    def test_all_null(self) -> None:
        """
        Verifies that an all-null series yields an all-null output.
        """
        assert_matches(
            apply_expr([None, None, None, None, None], rma(pl.col(COLUMN_X), 3)), [None, None, None, None, None]
        )

    def test_null_bridged(self) -> None:
        """
        Verifies that an interior ``null`` yields ``null`` at that row while the recursion is bridged across the gap.
        """
        assert_matches(
            apply_expr([1.0, None, 3.0, 4.0], rma(pl.col(COLUMN_X), 2)),
            [None, None, 2.0, 3.0],
        )

    def test_nan_latches(self) -> None:
        """
        Verifies that a ``NaN`` latches into the recursion and poisons every subsequent value.
        """
        assert_matches(
            apply_expr([1.0, math.nan, 3.0, 4.0], rma(pl.col(COLUMN_X), 2)), [None, math.nan, math.nan, math.nan]
        )

    def test_warmup_null_count(self) -> None:
        """
        Verifies that the first ``window - 1`` rows are null and the first full window is defined.
        """
        result = apply_expr([1.0, 2.0, 3.0, 4.0, 5.0], rma(pl.col(COLUMN_X), 3))
        assert result[:2] == [None, None]
        assert result[2] is not None

    def test_window_exceeds_length(self) -> None:
        """
        Verifies that a window exceeding the series length yields an all-null output.
        """
        assert_matches(apply_expr([1.0, 2.0, 3.0], rma(pl.col(COLUMN_X), 5)), [None, None, None])

    def test_window_equals_length(self) -> None:
        """
        Verifies the single defined value when ``window`` equals the series length.
        """
        assert_matches(apply_expr([1.0, 2.0, 3.0], rma(pl.col(COLUMN_X), 3)), [None, None, 2.0])

    def test_window_one_is_identity(self) -> None:
        """
        Verifies that ``window == 1`` (alpha == 1) reproduces the input with no warm-up.
        """
        assert_matches(apply_expr([1.0, 2.0, 3.0], rma(pl.col(COLUMN_X), 1)), [1.0, 2.0, 3.0])

    def test_all_zero_series_is_zero(self) -> None:
        """
        Verifies the degenerate all-zero window: the recursion stays at zero. This is the case the subnormal-floor note
        pins here, kept out of the property fuzz by ``subnormal_safe_floats``.
        """
        assert_matches(apply_expr([0.0, 0.0, 0.0, 0.0], rma(pl.col(COLUMN_X), 3)), [None, None, 0.0, 0.0])

    def test_constant_series(self) -> None:
        """
        Verifies that a constant input yields that same constant at every defined (post-warm-up) row.
        """
        assert_matches(apply_expr([5.0, 5.0, 5.0, 5.0], rma(pl.col(COLUMN_X), 3)), [None, None, 5.0, 5.0])

    def test_interior_null_bridged(self) -> None:
        """
        Verifies the warm-up gate and gap-bridging for an interior ``null`` (``[2, 4, None, 8, 10, 12]``, window 3).
        """
        assert_matches(
            apply_expr([2.0, 4.0, None, 8.0, 10.0, 12.0], rma(pl.col(COLUMN_X), 3)),
            [None, None, None, 4.666666666666666, 6.444444444444444, 8.296296296296296],
        )

    def test_interior_null_after_seed_bridged(self) -> None:
        """
        Verifies the gap-aware renormalization on a ``null`` strictly AFTER the seed: the recursion must carry its
        state across the gap with the documented ``(1 - alpha) ** k`` decay, pinned deterministically against the
        closed-form reference (a pre-seed gap alone leaves this branch to the property tier's chance).
        """
        values = [2.0, 4.0, 6.0, None, 8.0, 10.0]
        assert_matches(apply_expr(values, rma(pl.col(COLUMN_X), 3)), rma_reference(values, 3))


class TestRmaCorrectness:
    """
    Against the naive reference oracle and frozen golden-master values.
    """

    def test_matches_reference(self) -> None:
        """
        Verifies agreement with the naive recursive reference across several windows.
        """
        values = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]
        for window in (1, 2, 3, 4, 5):
            assert_matches(apply_expr(values, rma(pl.col(COLUMN_X), window)), rma_reference(values, window))

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference: RMA(window=3) over [2, 4, 6, 8, 10] == [None, None, 4.0, 5.3333, 6.8889].
        """
        result = apply_expr([2.0, 4.0, 6.0, 8.0, 10.0], rma(pl.col(COLUMN_X), 3))
        expected = [None, None, 4.0, 5.333333333333333, 6.888888888888888]
        assert_matches(result, expected, rel_tol=RELATIVE_TOLERANCE_REFERENCE, abs_tol=ABSOLUTE_TOLERANCE_REFERENCE)


class TestRmaProperties:
    """
    Invariants that must hold for all inputs (property-based).
    """

    @given(case=_cases(subnormal_safe_floats(1e6)))
    def test_matches_reference_for_any_input(
        self,
        case: tuple[list[float], int],
    ) -> None:
        """
        Verifies that, for any series and window, the implementation matches the naive reference.
        """
        values, window = case
        assert_matches(
            apply_expr(values, rma(pl.col(COLUMN_X), window)),
            rma_reference(values, window),
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=input_scale(values) * EXACT_TOLERANCE_FACTOR,
        )

    @given(case=_cases(missing_data_floats(min_magnitude=SUBNORMAL_FLOOR)))
    def test_matches_reference_under_missing_data(
        self,
        case: tuple[list[float | None], int],
    ) -> None:
        """
        Verifies that, for inputs freely mixing null / NaN / finite, the implementation matches the naive reference.
        """
        values, window = case
        assert_matches(
            apply_expr(values, rma(pl.col(COLUMN_X), window)),
            rma_reference(values, window),
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=input_scale(values) * EXACT_TOLERANCE_FACTOR,
        )

    @given(
        case=_cases(subnormal_safe_floats(1e3)),
        exponent=st.sampled_from([-4, -3, -2, -1, 1, 2, 3, 4]),
    )
    def test_scale_homogeneity(
        self,
        case: tuple[list[float], int],
        exponent: int,
    ) -> None:
        """
        Verifies that ``rma`` is homogeneous of degree 1: scaling every input value by a constant ``k`` scales the
        output by the same ``k`` -- ``rma(k * x) == k * rma(x)``. ``k`` is a power of two, so the rescale is exact
        and adds no floating-point error.
        """
        k = 2.0**exponent
        values, window = case
        scaled_values = [value * k for value in values]
        result_base = apply_expr(values, rma(pl.col(COLUMN_X), window))
        result_scaled = apply_expr(scaled_values, rma(pl.col(COLUMN_X), window))
        assert_scale_homogeneous(result_scaled, result_base, k=k, degree=1)

    @given(
        case=_cases(st.floats(min_value=1e-3, max_value=1.0, allow_nan=False, allow_infinity=False)),
        scale=st.sampled_from([1e-6, 1e6, 1e9]),
    )
    def test_matches_reference_at_large_magnitude(
        self,
        case: tuple[list[float], int],
        scale: float,
    ) -> None:
        """
        Verifies that at extreme positive magnitudes the implementation stays finite where the reference is and agrees.
        """
        values, window = case
        scaled_values = [value * scale for value in values]
        assert_matches(
            apply_expr(scaled_values, rma(pl.col(COLUMN_X), window)),
            rma_reference(scaled_values, window),
            rel_tol=RELATIVE_TOLERANCE_SCALE,
            abs_tol=input_scale(scaled_values) * EXACT_TOLERANCE_FACTOR,
        )

    @given(case=_cases(subnormal_safe_floats(1e3)))
    def test_over_matches_per_group(
        self,
        case: tuple[list[float], int],
    ) -> None:
        """
        Verifies that ``.over`` computes each group exactly as the reference computes the group in isolation.
        """
        values, window = case
        group_labels = ["a"] * len(values) + ["b"] * len(values)
        frame = pl.DataFrame({GROUP_KEY: group_labels, COLUMN_X: values + values})
        result = frame.select(rma(pl.col(COLUMN_X), window).over(GROUP_KEY).alias("y"))["y"].to_list()
        expected_one = rma_reference(values, window)
        assert_matches(
            result,
            expected_one + expected_one,
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=input_scale(values) * EXACT_TOLERANCE_FACTOR,
        )
