"""
Tests for ``pomata.indicators.t3`` — Tillson's T3 Moving Average.

``t3`` is single-input and windowed, so tests use the shared ``apply_expr`` helper to materialize the factory over a
one-column ``Float64`` frame; ``assert_matches`` and the naive ``t3_reference`` oracle are shared across the suite. T3
is a six-EMA cascade (a Generalized DEMA applied three times), so it carries a ``6 * (window - 1)`` warm-up and the
recursive EMA state both propagates an interior ``null`` and latches a ``NaN`` for every subsequent value.

The ladder is the canonical one: contract (type / shape / lazy-eager / ``.over`` parity), edge (warm-up / window
boundaries / single-row / null / NaN), correctness (vs the closed-form reference and frozen golden masters across both
``adjust`` modes, plus the constant-series invariant), and properties (reference agreement for any input,
scale-homogeneity, and the exact leading-null run). Categories are split into classes; cross-cutting categories
elsewhere use markers (see ``tests/README.md``).
"""

import math

import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st
from polars.testing import assert_frame_equal
from tests.indicators.oracles import t3_reference
from tests.support import (
    ABSOLUTE_TOLERANCE_REFERENCE,
    COLUMN_X,
    EXACT_TOLERANCE_FACTOR,
    GROUP_KEY,
    RELATIVE_TOLERANCE_PROPERTY,
    RELATIVE_TOLERANCE_REFERENCE,
    RELATIVE_TOLERANCE_SCALE,
    SUBNORMAL_FLOOR,
    apply_expr,
    assert_matches,
    assert_scale_homogeneous,
    count_leading_nulls,
    input_scale,
    missing_data_floats,
    subnormal_safe_floats,
)

from pomata.indicators import t3

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- derived, not chosen; the rationale is the shared method in CORRECTNESS.md, the numbers are this
# indicator's. To add an indicator, set its three facts here; the property tier below is then the same shape as every
# other indicator's.
#   1. warmup  W(window) = 6 * (window - 1)   (T3 is a six-EMA cascade of the same ``window``, each pass carrying a
#              ``window - 1`` warm-up, so the lead-in is six times a plain EMA's)
#   2. memory  the oracle shares pomata's recursive EMA seeding, so the property holds from the
#              first defined row (M = 0); each example carries D in [window, 2 * window] defined values past the
#              warm-up -- always output to check, never an all-warm-up series
#   3. domain  subnormal_safe_floats(bound): finite values floored away from the subnormal range where a recursive
#              mean's ``input_scale * EXACT_TOLERANCE_FACTOR`` abs-tol collapses to 0.0 and it rounds one ULP from the
#              oracle (see the helper); the bound is widened per test below
# Windows span ``window_min`` .. WINDOW_MAX; WINDOW_MAX is held at 8 so the six-fold warm-up keeps the series lengths
# bounded. Repetitions N are the shared CI profile (tests/conftest.py); override per-test only if its parameter space
# is larger.
# ----------------------------------------------------------------------------------------------------------------------
WINDOW_MAX = 8


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
    length = 6 * (window - 1) + defined
    return draw(st.lists(values, min_size=length, max_size=length)), window


class TestT3Contract:
    """
    Type, shape, and lazy/eager guarantees.
    """

    def test_returns_expr(self) -> None:
        """
        Verifies that the factory returns a ``pl.Expr`` without touching a frame.
        """
        assert isinstance(t3(pl.col(COLUMN_X), 3), pl.Expr)

    def test_preserves_length_and_dtype(self) -> None:
        """
        Verifies that the output has one value per input row and is ``Float64``.
        """
        frame = pl.DataFrame({COLUMN_X: pl.Series(COLUMN_X, [float(i) for i in range(12)])})
        result = frame.select(t3(pl.col(COLUMN_X), 2).alias("y"))
        assert result.height == frame.height
        assert result.schema["y"] == pl.Float64

    def test_lazy_eager_parity(self) -> None:
        """
        Verifies that eager and lazy application produce identical materialized output.
        """
        frame = pl.DataFrame({COLUMN_X: pl.Series(COLUMN_X, [float(i) for i in range(12)])})
        result_eager = frame.select(t3(pl.col(COLUMN_X), 2).alias("y"))
        result_lazy = frame.lazy().select(t3(pl.col(COLUMN_X), 2).alias("y")).collect()
        assert_frame_equal(result_eager, result_lazy)

    def test_over_partitions_independently(self) -> None:
        """
        Verifies that under ``.over`` each EMA pass resets per group and never spans group boundaries.
        """
        values_a = [float(i) for i in range(10)]
        values_b = [10.0 * i for i in range(10)]
        frame = pl.DataFrame({GROUP_KEY: ["a"] * 10 + ["b"] * 10, COLUMN_X: values_a + values_b})
        result_over = frame.select(t3(pl.col(COLUMN_X), 2).over(GROUP_KEY).alias("y"))["y"].to_list()
        result_a = apply_expr(values_a, t3(pl.col(COLUMN_X), 2))
        result_b = apply_expr(values_b, t3(pl.col(COLUMN_X), 2))
        assert_matches(result_over, result_a + result_b)


class TestT3Edge:
    """
    Boundaries, warm-up, and null / NaN handling.
    """

    def test_window_below_one_raises(self) -> None:
        """
        Verifies that ``window < 1`` raises ``ValueError``.
        """
        with pytest.raises(ValueError, match="window must be >= 1"):
            t3(pl.col(COLUMN_X), 0)

    def test_non_finite_volume_factor_raises(self) -> None:
        """
        Verifies that a non-finite ``volume_factor`` (``NaN`` or ``±inf``) raises ``ValueError``.
        """
        for invalid in (math.nan, math.inf, -math.inf):
            with pytest.raises(ValueError, match="volume_factor must be a finite number"):
                t3(pl.col(COLUMN_X), 3, volume_factor=invalid)

    def test_warmup_null_count(self) -> None:
        """
        Verifies that the first ``6 * (window - 1)`` rows are null and the next is defined.
        """
        result = apply_expr([float(i) for i in range(15)], t3(pl.col(COLUMN_X), 3))
        assert result[:12] == [None] * 12
        assert result[12] is not None

    def test_empty(self) -> None:
        """
        Verifies that an empty input yields an empty output.
        """
        assert_matches(apply_expr([], t3(pl.col(COLUMN_X), 3)), [])

    def test_all_null(self) -> None:
        """
        Verifies that an all-null series yields an all-null output.
        """
        assert_matches(
            apply_expr([None, None, None, None, None], t3(pl.col(COLUMN_X), 3)), [None, None, None, None, None]
        )

    def test_all_zero_series_is_zero(self) -> None:
        """
        Verifies the degenerate all-zero window: the recursion stays at zero. This is the case the subnormal-floor note
        pins here, kept out of the property fuzz by ``subnormal_safe_floats``.
        """
        assert_matches(apply_expr([0.0] * 8, t3(pl.col(COLUMN_X), 2)), [None, None, None, None, None, None, 0.0, 0.0])

    def test_window_one_is_identity(self) -> None:
        """
        Verifies that ``window == 1`` reproduces the input (every EMA is the identity and the coefficients sum to 1).
        """
        assert_matches(apply_expr([1.0, 2.0, 3.0], t3(pl.col(COLUMN_X), 1)), [1.0, 2.0, 3.0])

    def test_window_fills_entire_series_with_warmup(self) -> None:
        """
        Verifies that when the ``6 * (window - 1)`` warm-up meets or exceeds the series length the whole output is null.
        """
        assert_matches(apply_expr([1.0, 2.0], t3(pl.col(COLUMN_X), 2)), [None, None])

    def test_single_row(self) -> None:
        """
        Verifies behavior on a one-element series.
        """
        assert_matches(apply_expr([42.0], t3(pl.col(COLUMN_X), 1)), [42.0])
        assert_matches(apply_expr([42.0], t3(pl.col(COLUMN_X), 2)), [None])

    def test_interior_null(self) -> None:
        """
        Verifies that an interior ``null`` yields ``null`` at its position while the recursion bridges the gap.
        """
        assert_matches(
            apply_expr([2.0, 4.0, None, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0], t3(pl.col(COLUMN_X), 2)),
            [None, None, None, None, None, None, None, 15.14916985651142, 17.11616757027457, 19.104042047403723],
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    def test_nan_propagates(self) -> None:
        """
        Verifies that a ``NaN`` poisons the recursion and latches for every subsequent value.
        """
        assert_matches(
            apply_expr([1.0, math.nan, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], t3(pl.col(COLUMN_X), 2)),
            [None, None, None, None, None, None, math.nan, math.nan, math.nan, math.nan],
        )


class TestT3Correctness:
    """
    Against the naive reference oracle and frozen golden-master values.
    """

    def test_matches_reference(self) -> None:
        """
        Verifies agreement with the naive reference across several windows, volume factors, and both ``adjust`` modes.
        """
        values = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0, 3.0, 5.0, 8.0, 9.0, 7.0, 9.0, 3.0]
        for window in (1, 2, 3, 4, 5):
            for volume_factor in (0.7, 0.0, 0.5):
                for adjust in (False, True):
                    assert_matches(
                        apply_expr(values, t3(pl.col(COLUMN_X), window, volume_factor=volume_factor, adjust=adjust)),
                        t3_reference(values, window, volume_factor=volume_factor, adjust=adjust),
                        rel_tol=RELATIVE_TOLERANCE_REFERENCE,
                        abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
                    )

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference: T3(window=2) over [2, 4, ..., 20].
        """
        assert_matches(
            apply_expr([2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0], t3(pl.col(COLUMN_X), 2)),
            [
                None,
                None,
                None,
                None,
                None,
                None,
                13.099999999999994,
                15.100000000000001,
                17.10000000000001,
                19.099999999999994,
            ],
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    def test_golden_master_adjust(self) -> None:
        """
        Verifies the frozen adjusted reference: T3(window=2, adjust=True) over [2, 4, ..., 20].
        """
        assert_matches(
            apply_expr([2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0], t3(pl.col(COLUMN_X), 2, adjust=True)),
            [None, None, None, None, None, None, 13.065715164239, 15.090278304463, 17.098149765047, 19.10020550466],
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    def test_constant_series_is_constant(self) -> None:
        """
        Verifies that T3 of a constant series equals that constant once warmed up.
        """
        assert_matches(
            apply_expr([5.0] * 10, t3(pl.col(COLUMN_X), 2)),
            [None, None, None, None, None, None, 5.0, 5.0, 5.0, 5.0],
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )


class TestT3Properties:
    """
    Invariants that must hold for all inputs (property-based).
    """

    @given(
        case=_cases(subnormal_safe_floats(1e6)),
        volume_factor=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        adjust=st.booleans(),
    )
    def test_matches_reference_for_any_input(
        self,
        case: tuple[list[float], int],
        volume_factor: float,
        adjust: bool,
    ) -> None:
        """
        Verifies the implementation matches the reference for any series, window, volume factor, and ``adjust`` mode.
        """
        values, window = case
        assert_matches(
            apply_expr(values, t3(pl.col(COLUMN_X), window, volume_factor=volume_factor, adjust=adjust)),
            t3_reference(values, window, volume_factor=volume_factor, adjust=adjust),
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
        Verifies that T3 is homogeneous of degree 1: ``t3(k * x) == k * t3(x)``. ``k`` is a power of two so the
        rescaling is lossless.
        """
        k = 2.0**exponent
        values, window = case
        scaled_values = [value * k for value in values]
        result_base = apply_expr(values, t3(pl.col(COLUMN_X), window))
        result_scaled = apply_expr(scaled_values, t3(pl.col(COLUMN_X), window))
        assert_scale_homogeneous(result_scaled, result_base, k=k, degree=1)

    @given(case=_cases(subnormal_safe_floats(1e6)))
    def test_warmup_null_count_property(
        self,
        case: tuple[list[float], int],
    ) -> None:
        """
        Verifies that the leading-null run is exactly ``min(6 * (window - 1), len(values))``.
        """
        values, window = case
        result = apply_expr(values, t3(pl.col(COLUMN_X), window))
        leading_nulls = count_leading_nulls(result)
        assert leading_nulls == min(6 * (window - 1), len(values))

    # The missing-data tier floors out subnormal-magnitude draws via ``min_magnitude``, exactly as the finite tiers do
    # via ``subnormal_safe_floats``: at a subnormal-magnitude window a recursive ewm_mean and the two-pass oracle round
    # one ULP apart while ``input_scale(values) * EXACT_TOLERANCE_FACTOR`` collapses the abs_tol to 0.0 -- a benign
    # last-bit artifact, not a bug. The excised range is numerically equivalent (the recurrence is scale-invariant),
    # so the floor drops the artifact, not real coverage; the degenerate all-zero window is pinned in the Edge tier.
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
            apply_expr(values, t3(pl.col(COLUMN_X), window)),
            t3_reference(values, window),
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=input_scale(values) * EXACT_TOLERANCE_FACTOR,
        )

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
            apply_expr(scaled_values, t3(pl.col(COLUMN_X), window)),
            t3_reference(scaled_values, window),
            rel_tol=RELATIVE_TOLERANCE_SCALE,
            abs_tol=input_scale(scaled_values) * EXACT_TOLERANCE_FACTOR,
        )
