"""
Tests for ``pomata.indicators.dema`` — Mulloy's Double Exponential Moving Average.

``dema`` is single-input and windowed, so tests use the shared ``apply_expr`` helper to materialize the factory over a
one-column ``Float64`` frame; ``assert_matches`` and the naive ``dema_reference`` oracle are shared across the suite.
DEMA is a two-EMA combination (``2 * EMA(x) - EMA(EMA(x))``, not an EMA applied twice), so it carries a
``2 * (window - 1)`` warm-up and the recursive EMA state both propagates an interior ``null`` and latches a ``NaN`` for
every subsequent value.

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
from tests.indicators.oracles import dema_reference
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
    count_leading_nulls,
    input_scale,
    missing_data_floats,
    subnormal_safe_floats,
)

from pomata.indicators import dema

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- derived, not chosen; the rationale is the shared method in CORRECTNESS.md, the numbers are this
# indicator's. To add an indicator, set its three facts here; the property tier below is then the same shape as every
# other indicator's.
#   1. warmup  W(window) = 2 * (window - 1)   (DEMA chains two EMA passes of the same ``window``, each carrying a
#              ``window - 1`` warm-up, so the lead-in is twice a plain EMA's)
#   2. memory  the oracle shares pomata's recursive EMA seeding, so the property holds from the
#              first defined row (M = 0); each example carries D in [window, 2 * window] defined values past the
#              warm-up -- always output to check, never an all-warm-up series
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
    length = 2 * (window - 1) + defined
    return draw(st.lists(values, min_size=length, max_size=length)), window


class TestDemaContract:
    """
    Type, shape, and lazy/eager guarantees.
    """

    def test_over_partitions_independently(self) -> None:
        """
        Verifies that under ``.over`` each EMA pass resets per group and never spans group boundaries.
        """
        frame = pl.DataFrame(
            {
                GROUP_KEY: ["a", "a", "a", "a", "b", "b", "b", "b"],
                COLUMN_X: [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
            }
        )
        result = frame.select(dema(pl.col(COLUMN_X), 2).over(GROUP_KEY).alias("y"))["y"].to_list()
        assert_matches(
            result,
            [None, None, 3.0, 4.0, None, None, 30.0, 40.0],
        )


class TestDemaEdge:
    """
    Boundaries, warm-up, and null / NaN handling.
    """

    def test_window_below_one_raises(self) -> None:
        """
        Verifies that ``window < 1`` raises ``ValueError``.
        """
        with pytest.raises(ValueError, match="window must be >= 1"):
            dema(pl.col(COLUMN_X), 0)

    def test_warmup_null_count(self) -> None:
        """
        Verifies that the first ``2 * (window - 1)`` rows are null and the next is defined.
        """
        result = apply_expr([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dema(pl.col(COLUMN_X), 3))
        assert result[:4] == [None, None, None, None]
        assert result[4] is not None

    def test_all_null(self) -> None:
        """
        Verifies that an all-null series yields an all-null output.
        """
        assert_matches(
            apply_expr([None, None, None, None, None], dema(pl.col(COLUMN_X), 3)), [None, None, None, None, None]
        )

    def test_all_zero_series_is_zero(self) -> None:
        """
        Verifies the degenerate all-zero window: the recursion stays at zero. This is the case the subnormal-floor note
        pins here, kept out of the property fuzz by ``subnormal_safe_floats``.
        """
        assert_matches(apply_expr([0.0] * 6, dema(pl.col(COLUMN_X), 3)), [None, None, None, None, 0.0, 0.0])

    def test_window_one_is_identity(self) -> None:
        """
        Verifies that ``window == 1`` reproduces the input (each EMA is the identity).
        """
        assert_matches(apply_expr([1.0, 2.0, 3.0], dema(pl.col(COLUMN_X), 1)), [1.0, 2.0, 3.0])

    def test_window_fills_entire_series_with_warmup(self) -> None:
        """
        Verifies that when the ``2 * (window - 1)`` warm-up meets or exceeds the series length the whole output is null.
        """
        assert_matches(apply_expr([1.0, 2.0], dema(pl.col(COLUMN_X), 2)), [None, None])

    def test_single_row(self) -> None:
        """
        Verifies behavior on a one-element series.
        """
        assert_matches(apply_expr([42.0], dema(pl.col(COLUMN_X), 1)), [42.0])
        assert_matches(apply_expr([42.0], dema(pl.col(COLUMN_X), 2)), [None])

    def test_null_bridged(self) -> None:
        """
        Verifies that an early ``null`` extends the warm-up and yields ``null`` at that position, the value resuming
        once enough non-null observations have seeded both EMA passes.
        """
        assert_matches(
            apply_expr([1.0, None, 3.0, 4.0], dema(pl.col(COLUMN_X), 2)), [None, None, None, 3.9999999999999996]
        )

    def test_interior_null_bridged(self) -> None:
        """
        Verifies that an interior ``null`` yields ``null`` at its position while the recursion bridges the gap.
        """
        assert_matches(
            apply_expr([2.0, 4.0, None, 8.0, 10.0, 12.0], dema(pl.col(COLUMN_X), 2)),
            [None, None, None, 9.428571428571429, 10.412698412698413, 12.116402116402117],
        )

    def test_nan_latches(self) -> None:
        """
        Verifies that a ``NaN`` poisons the recursion and latches for every subsequent value.
        """
        assert_matches(
            apply_expr([1.0, math.nan, 3.0, 4.0, 5.0], dema(pl.col(COLUMN_X), 2)),
            [None, None, math.nan, math.nan, math.nan],
        )


class TestDemaCorrectness:
    """
    Against the naive reference oracle and frozen golden-master values.
    """

    def test_matches_reference(self) -> None:
        """
        Verifies agreement with the naive reference across several windows and both ``adjust`` modes.
        """
        values = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]
        for window in (1, 2, 3, 4, 5):
            for adjust in (False, True):
                assert_matches(
                    apply_expr(values, dema(pl.col(COLUMN_X), window, adjust=adjust)),
                    dema_reference(values, window, adjust=adjust),
                    rel_tol=RELATIVE_TOLERANCE_REFERENCE,
                    abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
                )

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference: DEMA(window=2) over [2, 4, 6, 8, 10, 12].
        """
        assert_matches(
            apply_expr([2.0, 4.0, 6.0, 8.0, 10.0, 12.0], dema(pl.col(COLUMN_X), 2)),
            [None, None, 6.0, 8.0, 10.0, 12.0],
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    def test_golden_master_adjust(self) -> None:
        """
        Verifies the frozen adjusted reference: DEMA(window=3, adjust=True) over [3, 1, 4, 1, 5, 9, 2, 6].
        """
        assert_matches(
            apply_expr([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0], dema(pl.col(COLUMN_X), 3, adjust=True)),
            [None, None, None, None, 4.042089093701996, 7.846915855948113, 3.832696745373009, 5.383328263901351],
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    def test_constant_series_is_constant(self) -> None:
        """
        Verifies that DEMA of a constant series equals that constant once warmed up.
        """
        assert_matches(
            apply_expr([5.0, 5.0, 5.0, 5.0, 5.0, 5.0], dema(pl.col(COLUMN_X), 3)),
            [None, None, None, None, 5.0, 5.0],
        )


class TestDemaProperties:
    """
    Invariants that must hold for all inputs (property-based).
    """

    @given(
        case=_cases(subnormal_safe_floats(1e6)),
        adjust=st.booleans(),
    )
    def test_matches_reference_for_any_input(
        self,
        case: tuple[list[float], int],
        adjust: bool,
    ) -> None:
        """
        Verifies the implementation matches the reference for any series, window, and ``adjust`` mode.
        """
        values, window = case
        assert_matches(
            apply_expr(values, dema(pl.col(COLUMN_X), window, adjust=adjust)),
            dema_reference(values, window, adjust=adjust),
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
            apply_expr(values, dema(pl.col(COLUMN_X), window)),
            dema_reference(values, window),
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
        Verifies that ``dema`` is homogeneous of degree 1: scaling every input value by a constant ``k`` scales the
        output by the same ``k`` -- ``dema(k * x) == k * dema(x)``. ``k`` is a power of two, so the rescale is exact
        and adds no floating-point error.
        """
        k = 2.0**exponent
        values, window = case
        scaled_values = [value * k for value in values]
        result_base = apply_expr(values, dema(pl.col(COLUMN_X), window))
        result_scaled = apply_expr(scaled_values, dema(pl.col(COLUMN_X), window))
        assert_scale_homogeneous(result_scaled, result_base, k=k, degree=1)

    @given(case=_cases(subnormal_safe_floats(1e6)))
    def test_warmup_null_count_property(
        self,
        case: tuple[list[float], int],
    ) -> None:
        """
        Verifies that the leading-null run is exactly ``min(2 * (window - 1), len(values))``.
        """
        values, window = case
        result = apply_expr(values, dema(pl.col(COLUMN_X), window))
        leading_nulls = count_leading_nulls(result)
        assert leading_nulls == min(2 * (window - 1), len(values))

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
            apply_expr(scaled_values, dema(pl.col(COLUMN_X), window)),
            dema_reference(scaled_values, window),
            rel_tol=RELATIVE_TOLERANCE_SCALE,
            abs_tol=input_scale(scaled_values) * EXACT_TOLERANCE_FACTOR,
        )
