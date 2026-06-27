"""
Tests for ``pomata.indicators.tema`` — Triple Exponential Moving Average.

``tema`` is single-input and windowed, so tests use the shared ``apply_expr`` helper to materialize the factory over a
one-column ``Float64`` frame; ``assert_matches`` and the naive ``tema_reference`` oracle are shared across the suite.
The reference chains three exponential passes exactly as the implementation does, so agreement is by construction.

The ladder is the canonical one: contract (type / shape / lazy-eager / ``.over`` parity), edge (warm-up over the
compounded ``3 * (window - 1)`` lead-in / window boundaries / single-row / null / NaN), correctness (vs the recursive
reference, in both EMA modes, plus frozen golden masters), and properties (reference agreement for any input and
scale-homogeneity). Categories are split into classes; cross-cutting categories elsewhere use markers (see
``tests/README.md``).
"""

import math

import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st
from polars.testing import assert_frame_equal
from tests.indicators.oracles import tema_reference
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
    input_scale,
    missing_data_floats,
    subnormal_safe_floats,
)

from pomata.indicators import tema

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- derived, not chosen; the rationale is the shared method in CORRECTNESS.md, the numbers are this
# indicator's. To add an indicator, set its three facts here; the property tier below is then the same shape as every
# other indicator's.
#   1. warmup  W(window) = 3 * (window - 1)   (TEMA chains three EMA passes of the same ``window``, each carrying a
#              ``window - 1`` warm-up, so the lead-in is three times a plain EMA's)
#   2. memory  the oracle shares pomata's recursive EMA seeding, so the property holds from the
#              first defined row (M = 0); each example carries D in [window, 2 * window] defined values past the
#              warm-up -- always output to check, never an all-warm-up series
#   3. domain  subnormal_safe_floats(bound): finite values floored away from the subnormal range where a recursive
#              mean's ``input_scale * EXACT_TOLERANCE_FACTOR`` abs-tol collapses to 0.0 and it rounds one ULP from the
#              oracle (see the helper); the bound is widened per test below
# Windows span ``window_min`` .. WINDOW_MAX. Repetitions N are the shared CI profile (tests/conftest.py); override
# per-test only if its parameter space is larger.
# ----------------------------------------------------------------------------------------------------------------------
WINDOW_MAX = 12


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
    length = 3 * (window - 1) + defined
    return draw(st.lists(values, min_size=length, max_size=length)), window


class TestTemaContract:
    """
    Type, shape, and lazy/eager guarantees.
    """

    def test_returns_expr(self) -> None:
        """
        Verifies that the factory returns a ``pl.Expr`` without touching a frame.
        """
        assert isinstance(tema(pl.col(COLUMN_X), 3), pl.Expr)

    def test_preserves_length_and_dtype(self) -> None:
        """
        Verifies that the output has one value per input row and is ``Float64``.
        """
        frame = pl.DataFrame({COLUMN_X: pl.Series(COLUMN_X, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])})
        result = frame.select(tema(pl.col(COLUMN_X), 2).alias("y"))
        assert result.height == frame.height
        assert result.schema["y"] == pl.Float64

    def test_lazy_eager_parity(self) -> None:
        """
        Verifies that eager and lazy application produce identical materialized output.
        """
        frame = pl.DataFrame({COLUMN_X: pl.Series(COLUMN_X, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])})
        result_eager = frame.select(tema(pl.col(COLUMN_X), 2).alias("y"))
        result_lazy = frame.lazy().select(tema(pl.col(COLUMN_X), 2).alias("y")).collect()
        assert_frame_equal(result_eager, result_lazy)

    def test_over_partitions_independently(self) -> None:
        """
        Verifies that under ``.over`` the recursion resets per group and never spans group boundaries.
        """
        frame = pl.DataFrame(
            {
                GROUP_KEY: ["a", "a", "a", "a", "a", "a", "b", "b", "b", "b", "b", "b"],
                COLUMN_X: [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 20.0, 40.0, 60.0, 80.0, 100.0, 120.0],
            }
        )
        result = frame.select(tema(pl.col(COLUMN_X), 2).over(GROUP_KEY).alias("y"))["y"].to_list()
        expected_group_a = tema_reference([2.0, 4.0, 6.0, 8.0, 10.0, 12.0], 2)
        expected_group_b = tema_reference([20.0, 40.0, 60.0, 80.0, 100.0, 120.0], 2)
        assert_matches(
            result,
            expected_group_a + expected_group_b,
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )


class TestTemaEdge:
    """
    Boundaries, warm-up, and null / NaN handling.
    """

    def test_window_below_one_raises(self) -> None:
        """
        Verifies that ``window < 1`` raises ``ValueError``.
        """
        with pytest.raises(ValueError, match="window must be >= 1"):
            tema(pl.col(COLUMN_X), 0)

    def test_warmup_null_count(self) -> None:
        """
        Verifies that the cascade's first ``3 * (window - 1)`` rows are null (warm-up) and the next is defined.
        """
        values = [float(index + 1) for index in range(10)]
        result = apply_expr(values, tema(pl.col(COLUMN_X), 3))
        assert result[:6] == [None, None, None, None, None, None]
        assert result[6] is not None

    def test_empty(self) -> None:
        """
        Verifies that an empty input yields an empty output.
        """
        assert_matches(apply_expr([], tema(pl.col(COLUMN_X), 3)), [])

    def test_all_null(self) -> None:
        """
        Verifies that an all-null series yields an all-null output.
        """
        assert_matches(
            apply_expr([None, None, None, None, None], tema(pl.col(COLUMN_X), 3)), [None, None, None, None, None]
        )

    def test_all_zero_series_is_zero(self) -> None:
        """
        Verifies the degenerate all-zero window: the recursion stays at zero. This is the case the subnormal-floor note
        pins here, kept out of the property fuzz by ``subnormal_safe_floats``.
        """
        assert_matches(apply_expr([0.0] * 8, tema(pl.col(COLUMN_X), 3)), [None, None, None, None, None, None, 0.0, 0.0])

    def test_window_one_is_identity(self) -> None:
        """
        Verifies that ``window == 1`` reproduces the input, since each of the three nested EMAs collapses to the
        identity.
        """
        assert_matches(apply_expr([1.0, 2.0, 3.0, 4.0], tema(pl.col(COLUMN_X), 1)), [1.0, 2.0, 3.0, 4.0])

    def test_window_exceeds_length(self) -> None:
        """
        Verifies that when the ``3 * (window - 1)`` cascade warm-up exceeds the series length the whole output is null.
        """
        assert_matches(apply_expr([1.0, 2.0, 3.0], tema(pl.col(COLUMN_X), 5)), [None, None, None])

    def test_single_row(self) -> None:
        """
        Verifies behavior on a one-element series: ``window == 1`` returns the value, a larger window is all warm-up.
        """
        assert_matches(apply_expr([42.0], tema(pl.col(COLUMN_X), 1)), [42.0])
        assert_matches(apply_expr([42.0], tema(pl.col(COLUMN_X), 3)), [None])

    def test_constant_series(self) -> None:
        """
        Verifies that, after warm-up, the TEMA of a constant series is that constant (the lag correction is exact).
        """
        assert_matches(
            apply_expr([7.0, 7.0, 7.0, 7.0, 7.0, 7.0], tema(pl.col(COLUMN_X), 2)), [None, None, None, 7.0, 7.0, 7.0]
        )

    def test_interior_null(self) -> None:
        """
        Verifies that an interior ``null`` yields ``null`` at that position and agrees with the reference.
        """
        values: list[float | None] = [2.0, 4.0, None, 8.0, 10.0, 12.0]
        result = apply_expr(values, tema(pl.col(COLUMN_X), 2))
        assert result[2] is None
        assert_matches(
            result,
            tema_reference(values, 2),
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    def test_nan_propagates(self) -> None:
        """
        Verifies that a ``NaN`` poisons the recursion (latching to ``NaN`` thereafter) and agrees with the reference.
        """
        values = [2.0, 4.0, math.nan, 8.0, 10.0, 12.0, 14.0]
        result = apply_expr(values, tema(pl.col(COLUMN_X), 2))
        assert_matches(
            result,
            tema_reference(values, 2),
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )


class TestTemaCorrectness:
    """
    Against the naive reference oracle and frozen golden-master values.
    """

    def test_matches_reference(self) -> None:
        """
        Verifies agreement with the naive recursive-EMA reference across several windows.
        """
        values = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0, 3.0, 5.0, 8.0]
        for window in (1, 2, 3, 4, 5):
            assert_matches(
                apply_expr(values, tema(pl.col(COLUMN_X), window)),
                tema_reference(values, window),
                rel_tol=RELATIVE_TOLERANCE_REFERENCE,
                abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
            )

    def test_matches_reference_adjusted(self) -> None:
        """
        Verifies agreement with the naive reference in the adjusted (``adjust=True``) expanding-weights EMA mode.
        """
        values = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0, 3.0, 5.0, 8.0]
        for window in (1, 2, 3, 4):
            assert_matches(
                apply_expr(values, tema(pl.col(COLUMN_X), window, adjust=True)),
                tema_reference(values, window, adjust=True),
                rel_tol=RELATIVE_TOLERANCE_REFERENCE,
                abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
            )

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference: TEMA(window=2) over [2, 4, 6, 8, 10, 12].
        """
        assert_matches(
            apply_expr([2.0, 4.0, 6.0, 8.0, 10.0, 12.0], tema(pl.col(COLUMN_X), 2)),
            [None, None, None, 8.0, 10.0, 12.0],
        )

    def test_golden_master_window_three(self) -> None:
        """
        Verifies the frozen reference: TEMA(window=3) over [3, 1, 4, 1, 5, 9, 2, 6, 5, 3].
        """
        assert_matches(
            apply_expr([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0, 3.0], tema(pl.col(COLUMN_X), 3)),
            [
                None,
                None,
                None,
                None,
                None,
                None,
                3.296296296296295,
                5.399016203703703,
                5.081452546296296,
                3.234953703703704,
            ],
        )


class TestTemaProperties:
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
        Verifies that, for any series, window, and ``adjust`` mode, the implementation matches the naive reference.
        """
        values, window = case
        assert_matches(
            apply_expr(values, tema(pl.col(COLUMN_X), window, adjust=adjust)),
            tema_reference(values, window, adjust=adjust),
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
        Verifies that TEMA is homogeneous of degree 1: ``tema(k * x) == k * tema(x)``. ``k`` is a power of two so the
        rescaling is lossless and cannot introduce a sub-ULP drift into the comparison.
        """
        k = 2.0**exponent
        values, window = case
        scaled_values = [value * k for value in values]
        result_base = apply_expr(values, tema(pl.col(COLUMN_X), window))
        result_scaled = apply_expr(scaled_values, tema(pl.col(COLUMN_X), window))
        assert_scale_homogeneous(result_scaled, result_base, k=k, degree=1)

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
            apply_expr(values, tema(pl.col(COLUMN_X), window)),
            tema_reference(values, window),
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
            apply_expr(scaled_values, tema(pl.col(COLUMN_X), window)),
            tema_reference(scaled_values, window),
            rel_tol=RELATIVE_TOLERANCE_SCALE,
            abs_tol=input_scale(scaled_values) * EXACT_TOLERANCE_FACTOR,
        )
