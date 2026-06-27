"""
Tests for ``pomata.pnl.turnover`` — the traded fraction of capital between consecutive bars.

``turnover`` is single-input and a fixed one-bar-lag transform, so tests use the shared ``apply_expr`` helper to
materialize the factory over a one-column ``Float64`` frame; ``assert_matches`` and the naive ``turnover_reference``
oracle are shared across the suite. The pre-series weight is taken as flat (``0``), so there is NO warm-up null: the
first row is ``|weight_0|``. Turnover is degree-1 homogeneous, so it carries the scale-homogeneity and large-magnitude
tiers.

The ladder is the canonical one: contract (type / shape / lazy-eager / ``.over`` per-group independence), edge
(flat-start / single-row / null / NaN), correctness (vs the closed-form reference and a frozen golden master), and
properties (reference agreement incl. missing data, scale-homogeneity, large-magnitude). Categories are split into
classes; cross-cutting categories use markers (see ``tests/README.md``).
"""

import math

import polars as pl
from hypothesis import given
from hypothesis import strategies as st
from polars.testing import assert_frame_equal
from tests.pnl.oracles import turnover_reference
from tests.support import (
    ABSOLUTE_TOLERANCE_REFERENCE,
    ABSOLUTE_TOLERANCE_STREAMING,
    COLUMN_X,
    GROUP_KEY,
    RELATIVE_TOLERANCE_PROPERTY,
    RELATIVE_TOLERANCE_REFERENCE,
    RELATIVE_TOLERANCE_SCALE,
    apply_expr,
    assert_matches,
    assert_scale_homogeneous,
    finite_floats,
    missing_data_floats,
)

from pomata.pnl import turnover

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- derived, not chosen; the rationale is the shared method in CORRECTNESS.md, the numbers are this
# primitive's. turnover has a one-bar lag but the pre-series weight is flat (``0``), so there is NO warm-up: every row
# (including the first, ``|weight_0|``) is a defined output (W = 0, M = 0). It has no window parameter, so ``_cases``
# draws only the series. It is degree-1 homogeneous, so it keeps the scale-homogeneity and large-magnitude tiers.
# Repetitions N are the shared CI profile (tests/conftest.py); override per-test only if its parameter space is larger.
# ----------------------------------------------------------------------------------------------------------------------
SERIES_MAX = 50


@st.composite
def _cases[T](draw: st.DrawFn, weights: st.SearchStrategy[T], min_size: int = 1) -> list[T]:
    """
    A weight series sized from the facts above. turnover is windowless with a flat start, so a case is just the
    series; every row is a defined output.
    """
    return draw(st.lists(weights, min_size=min_size, max_size=SERIES_MAX))


class TestTurnoverContract:
    """
    Type, shape, and lazy/eager guarantees.
    """

    def test_returns_expr(self) -> None:
        """
        Verifies that the factory returns a ``pl.Expr`` without touching a frame.
        """
        assert isinstance(turnover(pl.col(COLUMN_X)), pl.Expr)

    def test_preserves_length_and_dtype(self) -> None:
        """
        Verifies that the output has one value per input row and is ``Float64``.
        """
        frame = pl.DataFrame({COLUMN_X: pl.Series(COLUMN_X, [0.5, 1.0, -0.5, -0.5, 0.0], dtype=pl.Float64)})
        result = frame.select(turnover(pl.col(COLUMN_X)).alias("y"))
        assert result.height == frame.height
        assert result.schema["y"] == pl.Float64

    def test_lazy_eager_parity(self) -> None:
        """
        Verifies that eager and lazy application produce identical materialized output.
        """
        frame = pl.DataFrame({COLUMN_X: pl.Series(COLUMN_X, [0.5, 1.0, -0.5, -0.5, 0.0], dtype=pl.Float64)})
        expr = turnover(pl.col(COLUMN_X)).alias("y")
        result_eager = frame.select(expr)
        result_lazy = frame.lazy().select(expr).collect()
        assert_frame_equal(result_eager, result_lazy)

    def test_over_partitions_independently(self) -> None:
        """
        Verifies that under ``.over`` the one-bar difference resets per group (each group gets its own flat start) and
        never reaches across group boundaries.
        """
        frame = pl.DataFrame({GROUP_KEY: ["a"] * 3 + ["b"] * 3, COLUMN_X: [0.5, 1.0, -0.5, 1.0, 1.0, 0.0]})
        expr = turnover(pl.col(COLUMN_X)).over(GROUP_KEY)
        grouped = frame.select(expr.alias("y"))["y"].to_list()
        group_a = apply_expr([0.5, 1.0, -0.5], turnover(pl.col(COLUMN_X)))
        group_b = apply_expr([1.0, 1.0, 0.0], turnover(pl.col(COLUMN_X)))
        assert_matches(grouped, group_a + group_b)


class TestTurnoverEdge:
    """
    Boundaries, the flat start, and null / NaN handling.
    """

    def test_flat_start_first_row(self) -> None:
        """
        Verifies the first row is ``|weight_0|`` (the entry trade from a flat start), not null.
        """
        assert_matches(apply_expr([0.5, 1.0, -0.5], turnover(pl.col(COLUMN_X))), [0.5, 0.5, 1.5])

    def test_single_row(self) -> None:
        """
        Verifies that a one-element series resolves to ``|weight_0|`` (the entry trade), not null.
        """
        assert_matches(apply_expr([0.7], turnover(pl.col(COLUMN_X))), [0.7])

    def test_empty(self) -> None:
        """
        Verifies that an empty series yields an empty result.
        """
        assert_matches(apply_expr([], turnover(pl.col(COLUMN_X))), [])

    def test_all_null(self) -> None:
        """
        Verifies that an all-null series stays null.
        """
        assert_matches(apply_expr([None, None, None], turnover(pl.col(COLUMN_X))), [None, None, None])

    def test_null_propagates(self) -> None:
        """
        Verifies that a null voids its own row and the next (the difference references the previous weight), matching
        the naive reference.
        """
        values = [0.5, None, 1.0, -0.5]
        assert_matches(apply_expr(values, turnover(pl.col(COLUMN_X))), turnover_reference(values))

    def test_nan_propagates(self) -> None:
        """
        Verifies that a NaN propagates to its own row and the next (matching the naive reference).
        """
        values = [0.5, math.nan, 1.0, -0.5]
        assert_matches(apply_expr(values, turnover(pl.col(COLUMN_X))), turnover_reference(values))

    def test_infinity_propagates(self) -> None:
        """
        Verifies IEEE infinity handling against the reference: a single ``inf`` yields ``|inf|`` and carries ``inf``
        into the next bar, while two consecutive equal-sign infinities make the second bar ``inf - inf = NaN``. The
        property tiers cannot reach this (their strategies set ``allow_infinity=False``), so it is pinned here.
        """
        values = [math.inf, math.inf, 1.0, -math.inf]
        assert_matches(apply_expr(values, turnover(pl.col(COLUMN_X))), turnover_reference(values))


class TestTurnoverCorrectness:
    """
    Against the naive reference oracle and frozen golden-master values.
    """

    def test_matches_reference(self) -> None:
        """
        Verifies agreement with the naive closed-form reference over a representative weight series.
        """
        values = [0.5, 1.0, -0.5, -0.5, 0.0, 1.5, -1.0, 0.25]
        assert_matches(
            apply_expr(values, turnover(pl.col(COLUMN_X))),
            turnover_reference(values),
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference over a five-bar weight series.
        """
        result = apply_expr([0.5, 1.0, -0.5, -0.5, 0.0], turnover(pl.col(COLUMN_X)).round(4))
        assert_matches(result, [0.5, 0.5, 1.5, 0.0, 0.5])


class TestTurnoverProperties:
    """
    Invariants that must hold for all inputs (property-based).
    """

    @given(case=_cases(finite_floats(), min_size=0))
    def test_matches_reference_for_any_input(
        self,
        case: list[float],
    ) -> None:
        """
        Verifies that, for any weight series, the implementation matches the naive reference.
        """
        values = case
        assert_matches(
            apply_expr(values, turnover(pl.col(COLUMN_X))),
            turnover_reference(values),
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    @given(case=_cases(missing_data_floats(), min_size=0))
    def test_matches_reference_under_missing_data(
        self,
        case: list[float | None],
    ) -> None:
        """
        Verifies that, for inputs freely mixing null / NaN / finite, the implementation matches the naive reference.
        """
        values = case
        assert_matches(
            apply_expr(values, turnover(pl.col(COLUMN_X))),
            turnover_reference(values),
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    @given(case=_cases(finite_floats()), exponent=st.sampled_from([-4, -3, -2, -1, 1, 2, 3, 4]))
    def test_scale_homogeneity(
        self,
        case: list[float],
        exponent: int,
    ) -> None:
        """
        Verifies degree-1 homogeneity: scaling the weight by a positive constant scales the turnover by the same
        constant. ``k`` is a power of two so the rescaling is lossless.
        """
        k = 2.0**exponent
        values = case
        result_base = apply_expr(values, turnover(pl.col(COLUMN_X)))
        result_scaled = apply_expr([value * k for value in values], turnover(pl.col(COLUMN_X)))
        assert_scale_homogeneous(result_scaled, result_base, k=k, degree=1)

    @given(case=_cases(finite_floats()), scale=st.sampled_from([1e-6, 1e6, 1e9]))
    def test_matches_reference_at_large_magnitude(
        self,
        case: list[float],
        scale: float,
    ) -> None:
        """
        Verifies that at extreme magnitudes the implementation stays finite where the reference is and agrees.
        """
        values = [value * scale for value in case]
        assert_matches(
            apply_expr(values, turnover(pl.col(COLUMN_X))),
            turnover_reference(values),
            rel_tol=RELATIVE_TOLERANCE_SCALE,
            abs_tol=ABSOLUTE_TOLERANCE_STREAMING,
        )
