"""
Tests for ``pomata.metrics.win_rate`` — the fraction of decisive returns that are positive.

``win_rate`` is single-input and REDUCING (a return series → one scalar), so tests read the single output row of
``apply_expr``; ``assert_matches`` and the naive ``win_rate_reference`` oracle (positive count over non-zero count) are
shared across the suite. It is scale-invariant (signs do not change under a positive rescale), so it carries a
scale-invariance tier.

The ladder is the canonical one: contract (type / reduces-to-scalar / lazy-eager / ``.over`` per-group independence),
edge (empty / single-row / constant / zero / null / NaN), correctness (vs the closed-form reference and a frozen golden
master), and properties (reference agreement incl. missing data, scale invariance). Categories are split into classes;
cross-cutting categories use markers.
"""

import math

import polars as pl
from hypothesis import given
from hypothesis import strategies as st
from polars.testing import assert_frame_equal
from tests.metrics.oracles import win_rate_reference
from tests.support import (
    ABSOLUTE_TOLERANCE_REFERENCE,
    COLUMN_X,
    GROUP_KEY,
    RELATIVE_TOLERANCE_PROPERTY,
    RELATIVE_TOLERANCE_REFERENCE,
    RELATIVE_TOLERANCE_SCALE,
    apply_expr,
    assert_matches,
    missing_data_floats,
    subnormal_safe_floats,
)

from pomata.metrics import win_rate

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- win_rate is windowless and REDUCING (M = 0); a case is just a return series. Facts:
#   1. shape   reducing: the output is one scalar (one row in ``select``; one per group under ``.over``)
#   2. domain  subnormal_safe_floats(bound): finite returns floored off subnormal; the missing variant mixes null / NaN
#   3. scale   invariant (signs unchanged by a positive rescale) -> scale-invariance tier
# Repetitions N are the shared CI profile (tests/conftest.py).
# ----------------------------------------------------------------------------------------------------------------------
SERIES_MAX = 50


@st.composite
def _cases[T](draw: st.DrawFn, returns: st.SearchStrategy[T], min_size: int = 1) -> list[T]:
    """A return series sized from the facts above."""
    return draw(st.lists(returns, min_size=min_size, max_size=SERIES_MAX))


class TestWinRateContract:
    """
    Type, shape, and lazy/eager guarantees.
    """

    def test_returns_expr(self) -> None:
        """
        Verifies that the factory returns a ``pl.Expr`` without touching a frame.
        """
        assert isinstance(win_rate(pl.col(COLUMN_X)), pl.Expr)

    def test_reduces_to_scalar(self) -> None:
        """
        Verifies that the metric reduces a series to one ``Float64`` row.
        """
        frame = pl.DataFrame({COLUMN_X: pl.Series(COLUMN_X, [0.01, -0.02, 0.015, -0.03], dtype=pl.Float64)})
        result = frame.select(win_rate(pl.col(COLUMN_X)).alias("w"))
        assert result.height == 1
        assert result.schema["w"] == pl.Float64

    def test_lazy_eager_parity(self) -> None:
        """
        Verifies that eager and lazy application produce identical materialized output.
        """
        frame = pl.DataFrame({COLUMN_X: pl.Series(COLUMN_X, [0.01, -0.02, 0.015, -0.03], dtype=pl.Float64)})
        expr = win_rate(pl.col(COLUMN_X)).alias("w")
        assert_frame_equal(frame.select(expr), frame.lazy().select(expr).collect())

    def test_over_partitions_independently(self) -> None:
        """
        Verifies that under ``.over`` the win rate is computed per group (broadcast) and never spans boundaries.
        """
        group_a = [0.01, -0.02, 0.015, -0.03, 0.005, 0.04]
        group_b = [0.02, -0.05, 0.01, -0.01, 0.03]
        frame = pl.DataFrame({GROUP_KEY: ["a"] * len(group_a) + ["b"] * len(group_b), COLUMN_X: group_a + group_b})
        grouped = frame.select(win_rate(pl.col(COLUMN_X)).over(GROUP_KEY).alias("w"))["w"].to_list()
        expected_a = win_rate_reference(group_a)
        expected_b = win_rate_reference(group_b)
        assert_matches(
            grouped, [expected_a] * len(group_a) + [expected_b] * len(group_b), rel_tol=RELATIVE_TOLERANCE_REFERENCE
        )


class TestWinRateEdge:
    """
    Boundaries and null / NaN handling.
    """

    def test_empty(self) -> None:
        """
        Verifies that an empty series yields ``null``.
        """
        assert_matches(apply_expr([], win_rate(pl.col(COLUMN_X))), [None])

    def test_all_positive_is_one(self) -> None:
        """
        Verifies that an all-positive series wins every decisive period, so the rate is ``1``.
        """
        assert_matches(apply_expr([0.01, 0.02, 0.03], win_rate(pl.col(COLUMN_X))), [1.0])

    def test_all_negative_is_zero(self) -> None:
        """
        Verifies that an all-negative series wins no decisive period, so the rate is ``0``.
        """
        assert_matches(apply_expr([-0.01, -0.02, -0.03], win_rate(pl.col(COLUMN_X))), [0.0])

    def test_all_zero_is_null(self) -> None:
        """
        Verifies that an all-zero series has no decisive returns, so the rate is ``null``.
        """
        assert_matches(apply_expr([0.0, 0.0, 0.0], win_rate(pl.col(COLUMN_X))), [None])

    def test_zero_excluded_from_denominator(self) -> None:
        """
        Verifies that exact-zero returns are excluded: two wins out of two decisive (one flat) returns is ``1``.
        """
        assert_matches(apply_expr([0.01, 0.0, 0.02], win_rate(pl.col(COLUMN_X))), [1.0])

    def test_all_null(self) -> None:
        """
        Verifies that an all-null series yields ``null``.
        """
        assert_matches(apply_expr([None, None], win_rate(pl.col(COLUMN_X))), [None])

    def test_nan_poisons(self) -> None:
        """
        Verifies that a NaN return poisons the result to NaN.
        """
        assert_matches(apply_expr([0.01, math.nan, -0.02, 0.03], win_rate(pl.col(COLUMN_X))), [math.nan])

    def test_null_skipped(self) -> None:
        """
        Verifies that null returns are skipped, matching the reference.
        """
        values = [0.01, None, 0.02, -0.03, 0.04, None, -0.01]
        assert_matches(
            apply_expr(values, win_rate(pl.col(COLUMN_X))),
            [win_rate_reference(values)],
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
        )


class TestWinRateCorrectness:
    """
    Against the naive reference oracle and frozen golden-master values.
    """

    def test_matches_reference(self) -> None:
        """
        Verifies agreement with the naive closed-form reference over a representative return series.
        """
        values = [0.012, -0.008, 0.02, -0.015, 0.0, 0.005, -0.02, 0.018]
        assert_matches(
            apply_expr(values, win_rate(pl.col(COLUMN_X))),
            [win_rate_reference(values)],
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference: four wins out of seven decisive returns is 0.5714.
        """
        values = [0.03, -0.01, 0.02, -0.015, 0.01, 0.005, -0.02]
        assert_matches(apply_expr(values, win_rate(pl.col(COLUMN_X)).round(4)), [0.5714])


class TestWinRateProperties:
    """
    Invariants that must hold for all inputs (property-based).
    """

    @given(case=_cases(subnormal_safe_floats(bound=1e3)))
    def test_matches_reference_for_any_input(self, case: list[float]) -> None:
        """
        Verifies that, for any return series, the implementation matches the naive reference.
        """
        assert_matches(
            apply_expr(case, win_rate(pl.col(COLUMN_X))),
            [win_rate_reference(case)],
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    @given(case=_cases(missing_data_floats(min_magnitude=1e-3), min_size=0))
    def test_matches_reference_under_missing_data(self, case: list[float | None]) -> None:
        """
        Verifies that, for inputs freely mixing null / NaN / finite, the implementation matches the naive reference.
        """
        assert_matches(
            apply_expr(case, win_rate(pl.col(COLUMN_X))),
            [win_rate_reference(case)],
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    @given(case=_cases(subnormal_safe_floats(bound=1e3), min_size=2), exponent=st.sampled_from([-4, -2, -1, 1, 2, 4]))
    def test_scale_invariant(self, case: list[float], exponent: int) -> None:
        """
        Verifies that a positive rescale of the returns leaves the win rate unchanged (signs are preserved), using
        powers of two so the rescaling is lossless.
        """
        k = 2.0**exponent
        base = apply_expr(case, win_rate(pl.col(COLUMN_X)))
        scaled = apply_expr([value * k for value in case], win_rate(pl.col(COLUMN_X)))
        assert_matches(scaled, base, rel_tol=RELATIVE_TOLERANCE_SCALE, abs_tol=ABSOLUTE_TOLERANCE_REFERENCE)
