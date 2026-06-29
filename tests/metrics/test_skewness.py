"""
Tests for ``pomata.metrics.skewness`` — the asymmetry (standardized third moment) of a return series.

``skewness`` is single-input and REDUCING (a return series → one scalar), so tests read the single output row of
``apply_expr``; ``assert_matches`` and the naive ``skewness_reference`` oracle (the population moments) are shared
across the suite. It is scale-invariant (a standardized moment), so it carries a scale-invariance tier; being a ratio
of moments it is conditioning-sensitive, so the property comparisons use the scale band.

The ladder is the canonical one: contract (type / reduces-to-scalar / lazy-eager / ``.over`` per-group independence),
edge (empty / single-row / constant / null / NaN), correctness (vs the closed-form reference and a frozen golden
master), and properties (reference agreement incl. missing data, scale invariance). Categories are split into classes.
"""

import math

import polars as pl
from hypothesis import assume, given
from hypothesis import strategies as st
from polars.testing import assert_frame_equal
from tests.metrics.oracles import skewness_reference
from tests.support import (
    ABSOLUTE_TOLERANCE_REFERENCE,
    COLUMN_X,
    GROUP_KEY,
    RELATIVE_TOLERANCE_REFERENCE,
    RELATIVE_TOLERANCE_SCALE,
    STANDARDIZED_MOMENT_FLOOR,
    apply_expr,
    assert_matches,
    missing_data_floats,
    standardized_moment_floats,
    well_spread,
)

from pomata.metrics import skewness

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- skewness is windowless and REDUCING (M = 0); a case is just a return series. Facts:
#   1. shape   reducing: the output is one scalar (one row in ``select``; one per group under ``.over``)
#   2. domain  standardized_moment_floats: spread returns floored off subnormal (m2**1.5 underflows there); +null/NaN
#   3. scale   invariant (a standardized moment) -> scale-invariance tier; conditioning-sensitive -> the scale band
# Repetitions N are the shared CI profile (tests/conftest.py).
# ----------------------------------------------------------------------------------------------------------------------
SERIES_MAX = 50


@st.composite
def _cases[T](draw: st.DrawFn, returns: st.SearchStrategy[T], min_size: int = 1) -> list[T]:
    """A return series sized from the facts above."""
    return draw(st.lists(returns, min_size=min_size, max_size=SERIES_MAX))


class TestSkewnessContract:
    """
    Type, shape, and lazy/eager guarantees.
    """

    def test_returns_expr(self) -> None:
        """
        Verifies that the factory returns a ``pl.Expr`` without touching a frame.
        """
        assert isinstance(skewness(pl.col(COLUMN_X)), pl.Expr)

    def test_reduces_to_scalar(self) -> None:
        """
        Verifies that the metric reduces a series to one ``Float64`` row.
        """
        frame = pl.DataFrame({COLUMN_X: pl.Series(COLUMN_X, [0.01, -0.02, 0.015, -0.03], dtype=pl.Float64)})
        result = frame.select(skewness(pl.col(COLUMN_X)).alias("s"))
        assert result.height == 1
        assert result.schema["s"] == pl.Float64

    def test_lazy_eager_parity(self) -> None:
        """
        Verifies that eager and lazy application produce identical materialized output.
        """
        frame = pl.DataFrame({COLUMN_X: pl.Series(COLUMN_X, [0.01, -0.02, 0.015, -0.03], dtype=pl.Float64)})
        expr = skewness(pl.col(COLUMN_X)).alias("s")
        assert_frame_equal(frame.select(expr), frame.lazy().select(expr).collect())

    def test_over_partitions_independently(self) -> None:
        """
        Verifies that under ``.over`` the skewness is computed per group (broadcast) and never spans boundaries.
        """
        group_a = [0.01, -0.02, 0.015, -0.03, 0.005, 0.04]
        group_b = [0.02, -0.05, 0.01, -0.01, 0.03]
        frame = pl.DataFrame({GROUP_KEY: ["a"] * len(group_a) + ["b"] * len(group_b), COLUMN_X: group_a + group_b})
        grouped = frame.select(skewness(pl.col(COLUMN_X)).over(GROUP_KEY).alias("s"))["s"].to_list()
        expected_a = skewness_reference(group_a)
        expected_b = skewness_reference(group_b)
        assert_matches(
            grouped, [expected_a] * len(group_a) + [expected_b] * len(group_b), rel_tol=RELATIVE_TOLERANCE_SCALE
        )


class TestSkewnessEdge:
    """
    Boundaries and null / NaN handling.
    """

    def test_empty(self) -> None:
        """
        Verifies that an empty series yields ``null``.
        """
        assert_matches(apply_expr([], skewness(pl.col(COLUMN_X))), [None])

    def test_single_row_is_nan(self) -> None:
        """
        Verifies that a one-element series has zero variance, so the standardized moment is NaN.
        """
        assert_matches(apply_expr([0.05], skewness(pl.col(COLUMN_X))), [math.nan])

    def test_constant_is_nan(self) -> None:
        """
        Verifies that a constant series has zero variance, so the skewness is NaN.
        """
        assert_matches(apply_expr([0.01, 0.01, 0.01], skewness(pl.col(COLUMN_X))), [math.nan])

    def test_subnormal_magnitude_is_nan(self) -> None:
        """
        Verifies that a subnormal-magnitude series, whose ``m2 ** 1.5`` underflows to zero, yields NaN -- the degenerate
        the property tier floors away from, pinned deterministically here.
        """
        assert_matches(apply_expr([0.0, 1e-160, 2e-160], skewness(pl.col(COLUMN_X))), [math.nan])

    def test_all_null(self) -> None:
        """
        Verifies that an all-null series yields ``null``.
        """
        assert_matches(apply_expr([None, None], skewness(pl.col(COLUMN_X))), [None])

    def test_nan_poisons(self) -> None:
        """
        Verifies that a NaN return poisons the result to NaN.
        """
        assert_matches(apply_expr([0.01, math.nan, 0.02, 0.03], skewness(pl.col(COLUMN_X))), [math.nan])

    def test_null_skipped(self) -> None:
        """
        Verifies that null returns are skipped, matching the reference.
        """
        values = [0.01, None, 0.02, -0.03, 0.04, None, -0.01]
        assert_matches(
            apply_expr(values, skewness(pl.col(COLUMN_X))),
            [skewness_reference(values)],
            rel_tol=RELATIVE_TOLERANCE_SCALE,
        )


class TestSkewnessCorrectness:
    """
    Against the naive reference oracle and frozen golden-master values.
    """

    def test_matches_reference(self) -> None:
        """
        Verifies agreement with the naive closed-form reference over a representative return series.
        """
        values = [0.01, -0.02, 0.015, -0.03, 0.005, -0.01, 0.02]
        assert_matches(
            apply_expr(values, skewness(pl.col(COLUMN_X))),
            [skewness_reference(values)],
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference: a left-leaning return series has negative skewness.
        """
        values = [0.01, -0.02, 0.015, -0.03, 0.005, -0.01, 0.02]
        assert_matches(apply_expr(values, skewness(pl.col(COLUMN_X)).round(4)), [-0.384])


class TestSkewnessProperties:
    """
    Invariants that must hold for all inputs (property-based).
    """

    @given(case=_cases(standardized_moment_floats(bound=1e3)))
    def test_matches_reference_for_any_input(self, case: list[float]) -> None:
        """
        Verifies that, for any well-conditioned return series, the implementation matches the naive reference.
        """
        assume(well_spread(case))
        assert_matches(
            apply_expr(case, skewness(pl.col(COLUMN_X))),
            [skewness_reference(case)],
            rel_tol=RELATIVE_TOLERANCE_SCALE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    @given(case=_cases(missing_data_floats(min_magnitude=STANDARDIZED_MOMENT_FLOOR), min_size=0))
    def test_matches_reference_under_missing_data(self, case: list[float | None]) -> None:
        """
        Verifies that, for well-conditioned inputs freely mixing null / NaN / finite, the implementation matches the
        naive reference.
        """
        assume(well_spread(case))
        assert_matches(
            apply_expr(case, skewness(pl.col(COLUMN_X))),
            [skewness_reference(case)],
            rel_tol=RELATIVE_TOLERANCE_SCALE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    @given(
        case=_cases(standardized_moment_floats(bound=1e3), min_size=2),
        exponent=st.sampled_from([-4, -2, -1, 1, 2, 4]),
    )
    def test_scale_invariance(self, case: list[float], exponent: int) -> None:
        """
        Verifies that a positive rescale of the returns leaves the skewness unchanged (a standardized moment), using
        powers of two so the rescaling is lossless.
        """
        k = 2.0**exponent
        base = apply_expr(case, skewness(pl.col(COLUMN_X)))
        scaled = apply_expr([value * k for value in case], skewness(pl.col(COLUMN_X)))
        assert_matches(scaled, base, rel_tol=RELATIVE_TOLERANCE_SCALE, abs_tol=ABSOLUTE_TOLERANCE_REFERENCE)
