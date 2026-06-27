"""
Tests for ``pomata.metrics.common_sense_ratio`` — the profit factor scaled by the tail ratio.

``common_sense_ratio`` is single-input and REDUCING (a return series → one scalar), so tests read the single output row
of ``apply_expr``; ``assert_matches`` and the naive ``common_sense_ratio_reference`` oracle are shared across the suite.
It is scale-invariant (a product of two scale-invariant factors), so it carries a scale-invariance tier.

The ladder is the canonical one: contract (type / reduces-to-scalar / lazy-eager / ``.over`` per-group independence),
edge (empty / single-row / no-losses / null / NaN), correctness (vs the closed-form reference and a frozen golden
master), and properties (reference agreement incl. missing data, scale invariance). Categories are split into classes;
cross-cutting categories use markers.
"""

import math

import polars as pl
from hypothesis import given
from hypothesis import strategies as st
from polars.testing import assert_frame_equal
from tests.metrics.oracles import common_sense_ratio_reference
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

from pomata.metrics import common_sense_ratio

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- common_sense_ratio is windowless and REDUCING (M = 0); a case is just a return series. Facts:
#   1. shape   reducing: the output is one scalar (one row in ``select``; one per group under ``.over``)
#   2. domain  subnormal_safe_floats(bound): finite returns floored off subnormal; the missing variant mixes null / NaN
#   3. scale   invariant (a product of two scale-invariant factors) -> scale-invariance tier
# Repetitions N are the shared CI profile (tests/conftest.py).
# ----------------------------------------------------------------------------------------------------------------------
SERIES_MAX = 50


@st.composite
def _cases[T](draw: st.DrawFn, returns: st.SearchStrategy[T], min_size: int = 1) -> list[T]:
    """A return series sized from the facts above."""
    return draw(st.lists(returns, min_size=min_size, max_size=SERIES_MAX))


class TestCommonSenseRatioContract:
    """
    Type, shape, and lazy/eager guarantees.
    """

    def test_returns_expr(self) -> None:
        """
        Verifies that the factory returns a ``pl.Expr`` without touching a frame.
        """
        assert isinstance(common_sense_ratio(pl.col(COLUMN_X)), pl.Expr)

    def test_reduces_to_scalar(self) -> None:
        """
        Verifies that the metric reduces a series to one ``Float64`` row.
        """
        frame = pl.DataFrame({COLUMN_X: pl.Series(COLUMN_X, [0.01, -0.02, 0.015, -0.03], dtype=pl.Float64)})
        result = frame.select(common_sense_ratio(pl.col(COLUMN_X)).alias("c"))
        assert result.height == 1
        assert result.schema["c"] == pl.Float64

    def test_lazy_eager_parity(self) -> None:
        """
        Verifies that eager and lazy application produce identical materialized output.
        """
        frame = pl.DataFrame({COLUMN_X: pl.Series(COLUMN_X, [0.01, -0.02, 0.015, -0.03], dtype=pl.Float64)})
        expr = common_sense_ratio(pl.col(COLUMN_X)).alias("c")
        assert_frame_equal(frame.select(expr), frame.lazy().select(expr).collect())

    def test_over_partitions_independently(self) -> None:
        """
        Verifies that under ``.over`` the ratio is computed per group (broadcast) and never spans boundaries.
        """
        group_a = [0.01, -0.02, 0.015, -0.03, 0.005, 0.04]
        group_b = [0.02, -0.05, 0.01, -0.01, 0.03]
        frame = pl.DataFrame({GROUP_KEY: ["a"] * len(group_a) + ["b"] * len(group_b), COLUMN_X: group_a + group_b})
        grouped = frame.select(common_sense_ratio(pl.col(COLUMN_X)).over(GROUP_KEY).alias("c"))["c"].to_list()
        expected_a = common_sense_ratio_reference(group_a)
        expected_b = common_sense_ratio_reference(group_b)
        assert_matches(
            grouped, [expected_a] * len(group_a) + [expected_b] * len(group_b), rel_tol=RELATIVE_TOLERANCE_REFERENCE
        )


class TestCommonSenseRatioEdge:
    """
    Boundaries and null / NaN handling.
    """

    def test_empty(self) -> None:
        """
        Verifies that an empty series yields ``null``.
        """
        assert_matches(apply_expr([], common_sense_ratio(pl.col(COLUMN_X))), [None])

    def test_no_losses_is_inf(self) -> None:
        """
        Verifies that an all-positive series has no loss, so the profit factor diverges and the ratio is ``+inf``.
        """
        assert_matches(apply_expr([0.01, 0.02, 0.03], common_sense_ratio(pl.col(COLUMN_X))), [math.inf])

    def test_all_null(self) -> None:
        """
        Verifies that an all-null series yields ``null``.
        """
        assert_matches(apply_expr([None, None], common_sense_ratio(pl.col(COLUMN_X))), [None])

    def test_nan_poisons(self) -> None:
        """
        Verifies that a NaN return poisons the result to NaN.
        """
        assert_matches(apply_expr([0.01, math.nan, -0.02, 0.03], common_sense_ratio(pl.col(COLUMN_X))), [math.nan])

    def test_null_skipped(self) -> None:
        """
        Verifies that null returns are skipped, matching the reference.
        """
        values = [0.01, None, 0.02, -0.03, 0.04, None, -0.01]
        assert_matches(
            apply_expr(values, common_sense_ratio(pl.col(COLUMN_X))),
            [common_sense_ratio_reference(values)],
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
        )


class TestCommonSenseRatioCorrectness:
    """
    Against the naive reference oracle and frozen golden-master values.
    """

    def test_matches_reference(self) -> None:
        """
        Verifies agreement with the naive closed-form reference over a representative return series.
        """
        values = [0.012, -0.008, 0.02, -0.015, 0.005, -0.02, 0.018, 0.01]
        assert_matches(
            apply_expr(values, common_sense_ratio(pl.col(COLUMN_X))),
            [common_sense_ratio_reference(values)],
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference: profit factor (1.4444) times tail ratio (1.4595) is 2.1081.
        """
        values = [0.03, -0.01, 0.02, -0.015, 0.01, 0.005, -0.02]
        assert_matches(apply_expr(values, common_sense_ratio(pl.col(COLUMN_X)).round(4)), [2.1081])


class TestCommonSenseRatioProperties:
    """
    Invariants that must hold for all inputs (property-based).
    """

    @given(case=_cases(subnormal_safe_floats(bound=1e3)))
    def test_matches_reference_for_any_input(self, case: list[float]) -> None:
        """
        Verifies that, for any return series, the implementation matches the naive reference.
        """
        assert_matches(
            apply_expr(case, common_sense_ratio(pl.col(COLUMN_X))),
            [common_sense_ratio_reference(case)],
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    @given(case=_cases(missing_data_floats(min_magnitude=1e-3), min_size=0))
    def test_matches_reference_under_missing_data(self, case: list[float | None]) -> None:
        """
        Verifies that, for inputs freely mixing null / NaN / finite, the implementation matches the naive reference.
        """
        assert_matches(
            apply_expr(case, common_sense_ratio(pl.col(COLUMN_X))),
            [common_sense_ratio_reference(case)],
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    @given(case=_cases(subnormal_safe_floats(bound=1e3), min_size=2), exponent=st.sampled_from([-4, -2, -1, 1, 2, 4]))
    def test_scale_invariant(self, case: list[float], exponent: int) -> None:
        """
        Verifies that a positive rescale of the returns leaves the common sense ratio unchanged (a product of
        scale-invariant factors), using powers of two so the rescaling is lossless.
        """
        k = 2.0**exponent
        base = apply_expr(case, common_sense_ratio(pl.col(COLUMN_X)))
        scaled = apply_expr([value * k for value in case], common_sense_ratio(pl.col(COLUMN_X)))
        assert_matches(scaled, base, rel_tol=RELATIVE_TOLERANCE_SCALE, abs_tol=ABSOLUTE_TOLERANCE_REFERENCE)
