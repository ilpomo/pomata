"""
Tests for ``pomata.metrics.tail_ratio`` — the right-tail-to-left-tail ratio of a return series.

``tail_ratio`` is single-input and REDUCING (a return series → one scalar), so tests read the single output row of
``apply_expr``; ``assert_matches`` and the naive ``tail_ratio_reference`` oracle (the type-7 quantile ratio) are shared
across the suite. It is scale-invariant (a ratio of two quantiles), so it carries a scale-invariance tier; unlike the
moment ratios it has no near-constant artifact (order statistics need no mean), so no conditioning filter is needed.

The ladder is the canonical one: contract (type / reduces-to-scalar / lazy-eager / ``.over`` per-group independence),
edge (empty / single-row / constant / zero-tail / null / NaN), correctness (vs the closed-form reference and a frozen
golden master), and properties (reference agreement incl. missing data, scale invariance). Categories are split into
classes; cross-cutting categories use markers.
"""

import math

import polars as pl
from hypothesis import given
from hypothesis import strategies as st
from tests.metrics.oracles import tail_ratio_reference
from tests.support import (
    ABSOLUTE_TOLERANCE_REFERENCE,
    COLUMN_X,
    RELATIVE_TOLERANCE_PROPERTY,
    RELATIVE_TOLERANCE_REFERENCE,
    RELATIVE_TOLERANCE_SCALE,
    apply_expr,
    assert_matches,
    missing_data_floats,
    subnormal_safe_floats,
)

from pomata.metrics import tail_ratio

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- tail_ratio is windowless and REDUCING (M = 0); a case is just a return series. Facts:
#   1. shape   reducing: the output is one scalar (one row in ``select``; one per group under ``.over``)
#   2. domain  subnormal_safe_floats(bound): finite returns floored off subnormal so the power-of-two rescale stays
#              bit-exact; the missing variant mixes null / NaN
#   3. scale   invariant (a ratio of quantiles) -> scale-invariance tier; no mean, so no conditioning filter
# Repetitions N are the shared CI profile (tests/conftest.py).
# ----------------------------------------------------------------------------------------------------------------------
SERIES_MAX = 50


@st.composite
def _cases[T](draw: st.DrawFn, returns: st.SearchStrategy[T], min_size: int = 1) -> list[T]:
    """A return series sized from the facts above."""
    return draw(st.lists(returns, min_size=min_size, max_size=SERIES_MAX))


class TestTailRatioContract:
    """
    Type, shape, and lazy/eager guarantees.
    """


class TestTailRatioEdge:
    """
    Boundaries and null / NaN handling.
    """

    def test_single_row_is_one(self) -> None:
        """
        Verifies that a one-element series has equal tails, so the ratio is ``1``.
        """
        assert_matches(apply_expr([0.05], tail_ratio(pl.col(COLUMN_X))), [1.0])

    def test_constant_is_one(self) -> None:
        """
        Verifies that a constant (non-zero) series has equal tails, so the ratio is ``1``.
        """
        assert_matches(apply_expr([0.01, 0.01, 0.01], tail_ratio(pl.col(COLUMN_X))), [1.0])

    def test_zero_left_tail_is_inf(self) -> None:
        """
        Verifies that a zero 5th-percentile with a non-zero 95th-percentile gives ``+inf`` (reported, not clipped).
        """
        assert_matches(apply_expr([0.0, 0.0, 0.0, 0.0, 0.02], tail_ratio(pl.col(COLUMN_X))), [math.inf])

    def test_all_zero_is_nan(self) -> None:
        """
        Verifies that an all-zero series gives ``0 / 0``, so the ratio is ``NaN``.
        """
        assert_matches(apply_expr([0.0, 0.0, 0.0], tail_ratio(pl.col(COLUMN_X))), [math.nan])

    def test_nan_poisons(self) -> None:
        """
        Verifies that a NaN return poisons the result to NaN.
        """
        assert_matches(apply_expr([0.01, math.nan, 0.02, 0.03], tail_ratio(pl.col(COLUMN_X))), [math.nan])

    def test_null_skipped(self) -> None:
        """
        Verifies that null returns are skipped, matching the reference.
        """
        values = [0.01, None, 0.02, -0.03, 0.04, None, -0.01]
        assert_matches(
            apply_expr(values, tail_ratio(pl.col(COLUMN_X))),
            [tail_ratio_reference(values)],
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
        )


class TestTailRatioCorrectness:
    """
    Against the naive reference oracle and frozen golden-master values.
    """

    def test_matches_reference(self) -> None:
        """
        Verifies agreement with the naive closed-form reference over a representative return series.
        """
        values = [0.012, -0.008, 0.02, -0.015, 0.005, 0.01, -0.02, 0.018, -0.03, 0.01]
        assert_matches(
            apply_expr(values, tail_ratio(pl.col(COLUMN_X))),
            [tail_ratio_reference(values)],
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference: the right tail (0.028) is half the left tail magnitude (0.056), so the ratio is
        0.5.
        """
        values = [0.02, -0.04, 0.01, -0.06, 0.03]
        assert_matches(apply_expr(values, tail_ratio(pl.col(COLUMN_X)).round(4)), [0.5])


class TestTailRatioProperties:
    """
    Invariants that must hold for all inputs (property-based).
    """

    @given(case=_cases(subnormal_safe_floats(bound=1e3)))
    def test_matches_reference_for_any_input(self, case: list[float]) -> None:
        """
        Verifies that, for any return series, the implementation matches the naive reference.
        """
        assert_matches(
            apply_expr(case, tail_ratio(pl.col(COLUMN_X))),
            [tail_ratio_reference(case)],
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    @given(case=_cases(missing_data_floats(min_magnitude=1e-3), min_size=0))
    def test_matches_reference_under_missing_data(self, case: list[float | None]) -> None:
        """
        Verifies that, for inputs freely mixing null / NaN / finite, the implementation matches the naive reference.
        """
        assert_matches(
            apply_expr(case, tail_ratio(pl.col(COLUMN_X))),
            [tail_ratio_reference(case)],
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    @given(case=_cases(subnormal_safe_floats(bound=1e3), min_size=2), exponent=st.sampled_from([-4, -2, -1, 1, 2, 4]))
    def test_scale_invariance(self, case: list[float], exponent: int) -> None:
        """
        Verifies that a positive rescale of the returns leaves the tail ratio unchanged (a ratio of quantiles), using
        powers of two so the rescaling is lossless.
        """
        k = 2.0**exponent
        base = apply_expr(case, tail_ratio(pl.col(COLUMN_X)))
        scaled = apply_expr([value * k for value in case], tail_ratio(pl.col(COLUMN_X)))
        assert_matches(scaled, base, rel_tol=RELATIVE_TOLERANCE_SCALE, abs_tol=ABSOLUTE_TOLERANCE_REFERENCE)
