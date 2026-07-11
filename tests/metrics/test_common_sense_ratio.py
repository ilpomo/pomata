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
from tests.metrics.oracles import common_sense_ratio_reference
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

from pomata.metrics import common_sense_ratio, profit_factor, tail_ratio

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


class TestCommonSenseRatioEdge:
    """
    Boundaries and null / NaN handling.
    """

    def test_single_row(self) -> None:
        """
        Verifies that a one-element negative series has no gain, so the profit factor is ``0`` and (with a tail ratio
        of ``1``) the ratio is ``0``.
        """
        assert_matches(apply_expr([-0.02], common_sense_ratio(pl.col(COLUMN_X))), [0.0])

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

    def test_nan_poisons(self) -> None:
        """
        Verifies that a NaN return poisons the result to NaN.
        """
        assert_matches(apply_expr([0.01, math.nan, -0.02, 0.03], common_sense_ratio(pl.col(COLUMN_X))), [math.nan])

    def test_no_losses_is_inf(self) -> None:
        """
        Verifies that an all-positive series has no loss, so the profit factor diverges and the ratio is ``+inf``.
        """
        assert_matches(apply_expr([0.01, 0.02, 0.03], common_sense_ratio(pl.col(COLUMN_X))), [math.inf])


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

    @given(
        case=_cases(subnormal_safe_floats(bound=1e3), min_size=2),
        exponent=st.sampled_from([-4, -3, -2, -1, 1, 2, 3, 4]),
    )
    def test_scale_invariance(self, case: list[float], exponent: int) -> None:
        """
        Verifies that ``common_sense_ratio`` is scale-invariant: scaling every input value by a constant ``k``
        leaves the output unchanged -- ``common_sense_ratio(k * x) == common_sense_ratio(x)``. ``k`` is a power of
        two, so the rescale is exact and adds no floating-point error.
        """
        k = 2.0**exponent
        base = apply_expr(case, common_sense_ratio(pl.col(COLUMN_X)))
        scaled = apply_expr([value * k for value in case], common_sense_ratio(pl.col(COLUMN_X)))
        assert_matches(scaled, base, rel_tol=RELATIVE_TOLERANCE_SCALE, abs_tol=ABSOLUTE_TOLERANCE_REFERENCE)

    @given(case=_cases(subnormal_safe_floats(bound=1e3)))
    def test_matches_component_definition(self, case: list[float]) -> None:
        """
        Verifies the metamorphic identity: ``common_sense_ratio`` equals :func:`profit_factor` times
        :func:`tail_ratio`, computed as separate metrics.
        """
        direct = apply_expr(case, common_sense_ratio(pl.col(COLUMN_X)))
        composed = apply_expr(case, profit_factor(pl.col(COLUMN_X)) * tail_ratio(pl.col(COLUMN_X)))
        assert_matches(direct, composed, rel_tol=RELATIVE_TOLERANCE_PROPERTY, abs_tol=ABSOLUTE_TOLERANCE_REFERENCE)
