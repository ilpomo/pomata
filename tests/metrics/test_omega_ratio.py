"""
Tests for ``pomata.metrics.omega_ratio`` — the gain-to-loss ratio of a return series about a threshold.

``omega_ratio`` is single-input and REDUCING (a return series → one scalar), so tests read the single output row of
``apply_expr``; ``assert_matches`` and the naive ``omega_ratio_reference`` oracle (the mean gain over the mean loss) are
shared across the suite. At ``threshold = 0`` it is scale-invariant (a ratio of means), so it carries a scale-invariance
tier (run at the default threshold).

The ladder is the canonical one: contract (type / reduces-to-scalar / lazy-eager / ``.over`` per-group independence),
edge (validation / empty / single-row / no-downside / null / NaN), correctness (vs the closed-form reference and a
frozen golden master), and properties (reference agreement incl. missing data, scale invariance). Categories are split
into classes; cross-cutting categories use markers.
"""

import math

import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st
from tests.metrics.oracles import omega_ratio_reference
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

from pomata.metrics import omega_ratio

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- omega is windowless and REDUCING (M = 0); a case is just a return series. Facts:
#   1. shape   reducing: the output is one scalar (one row in ``select``; one per group under ``.over``)
#   2. domain  subnormal_safe_floats(bound): finite returns floored off subnormal so the mean ratio stays
#              well-conditioned; the missing variant mixes null / NaN
#   3. scale   invariant at threshold 0 (a ratio of means) -> scale-invariance tier (run at the default threshold)
# THRESHOLD varies over a realistic set in the reference fuzz. Repetitions N are the shared CI profile (conftest.py).
# ----------------------------------------------------------------------------------------------------------------------
SERIES_MAX = 50
_THRESHOLD = st.sampled_from([0.0, 0.01, -0.01])


@st.composite
def _cases[T](draw: st.DrawFn, returns: st.SearchStrategy[T], min_size: int = 1) -> list[T]:
    """A return series sized from the facts above."""
    return draw(st.lists(returns, min_size=min_size, max_size=SERIES_MAX))


class TestOmegaRatioContract:
    """
    Type, shape, and lazy/eager guarantees.
    """


class TestOmegaRatioEdge:
    """
    Validation, boundaries, and null / NaN handling.
    """

    def test_non_finite_threshold_raises(self) -> None:
        """
        Verifies that a non-finite ``threshold`` raises ``ValueError``.
        """
        for invalid in (math.nan, math.inf, -math.inf):
            with pytest.raises(ValueError, match="threshold must be a finite number"):
                omega_ratio(pl.col(COLUMN_X), threshold=invalid)

    def test_single_row(self) -> None:
        """
        Verifies that a one-element series puts all mass on one side of the threshold: a single gain has zero mean
        loss, so the ratio is ``+inf``.
        """
        assert_matches(apply_expr([0.05], omega_ratio(pl.col(COLUMN_X))), [math.inf])

    def test_null_skipped(self) -> None:
        """
        Verifies that null returns are skipped, matching the reference.
        """
        values = [0.01, None, 0.02, -0.03, 0.04, None, -0.01]
        assert_matches(
            apply_expr(values, omega_ratio(pl.col(COLUMN_X))),
            [omega_ratio_reference(values, 0.0)],
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
        )

    def test_nan_poisons(self) -> None:
        """
        Verifies that a NaN return poisons the result to NaN.
        """
        assert_matches(apply_expr([0.01, math.nan, -0.02, 0.03], omega_ratio(pl.col(COLUMN_X))), [math.nan])

    def test_all_gain_is_inf(self) -> None:
        """
        Verifies that returns all above the threshold have no downside, so the ratio is ``+inf``.
        """
        assert_matches(apply_expr([0.01, 0.02, 0.03], omega_ratio(pl.col(COLUMN_X))), [math.inf])

    def test_all_loss_is_zero(self) -> None:
        """
        Verifies that returns all below the threshold have no upside, so the ratio is ``0``.
        """
        assert_matches(apply_expr([-0.01, -0.02, -0.03], omega_ratio(pl.col(COLUMN_X))), [0.0])

    def test_all_at_threshold_is_nan(self) -> None:
        """
        Verifies that returns all exactly at the threshold give ``0 / 0``, so the ratio is ``NaN``.
        """
        assert_matches(apply_expr([0.0, 0.0, 0.0], omega_ratio(pl.col(COLUMN_X))), [math.nan])


class TestOmegaRatioCorrectness:
    """
    Against the naive reference oracle and frozen golden-master values.
    """

    def test_matches_reference(self) -> None:
        """
        Verifies agreement with the naive closed-form reference over a representative return series.
        """
        values = [0.012, -0.008, 0.02, -0.015, 0.005, -0.02, 0.018]
        assert_matches(
            apply_expr(values, omega_ratio(pl.col(COLUMN_X), threshold=0.01)),
            [omega_ratio_reference(values, 0.01)],
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference: the mean gain (0.0093) over the mean loss (0.0064) about zero is 1.4444.
        """
        values = [0.03, -0.01, 0.02, -0.015, 0.01, 0.005, -0.02]
        assert_matches(apply_expr(values, omega_ratio(pl.col(COLUMN_X)).round(4)), [1.4444])


class TestOmegaRatioProperties:
    """
    Invariants that must hold for all inputs (property-based).
    """

    @given(case=_cases(subnormal_safe_floats(bound=1e3)), threshold=_THRESHOLD)
    def test_matches_reference_for_any_input(self, case: list[float], threshold: float) -> None:
        """
        Verifies that, for any return series, the implementation matches the naive reference.
        """
        assert_matches(
            apply_expr(case, omega_ratio(pl.col(COLUMN_X), threshold=threshold)),
            [omega_ratio_reference(case, threshold)],
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    @given(case=_cases(missing_data_floats(min_magnitude=1e-3), min_size=0), threshold=_THRESHOLD)
    def test_matches_reference_under_missing_data(self, case: list[float | None], threshold: float) -> None:
        """
        Verifies that, for inputs freely mixing null / NaN / finite, the implementation matches the naive reference.
        """
        assert_matches(
            apply_expr(case, omega_ratio(pl.col(COLUMN_X), threshold=threshold)),
            [omega_ratio_reference(case, threshold)],
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    @given(
        case=_cases(subnormal_safe_floats(bound=1e3), min_size=2),
        exponent=st.sampled_from([-4, -3, -2, -1, 1, 2, 3, 4]),
    )
    def test_scale_invariance(self, case: list[float], exponent: int) -> None:
        """
        Verifies that ``omega_ratio`` is scale-invariant: scaling every input value by a constant ``k`` leaves the
        output unchanged -- ``omega_ratio(k * x) == omega_ratio(x)``. ``k`` is a power of two, so the rescale is
        exact and adds no floating-point error.
        """
        k = 2.0**exponent
        base = apply_expr(case, omega_ratio(pl.col(COLUMN_X)))
        scaled = apply_expr([value * k for value in case], omega_ratio(pl.col(COLUMN_X)))
        assert_matches(scaled, base, rel_tol=RELATIVE_TOLERANCE_SCALE, abs_tol=ABSOLUTE_TOLERANCE_REFERENCE)
