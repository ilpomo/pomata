"""
Tests for ``pomata.metrics.value_at_risk_parametric`` — the Gaussian (variance-covariance) value-at-risk.

``value_at_risk_parametric`` is single-input and REDUCING (a return series → one scalar), so tests read the single
output row of ``apply_expr``; ``assert_matches`` and the naive ``value_at_risk_parametric_reference`` oracle (the normal
quantile of the return distribution) are shared across the suite. It is degree-1 homogeneous in the returns, so it
carries the scale-homogeneity and large-magnitude tiers.

The ladder is the canonical one: contract (type / reduces-to-scalar / lazy-eager / ``.over`` per-group independence),
edge (validation / empty / single-row / constant / null / NaN), correctness (vs the closed-form reference and a frozen
golden master), and properties (reference agreement incl. missing data, degree-1 scale-homogeneity, large-magnitude
stability). Categories are split into classes; cross-cutting categories use markers.
"""

import math
from collections.abc import Sequence

import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st
from tests.metrics.oracles import value_at_risk_parametric_reference
from tests.support import (
    COLUMN_X,
    RELATIVE_TOLERANCE_PROPERTY,
    RELATIVE_TOLERANCE_REFERENCE,
    RELATIVE_TOLERANCE_SCALE,
    apply_expr,
    assert_matches,
    assert_scale_homogeneous,
    missing_data_floats,
    streaming_abs_tol,
    subnormal_safe_floats,
)

from pomata.metrics import value_at_risk_parametric

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- value_at_risk_parametric is windowless and REDUCING (M = 0); a case is just a return series. Facts:
#   1. shape   reducing: the output is one scalar (one row in ``select``; one per group under ``.over``)
#   2. domain  subnormal_safe_floats(bound): finite returns floored off subnormal; the missing variant mixes null / NaN
#   3. scale   degree-1 homogeneous (mean + z * standard deviation) -> scale-homogeneity + large-magnitude tiers
# CONFIDENCE varies over a realistic set in the fuzz. Repetitions N are the shared CI profile (tests/conftest.py).
# ----------------------------------------------------------------------------------------------------------------------
SERIES_MAX = 50
CONFIDENCE = 0.95
_CONFIDENCE = st.sampled_from([0.9, 0.95, 0.99])


@st.composite
def _cases[T](draw: st.DrawFn, returns: st.SearchStrategy[T], min_size: int = 2) -> list[T]:
    """A return series sized from the facts above (at least two for the sample standard deviation)."""
    return draw(st.lists(returns, min_size=min_size, max_size=SERIES_MAX))


def _abs_tol(values: Sequence[float | None]) -> float:
    """The magnitude-relative absolute tolerance for a quantile on the returns' own scale."""
    return streaming_abs_tol(values)


class TestValueAtRiskParametricContract:
    """
    Type, shape, and lazy/eager guarantees.
    """


class TestValueAtRiskParametricEdge:
    """
    Validation, boundaries, and null / NaN handling.
    """

    def test_confidence_out_of_range_raises(self) -> None:
        """
        Verifies that a ``confidence`` outside the open interval ``(0, 1)`` raises ``ValueError``.
        """
        for invalid in (0.0, 1.0, -0.1, 1.5):
            with pytest.raises(ValueError, match="confidence must be in the open interval"):
                value_at_risk_parametric(pl.col(COLUMN_X), confidence=invalid)

    def test_single_row(self) -> None:
        """
        Verifies that a one-element series yields ``null`` (the sample standard deviation needs two observations).
        """
        assert_matches(apply_expr([0.05], value_at_risk_parametric(pl.col(COLUMN_X), confidence=CONFIDENCE)), [None])

    def test_null_skipped(self) -> None:
        """
        Verifies that ``null`` returns are skipped (excluded from the value-at-risk), matching the reference.
        """
        values = [0.01, None, 0.02, 0.03, None]
        assert_matches(
            apply_expr(values, value_at_risk_parametric(pl.col(COLUMN_X), confidence=CONFIDENCE)),
            [value_at_risk_parametric_reference(values, CONFIDENCE)],
            abs_tol=_abs_tol(values),
        )

    def test_nan_poisons(self) -> None:
        """
        Verifies that a NaN return poisons the result to NaN.
        """
        values = [0.01, math.nan, -0.02, 0.03]
        assert_matches(
            apply_expr(values, value_at_risk_parametric(pl.col(COLUMN_X), confidence=CONFIDENCE)), [math.nan]
        )

    def test_constant_is_mean(self) -> None:
        """
        Verifies that a constant series has zero dispersion, so the value-at-risk is the mean itself.
        """
        assert_matches(
            apply_expr([0.01, 0.01, 0.01], value_at_risk_parametric(pl.col(COLUMN_X), confidence=CONFIDENCE)), [0.01]
        )


class TestValueAtRiskParametricCorrectness:
    """
    Against the naive reference oracle and frozen golden-master values.
    """

    def test_matches_reference(self) -> None:
        """
        Verifies agreement with the naive closed-form reference over a representative return series.
        """
        values = [0.012, -0.008, 0.02, -0.015, 0.005, 0.0, -0.02, 0.018, -0.03, 0.01]
        assert_matches(
            apply_expr(values, value_at_risk_parametric(pl.col(COLUMN_X), confidence=CONFIDENCE)),
            [value_at_risk_parametric_reference(values, CONFIDENCE)],
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
            abs_tol=_abs_tol(values),
        )

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference: the Gaussian 5% value-at-risk of the series is -0.0732.
        """
        values = [0.02, -0.04, 0.01, -0.06, 0.03]
        assert_matches(
            apply_expr(values, value_at_risk_parametric(pl.col(COLUMN_X), confidence=0.95).round(4)), [-0.0732]
        )


class TestValueAtRiskParametricProperties:
    """
    Invariants that must hold for all inputs (property-based).
    """

    @given(case=_cases(subnormal_safe_floats(bound=1e3)), confidence=_CONFIDENCE)
    def test_matches_reference_for_any_input(self, case: list[float], confidence: float) -> None:
        """
        Verifies that, for any return series, the implementation matches the naive reference.
        """
        assert_matches(
            apply_expr(case, value_at_risk_parametric(pl.col(COLUMN_X), confidence=confidence)),
            [value_at_risk_parametric_reference(case, confidence)],
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=_abs_tol(case),
        )

    @given(case=_cases(missing_data_floats(min_magnitude=1e-3), min_size=0), confidence=_CONFIDENCE)
    def test_matches_reference_under_missing_data(self, case: list[float | None], confidence: float) -> None:
        """
        Verifies that, for inputs freely mixing null / NaN / finite, the implementation matches the naive reference.
        """
        assert_matches(
            apply_expr(case, value_at_risk_parametric(pl.col(COLUMN_X), confidence=confidence)),
            [value_at_risk_parametric_reference(case, confidence)],
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=_abs_tol(case),
        )

    @given(case=_cases(subnormal_safe_floats(bound=1e3)), exponent=st.sampled_from([-4, -3, -2, -1, 1, 2, 3, 4]))
    def test_scale_homogeneity(self, case: list[float], exponent: int) -> None:
        """
        Verifies that ``value_at_risk_parametric`` is homogeneous of degree 1: scaling every input value by a
        constant ``k`` scales the output by the same ``k`` -- ``value_at_risk_parametric(k * x) == k *
        value_at_risk_parametric(x)``. ``k`` is a power of two, so the rescale is exact and adds no floating-point
        error.
        """
        k = 2.0**exponent
        base = apply_expr(case, value_at_risk_parametric(pl.col(COLUMN_X), confidence=CONFIDENCE))
        scaled = apply_expr(
            [value * k for value in case], value_at_risk_parametric(pl.col(COLUMN_X), confidence=CONFIDENCE)
        )
        assert_scale_homogeneous(scaled, base, k=k, degree=1)

    @given(case=_cases(subnormal_safe_floats(bound=1e3)), scale=st.sampled_from([1e-6, 1e6, 1e9]))
    def test_matches_reference_at_large_magnitude(self, case: list[float], scale: float) -> None:
        """
        Verifies that at extreme magnitudes the implementation stays finite where the reference is and agrees.
        """
        scaled = [value * scale for value in case]
        assert_matches(
            apply_expr(scaled, value_at_risk_parametric(pl.col(COLUMN_X), confidence=CONFIDENCE)),
            [value_at_risk_parametric_reference(scaled, CONFIDENCE)],
            rel_tol=RELATIVE_TOLERANCE_SCALE,
            abs_tol=_abs_tol(scaled),
        )
