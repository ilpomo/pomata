"""
Tests for ``pomata.metrics.conditional_value_at_risk`` — the historical expected shortfall of a return series.

``conditional_value_at_risk`` is single-input and REDUCING (a return series → one scalar), so tests read the single
output row of ``apply_expr``; ``assert_matches`` and the naive ``conditional_value_at_risk_reference`` oracle (the mean
below the type-7 quantile) are shared across the suite. It is degree-1 homogeneous in the returns (a mean of a quantile
slice), so it carries the scale-homogeneity and large-magnitude tiers, plus the metamorphic relation to
:func:`value_at_risk` (the shortfall never exceeds the quantile it averages beyond).

The ladder is the canonical one: contract (type / reduces-to-scalar / lazy-eager / ``.over`` per-group independence),
edge (validation / empty / single-row / null / NaN), correctness (vs the closed-form reference and a frozen golden
master), and properties (reference agreement incl. missing data, degree-1 scale-homogeneity, large-magnitude stability,
the shortfall-at-most-value-at-risk identity). Categories are split into classes; cross-cutting categories use markers.
"""

import math
from collections.abc import Sequence

import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st
from tests.metrics.oracles import conditional_value_at_risk_reference
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

from pomata.metrics import conditional_value_at_risk, value_at_risk

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- conditional_value_at_risk is windowless and REDUCING (M = 0); a case is just a return series. Facts:
#   1. shape   reducing: the output is one scalar (one row in ``select``; one per group under ``.over``)
#   2. domain  subnormal_safe_floats(bound): finite returns floored away from the subnormal range so the shortfall mean
#              and the rescaled comparisons stay well-conditioned; the missing variant mixes null / NaN
#   3. scale   degree-1 homogeneous (a mean of a quantile slice) -> scale-homogeneity + large-magnitude tiers
# CONFIDENCE varies over a realistic set in the fuzz. Repetitions N are the shared CI profile (tests/conftest.py).
# ----------------------------------------------------------------------------------------------------------------------
SERIES_MAX = 50
CONFIDENCE = 0.95
_CONFIDENCE = st.sampled_from([0.9, 0.95, 0.99])


@st.composite
def _cases[T](draw: st.DrawFn, returns: st.SearchStrategy[T], min_size: int = 1) -> list[T]:
    """A return series sized from the facts above."""
    return draw(st.lists(returns, min_size=min_size, max_size=SERIES_MAX))


def _abs_tol(values: Sequence[float | None]) -> float:
    """The magnitude-relative absolute tolerance for a shortfall mean on the returns' own scale."""
    return streaming_abs_tol(values)


class TestConditionalValueAtRiskContract:
    """
    Type, shape, and lazy/eager guarantees.
    """


class TestConditionalValueAtRiskEdge:
    """
    Validation, boundaries, and null / NaN handling.
    """

    def test_confidence_out_of_range_raises(self) -> None:
        """
        Verifies that a ``confidence`` outside the open interval ``(0, 1)`` raises ``ValueError``.
        """
        for invalid in (0.0, 1.0, -0.1, 1.5):
            with pytest.raises(ValueError, match="confidence must be in the open interval"):
                conditional_value_at_risk(pl.col(COLUMN_X), confidence=invalid)

    def test_single_row(self) -> None:
        """
        Verifies that a one-element series resolves to that element (it is the whole shortfall slice).
        """
        values = [-0.02]
        assert_matches(
            apply_expr(values, conditional_value_at_risk(pl.col(COLUMN_X), confidence=CONFIDENCE)),
            [conditional_value_at_risk_reference(values, CONFIDENCE)],
            abs_tol=_abs_tol(values),
        )

    def test_null_skipped(self) -> None:
        """
        Verifies that ``null`` returns are skipped (excluded from the shortfall mean), matching the reference.
        """
        values = [0.01, None, 0.02, 0.03, None]
        assert_matches(
            apply_expr(values, conditional_value_at_risk(pl.col(COLUMN_X), confidence=CONFIDENCE)),
            [conditional_value_at_risk_reference(values, CONFIDENCE)],
            abs_tol=_abs_tol(values),
        )

    def test_nan_poisons(self) -> None:
        """
        Verifies that a NaN return poisons the result to NaN.
        """
        values = [0.01, math.nan, -0.02, 0.03]
        assert_matches(
            apply_expr(values, conditional_value_at_risk(pl.col(COLUMN_X), confidence=CONFIDENCE)), [math.nan]
        )


class TestConditionalValueAtRiskCorrectness:
    """
    Against the naive reference oracle and frozen golden-master values.
    """

    def test_matches_reference(self) -> None:
        """
        Verifies agreement with the naive closed-form reference over a representative return series.
        """
        values = [0.012, -0.008, 0.02, -0.015, 0.005, 0.0, -0.02, 0.018, -0.03, 0.01]
        assert_matches(
            apply_expr(values, conditional_value_at_risk(pl.col(COLUMN_X), confidence=0.8)),
            [conditional_value_at_risk_reference(values, 0.8)],
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
            abs_tol=_abs_tol(values),
        )

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference: with ``k = (1 - 0.75) * 8 = 2``, the average of the two worst returns is -0.07.
        """
        values = [0.03, -0.05, 0.02, -0.08, 0.01, -0.06, 0.04, -0.02]
        assert_matches(
            apply_expr(values, conditional_value_at_risk(pl.col(COLUMN_X), confidence=0.75).round(4)), [-0.07]
        )

    def test_fractional_weight_golden(self) -> None:
        """
        Verifies the Rockafellar-Uryasev fractional boundary weight: with ``k = (1 - 0.7) * 5 = 1.5`` the worst return
        is averaged in full and the second-worst at weight ``0.5``, so the shortfall is
        ``(-0.10 + 0.5 * -0.06) / 1.5 = -0.0867`` -- not the worst alone, as a hard tail cutoff would give.
        """
        values = [-0.10, -0.06, 0.0, 0.05, 0.10]
        assert_matches(
            apply_expr(values, conditional_value_at_risk(pl.col(COLUMN_X), confidence=0.7).round(4)), [-0.0867]
        )


class TestConditionalValueAtRiskProperties:
    """
    Invariants that must hold for all inputs (property-based).
    """

    @given(case=_cases(subnormal_safe_floats(bound=1e3)), confidence=_CONFIDENCE)
    def test_matches_reference_for_any_input(self, case: list[float], confidence: float) -> None:
        """
        Verifies that, for any return series, the implementation matches the naive reference.
        """
        assert_matches(
            apply_expr(case, conditional_value_at_risk(pl.col(COLUMN_X), confidence=confidence)),
            [conditional_value_at_risk_reference(case, confidence)],
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=_abs_tol(case),
        )

    @given(case=_cases(missing_data_floats(min_magnitude=1e-3), min_size=0), confidence=_CONFIDENCE)
    def test_matches_reference_under_missing_data(self, case: list[float | None], confidence: float) -> None:
        """
        Verifies that, for inputs freely mixing null / NaN / finite, the implementation matches the naive reference.
        """
        assert_matches(
            apply_expr(case, conditional_value_at_risk(pl.col(COLUMN_X), confidence=confidence)),
            [conditional_value_at_risk_reference(case, confidence)],
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=_abs_tol(case),
        )

    @given(case=_cases(subnormal_safe_floats(bound=1e3)), exponent=st.sampled_from([-4, -3, -2, -1, 1, 2, 3, 4]))
    def test_scale_homogeneity(self, case: list[float], exponent: int) -> None:
        """
        Verifies that ``conditional_value_at_risk`` is homogeneous of degree 1: scaling every input value by a
        constant ``k`` scales the output by the same ``k`` -- ``conditional_value_at_risk(k * x) == k *
        conditional_value_at_risk(x)``. ``k`` is a power of two, so the rescale is exact and adds no floating-point
        error.
        """
        k = 2.0**exponent
        base = apply_expr(case, conditional_value_at_risk(pl.col(COLUMN_X), confidence=CONFIDENCE))
        scaled = apply_expr(
            [value * k for value in case], conditional_value_at_risk(pl.col(COLUMN_X), confidence=CONFIDENCE)
        )
        assert_scale_homogeneous(scaled, base, k=k, degree=1)

    @given(case=_cases(subnormal_safe_floats(bound=1e3)), scale=st.sampled_from([1e-6, 1e6, 1e9]))
    def test_matches_reference_at_large_magnitude(self, case: list[float], scale: float) -> None:
        """
        Verifies that at extreme magnitudes the implementation stays finite where the reference is and agrees.
        """
        scaled = [value * scale for value in case]
        assert_matches(
            apply_expr(scaled, conditional_value_at_risk(pl.col(COLUMN_X), confidence=CONFIDENCE)),
            [conditional_value_at_risk_reference(scaled, CONFIDENCE)],
            rel_tol=RELATIVE_TOLERANCE_SCALE,
            abs_tol=_abs_tol(scaled),
        )

    @given(case=_cases(subnormal_safe_floats(bound=1e3), min_size=1), confidence=_CONFIDENCE)
    def test_at_most_value_at_risk(self, case: list[float], confidence: float) -> None:
        """
        Verifies the metamorphic relation to :func:`value_at_risk`: the expected shortfall (a mean of returns at or
        below the quantile) never exceeds the quantile itself.
        """
        shortfall = apply_expr(case, conditional_value_at_risk(pl.col(COLUMN_X), confidence=confidence))[0]
        quantile = apply_expr(case, value_at_risk(pl.col(COLUMN_X), confidence=confidence))[0]
        assert shortfall is not None
        assert quantile is not None
        assert shortfall <= quantile + _abs_tol(case)
