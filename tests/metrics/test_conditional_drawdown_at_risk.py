"""
Tests for ``pomata.metrics.conditional_drawdown_at_risk`` — the mean of the worst drawdowns beyond a confidence level.

``conditional_drawdown_at_risk`` is single-input and REDUCING (an equity series → one scalar), so tests read the single
output row of ``apply_expr``; ``assert_matches`` and the naive ``conditional_drawdown_at_risk_reference`` oracle (the
mean below the drawdown quantile) are shared across the suite. It is scale-invariant under a positive rescale, so it
carries a scale-invariance tier.

The ladder is the canonical one: contract (type / reduces-to-scalar / lazy-eager / ``.over`` per-group independence),
edge (validation / empty / single-row / no-drawdown / null / NaN), correctness (vs the closed-form reference and a
frozen golden master), and properties (reference agreement incl. missing data, scale invariance). Categories are split
into classes; cross-cutting categories use markers.
"""

import math

import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st
from tests.metrics.oracles import conditional_drawdown_at_risk_reference
from tests.support import (
    ABSOLUTE_TOLERANCE_REFERENCE,
    COLUMN_X,
    RELATIVE_TOLERANCE_PROPERTY,
    RELATIVE_TOLERANCE_REFERENCE,
    RELATIVE_TOLERANCE_SCALE,
    apply_expr,
    assert_matches,
)

from pomata.metrics import conditional_drawdown_at_risk

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- conditional_drawdown_at_risk is windowless and REDUCING (M = 0); a case is a positive equity series.
#   1. shape   reducing: the output is one scalar (one row in ``select``; one per group under ``.over``)
#   2. domain  positive equities (a growth factor is > 0); the missing variant mixes null / NaN
#   3. scale   invariant under a positive rescale (the drawdown ratio cancels) -> scale-invariance tier
# CONFIDENCE varies over a realistic set in the fuzz. Repetitions N are the shared CI profile (tests/conftest.py).
# ----------------------------------------------------------------------------------------------------------------------
SERIES_MAX = 50
CONFIDENCE = 0.95
_CONFIDENCE = st.sampled_from([0.9, 0.95, 0.99])
_EQUITY = st.floats(min_value=1e-2, max_value=1e4, allow_nan=False, allow_infinity=False)
_EQUITY_MISSING = st.one_of(st.none(), st.just(math.nan), _EQUITY)


@st.composite
def _cases[T](draw: st.DrawFn, equities: st.SearchStrategy[T], min_size: int = 1) -> list[T]:
    """A positive equity series sized from the facts above."""
    return draw(st.lists(equities, min_size=min_size, max_size=SERIES_MAX))


class TestConditionalDrawdownAtRiskContract:
    """
    Type, shape, and lazy/eager guarantees.
    """


class TestConditionalDrawdownAtRiskEdge:
    """
    Validation, boundaries, and null / NaN handling.
    """

    def test_confidence_out_of_range_raises(self) -> None:
        """
        Verifies that a ``confidence`` outside the open interval ``(0, 1)`` raises ``ValueError``.
        """
        for invalid in (0.0, 1.0, -0.1, 1.5):
            with pytest.raises(ValueError, match="confidence must be in the open interval"):
                conditional_drawdown_at_risk(pl.col(COLUMN_X), confidence=invalid)

    def test_single_row_is_zero(self) -> None:
        """
        Verifies that a one-element series is at its own peak, so the conditional drawdown at risk is ``0``.
        """
        assert_matches(apply_expr([1.0], conditional_drawdown_at_risk(pl.col(COLUMN_X), confidence=CONFIDENCE)), [0.0])

    def test_no_drawdown_is_zero(self) -> None:
        """
        Verifies that a monotonically rising curve has an all-zero drawdown series, so the measure is ``0``.
        """
        assert_matches(
            apply_expr([1.0, 1.1, 1.21], conditional_drawdown_at_risk(pl.col(COLUMN_X), confidence=CONFIDENCE)), [0.0]
        )

    def test_null_skipped(self) -> None:
        """
        Verifies that null equities are skipped, matching the reference.
        """
        values = [1.0, None, 0.9, 0.95, None, 1.1, 1.05]
        assert_matches(
            apply_expr(values, conditional_drawdown_at_risk(pl.col(COLUMN_X), confidence=CONFIDENCE)),
            [conditional_drawdown_at_risk_reference(values, CONFIDENCE)],
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
        )

    def test_nan_poisons(self) -> None:
        """
        Verifies that a NaN equity poisons the result to NaN.
        """
        values = [1.1, math.nan, 1.2, 0.9]
        assert_matches(
            apply_expr(values, conditional_drawdown_at_risk(pl.col(COLUMN_X), confidence=CONFIDENCE)), [math.nan]
        )


class TestConditionalDrawdownAtRiskCorrectness:
    """
    Against the naive reference oracle and frozen golden-master values.
    """

    def test_matches_reference(self) -> None:
        """
        Verifies agreement with the naive closed-form reference over a representative equity curve.
        """
        values = [1.0, 1.1, 1.05, 1.2, 1.15, 1.3, 1.25, 1.1]
        assert_matches(
            apply_expr(values, conditional_drawdown_at_risk(pl.col(COLUMN_X), confidence=0.8)),
            [conditional_drawdown_at_risk_reference(values, 0.8)],
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference: the mean of the worst 5% of drawdowns of the curve is -0.0455.
        """
        values = [1.1, 1.05, 1.2, 1.15, 1.3, 1.25, 1.4]
        assert_matches(
            apply_expr(values, conditional_drawdown_at_risk(pl.col(COLUMN_X), confidence=0.95).round(4)), [-0.0455]
        )

    def test_fractional_weight_golden(self) -> None:
        """
        Verifies the Rockafellar-Uryasev fractional boundary weight: the curve's drawdowns are
        ``{-0.3, -0.2, -0.1, 0, 0, 0}`` and with ``k = (1 - 0.75) * 6 = 1.5`` the worst is averaged in full and the
        second-worst at weight ``0.5``, so ``(-0.3 + 0.5 * -0.2) / 1.5 = -0.2667``.
        """
        values = [1.0, 0.8, 1.0, 0.9, 0.7, 1.0]
        assert_matches(
            apply_expr(values, conditional_drawdown_at_risk(pl.col(COLUMN_X), confidence=0.75).round(4)), [-0.2667]
        )


class TestConditionalDrawdownAtRiskProperties:
    """
    Invariants that must hold for all inputs (property-based).
    """

    @given(case=_cases(_EQUITY), confidence=_CONFIDENCE)
    def test_matches_reference_for_any_input(self, case: list[float], confidence: float) -> None:
        """
        Verifies that, for any positive equity series, the implementation matches the naive reference.
        """
        assert_matches(
            apply_expr(case, conditional_drawdown_at_risk(pl.col(COLUMN_X), confidence=confidence)),
            [conditional_drawdown_at_risk_reference(case, confidence)],
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    @given(case=_cases(_EQUITY_MISSING, min_size=0), confidence=_CONFIDENCE)
    def test_matches_reference_under_missing_data(self, case: list[float | None], confidence: float) -> None:
        """
        Verifies that, for inputs freely mixing null / NaN / finite, the implementation matches the naive reference.
        """
        assert_matches(
            apply_expr(case, conditional_drawdown_at_risk(pl.col(COLUMN_X), confidence=confidence)),
            [conditional_drawdown_at_risk_reference(case, confidence)],
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    @given(case=_cases(_EQUITY), exponent=st.sampled_from([-4, -3, -2, -1, 1, 2, 3, 4]))
    def test_scale_invariance(self, case: list[float], exponent: int) -> None:
        """
        Verifies that ``conditional_drawdown_at_risk`` is scale-invariant: scaling every input value by a constant
        ``k`` leaves the output unchanged -- ``conditional_drawdown_at_risk(k * x) ==
        conditional_drawdown_at_risk(x)``. ``k`` is a power of two, so the rescale is exact and adds no
        floating-point error.
        """
        k = 2.0**exponent
        base = apply_expr(case, conditional_drawdown_at_risk(pl.col(COLUMN_X), confidence=CONFIDENCE))
        scaled = apply_expr(
            [value * k for value in case], conditional_drawdown_at_risk(pl.col(COLUMN_X), confidence=CONFIDENCE)
        )
        assert_matches(scaled, base, rel_tol=RELATIVE_TOLERANCE_SCALE, abs_tol=ABSOLUTE_TOLERANCE_REFERENCE)
