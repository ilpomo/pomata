"""
The metrics residue the declarative spec cannot carry: the large-magnitude reference tier of the degree-1-homogeneous
dispersion / quantile metrics, and the cross-function metamorphic relations between two public metrics, ported as
ordinary property tests (the mirror of ``tests/pnl/test_bespoke.py``).

A :class:`~tests.support.spec.SpecPin` is a fixed crafted case and a :class:`~tests.support.spec.ScaleAxis` is
an exact homogeneity ratio at one ``k``; neither expresses a Hypothesis-quantified claim, and a relation *between two
public metrics* (an inequality, a self-identity, a bounded range) has no declarative slot at all. These claims survive
from the old metrics suite and live here as ordinary ``@given`` tests, named for the old-suite tests they port:

- ``test_matches_reference_at_large_magnitude`` — every degree-1-homogeneous dispersion / quantile metric keeps oracle
  agreement when its returns are pushed to extreme magnitude (a numeric-stability tier, distinct from the exact-ratio
  ``scale`` axis), parametrized over the specs that carry it.
- the cross-function metamorphic relations — the two tail-risk inequalities, the two self-capture identities, the
  cagr/total-return identity, and the two unit-interval bounds — each a relation between two public metrics or a
  bounded range, not a recomposition of one factory from others (which is what ``component_expr`` is for).
"""

import math
from collections.abc import Sequence

import polars as pl
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from tests.metrics.conditional_value_at_risk import CONDITIONAL_VALUE_AT_RISK
from tests.metrics.downside_deviation import DOWNSIDE_DEVIATION
from tests.metrics.value_at_risk import VALUE_AT_RISK
from tests.metrics.value_at_risk_modified import VALUE_AT_RISK_MODIFIED
from tests.metrics.value_at_risk_parametric import VALUE_AT_RISK_PARAMETRIC
from tests.metrics.volatility import VOLATILITY
from tests.support import (
    ABSOLUTE_TOLERANCE_REFERENCE,
    BENCHMARK,
    COLUMN_X,
    RELATIVE_TOLERANCE_PROPERTY,
    RELATIVE_TOLERANCE_REFERENCE,
    RELATIVE_TOLERANCE_SCALE,
    RETURNS,
    apply_expr,
    assert_matches,
    materialize,
    split_pairs,
    standardized_moment_floats,
    streaming_abs_tol,
    subnormal_safe_floats,
    well_spread,
)
from tests.support.spec import Spec, actual_lanes, fuzz_frames, reference_lanes, spec_id

from pomata.metrics import (
    cagr,
    capture_downside_ratio,
    capture_upside_ratio,
    conditional_drawdown_at_risk,
    conditional_value_at_risk,
    max_drawdown,
    probabilistic_sharpe_ratio,
    stability,
    total_return,
    value_at_risk,
)

# ======================================================================================================================
# Large magnitude — the numeric-stability tier of every degree-1-homogeneous dispersion / quantile metric
# ======================================================================================================================

# The specs whose old suite carried a ``test_matches_reference_at_large_magnitude`` tier: every returns-homogeneous
# dispersion or quantile metric (so the value genuinely rides the input scale). The scale-invariant siblings (skewness,
# kurtosis, stability, the ratios) had no such tier and are excluded.
LARGE_MAGNITUDE_SPECS: tuple[Spec, ...] = (
    CONDITIONAL_VALUE_AT_RISK,
    DOWNSIDE_DEVIATION,
    VALUE_AT_RISK,
    VALUE_AT_RISK_MODIFIED,
    VALUE_AT_RISK_PARAMETRIC,
    VOLATILITY,
)

_LARGE_MAGNITUDE_SCALES = (1e-6, 1e6, 1e9)


@pytest.mark.parametrize("spec", LARGE_MAGNITUDE_SPECS, ids=spec_id)
@given(data=st.data(), scale=st.sampled_from(_LARGE_MAGNITUDE_SCALES))
def test_matches_reference_at_large_magnitude(spec: Spec, data: st.DataObject, scale: float) -> None:
    """Verifies that at extreme magnitudes the implementation stays finite where the reference is and agrees."""
    frame = data.draw(fuzz_frames(spec, missing=False))
    if spec.conditioning is not None:
        assume(spec.conditioning(frame))
    scaled = frame.select([pl.col(role) * scale for role in spec.inputs])
    # The old suite sized the absolute band to the scaled input, annualized where the metric is (downside_deviation /
    # volatility carry ``periods_per_year``); the VaR family and cvar leave ``periods`` at one.
    periods = int(spec.params.get("periods_per_year", 1))
    abs_tol = streaming_abs_tol(scaled[spec.inputs[0]].to_list(), periods=periods)
    expected = reference_lanes(spec, scaled)
    actual = actual_lanes(spec, scaled)
    assert sorted(actual) == sorted(expected)
    for name, values in expected.items():
        assert_matches(actual[name], values, rel_tol=RELATIVE_TOLERANCE_SCALE, abs_tol=abs_tol)


# ======================================================================================================================
# Cross-function metamorphic relations — each a relation between two public metrics, or a bounded range
# ======================================================================================================================

_SERIES_MAX = 50
_CONFIDENCE = st.sampled_from([0.9, 0.95, 0.99])
_PERIODS = st.sampled_from([1, 4, 12, 52, 252])
# A positive equity path (a growth factor is > 0) for the drawdown-family relation.
_EQUITY = st.floats(min_value=1e-2, max_value=1e4, allow_nan=False, allow_infinity=False)
# Modest positive growth factors so the cagr annualizing power stays finite.
_GROWTH = st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False)
# Sign-varied returns bounded away from zero, so the geometric power of the capture ratios stays well-conditioned.
_VALUE = st.one_of(
    st.floats(min_value=0.01, max_value=0.5, allow_nan=False, allow_infinity=False),
    st.floats(min_value=-0.5, max_value=-0.01, allow_nan=False, allow_infinity=False),
)
_PAIR = st.tuples(_VALUE, _VALUE)
# Strictly positive returns so the cumulative log path of the stability regression is monotone and well-conditioned.
_POSITIVE_RETURNS = st.floats(min_value=1e-3, max_value=10.0, allow_nan=False, allow_infinity=False)


def _has_substantial_gain(returns: Sequence[float | None]) -> bool:
    """Whether any return is clearly positive, so ``1 + r`` exceeds one and the geometric growth is non-degenerate."""
    return any(value is not None and not math.isnan(value) and value > 1e-3 for value in returns)


def _has_substantial_loss(returns: Sequence[float | None]) -> bool:
    """Whether any return is clearly negative, so ``1 + r`` is below one and the geometric growth is non-degenerate."""
    return any(value is not None and not math.isnan(value) and value < -1e-3 for value in returns)


@given(equity=st.lists(_EQUITY, min_size=1, max_size=_SERIES_MAX), confidence=_CONFIDENCE)
def test_conditional_drawdown_at_least_max_drawdown(equity: list[float], confidence: float) -> None:
    """Verifies the tail mean of the worst drawdowns never falls below the single worst drawdown."""
    # test_conditional_drawdown_at_risk.py::TestConditionalDrawdownAtRiskProperties::test_at_least_max_drawdown
    tail_mean = apply_expr(equity, conditional_drawdown_at_risk(pl.col(COLUMN_X), confidence=confidence))[0]
    worst = apply_expr(equity, max_drawdown(pl.col(COLUMN_X)))[0]
    assert tail_mean is not None
    assert worst is not None
    assert tail_mean >= worst - ABSOLUTE_TOLERANCE_REFERENCE


@given(returns=st.lists(subnormal_safe_floats(bound=1e3), min_size=1, max_size=_SERIES_MAX), confidence=_CONFIDENCE)
def test_conditional_value_at_most_value_at_risk(returns: list[float], confidence: float) -> None:
    """Verifies the expected shortfall never exceeds the value-at-risk quantile it averages beyond."""
    # test_conditional_value_at_risk.py::TestConditionalValueAtRiskProperties::test_at_most_value_at_risk
    shortfall = apply_expr(returns, conditional_value_at_risk(pl.col(COLUMN_X), confidence=confidence))[0]
    quantile = apply_expr(returns, value_at_risk(pl.col(COLUMN_X), confidence=confidence))[0]
    assert shortfall is not None
    assert quantile is not None
    assert shortfall <= quantile + streaming_abs_tol(returns)


@given(case=st.lists(_PAIR, min_size=1, max_size=_SERIES_MAX), periods=_PERIODS)
def test_capture_upside_self_capture_is_one(case: list[tuple[float, float]], periods: int) -> None:
    """Verifies a portfolio identical to its benchmark captures exactly one of its upside."""
    # test_capture_upside_ratio.py::TestCaptureUpsideRatioProperties::test_self_capture_is_one
    returns, _ = split_pairs(case)
    assume(_has_substantial_gain(returns))
    assert_matches(
        materialize(
            {RETURNS: returns, BENCHMARK: returns},
            capture_upside_ratio(pl.col(RETURNS), pl.col(BENCHMARK), periods_per_year=periods),
        ),
        [1.0],
        rel_tol=RELATIVE_TOLERANCE_REFERENCE,
    )


@given(case=st.lists(_PAIR, min_size=1, max_size=_SERIES_MAX), periods=_PERIODS)
def test_capture_downside_self_capture_is_one(case: list[tuple[float, float]], periods: int) -> None:
    """Verifies a portfolio identical to its benchmark captures exactly one of its downside."""
    # test_capture_downside_ratio.py::TestCaptureDownsideRatioProperties::test_self_capture_is_one
    returns, _ = split_pairs(case)
    assume(_has_substantial_loss(returns))
    assert_matches(
        materialize(
            {RETURNS: returns, BENCHMARK: returns},
            capture_downside_ratio(pl.col(RETURNS), pl.col(BENCHMARK), periods_per_year=periods),
        ),
        [1.0],
        rel_tol=RELATIVE_TOLERANCE_REFERENCE,
    )


@given(growth=st.lists(_GROWTH, min_size=1, max_size=_SERIES_MAX))
def test_cagr_equals_total_return_over_one_year(growth: list[float]) -> None:
    """Verifies annualizing over exactly one year (``periods_per_year == N``) makes the cagr equal the total return."""
    # test_cagr.py::TestCagrProperties::test_cagr_equals_total_return_over_one_year
    cagr_value = apply_expr(growth, cagr(pl.col(COLUMN_X), periods_per_year=len(growth)))[0]
    total = apply_expr(growth, total_return(pl.col(COLUMN_X)))[0]
    assert cagr_value is not None
    assert total is not None
    assert math.isclose(cagr_value, total, rel_tol=RELATIVE_TOLERANCE_PROPERTY, abs_tol=ABSOLUTE_TOLERANCE_REFERENCE)


@given(returns=st.lists(standardized_moment_floats(bound=1e3), min_size=2, max_size=_SERIES_MAX))
def test_probabilistic_sharpe_ratio_within_unit_interval(returns: list[float]) -> None:
    """Verifies a defined probabilistic Sharpe ratio is a probability in ``[0, 1]``."""
    # test_probabilistic_sharpe_ratio.py::TestProbabilisticSharpeRatioProperties::test_within_unit_interval
    assume(well_spread(returns))
    result = apply_expr(returns, probabilistic_sharpe_ratio(pl.col(COLUMN_X), periods_per_year=252))[0]
    if result is not None and not math.isnan(result):
        assert -ABSOLUTE_TOLERANCE_REFERENCE <= result <= 1.0 + ABSOLUTE_TOLERANCE_REFERENCE


@given(returns=st.lists(_POSITIVE_RETURNS, min_size=2, max_size=_SERIES_MAX))
def test_stability_within_unit_interval(returns: list[float]) -> None:
    """Verifies a defined stability is a coefficient of determination in ``[0, 1]``."""
    # test_stability.py::TestStabilityProperties::test_within_unit_interval
    result = apply_expr(returns, stability(pl.col(COLUMN_X)))[0]
    if result is not None and not math.isnan(result):
        assert -ABSOLUTE_TOLERANCE_REFERENCE <= result <= 1.0 + ABSOLUTE_TOLERANCE_REFERENCE
