"""Spec for ``pomata.metrics.modigliani_risk_adjusted_performance`` — reducing, M-squared, scale-exempt."""

import math

import polars as pl
from tests.metrics.oracles import modigliani_risk_adjusted_performance_reference
from tests.support.spec import ScaleExempt, Shape, Spec, SpecPin

from pomata.metrics import modigliani_risk_adjusted_performance, sharpe_ratio, volatility


def _modigliani_component() -> pl.Expr:
    """M-squared recomposed from the public ``sharpe_ratio`` and ``volatility`` factories at the default params."""
    return 0.0 + sharpe_ratio(pl.col("returns"), periods_per_year=252, risk_free_rate=0.0) * volatility(
        pl.col("benchmark"), periods_per_year=252
    )


MODIGLIANI_RISK_ADJUSTED_PERFORMANCE = Spec(
    factory=modigliani_risk_adjusted_performance,
    inputs=("returns", "benchmark"),
    params={"periods_per_year": 252, "risk_free_rate": 0.0},
    shape=Shape.REDUCING,
    raises=(
        ({"periods_per_year": 0}, r"periods_per_year must be >= 1"),
        ({"risk_free_rate": math.nan}, r"risk_free_rate must be a finite number"),
        ({"risk_free_rate": math.inf}, r"risk_free_rate must be a finite number"),
        ({"risk_free_rate": -math.inf}, r"risk_free_rate must be a finite number"),
    ),
    oracle=modigliani_risk_adjusted_performance_reference,
    # An annualized return rescaled to benchmark risk — neither scale-invariant nor homogeneous
    # (tests/metrics/test_modigliani_risk_adjusted_performance.py module docstring).
    scale=ScaleExempt(
        reason="an annualized return rescaled to benchmark risk — neither scale-invariant nor homogeneous"
    ),
    golden_input={
        "returns": (0.012, -0.008, 0.02, -0.015, 0.005, 0.0, -0.02, 0.018),
        "benchmark": (0.01, -0.006, 0.018, -0.012, 0.004, 0.002, -0.018, 0.015),
    },
    golden_output=(0.3274,),
    golden_params={"risk_free_rate": 0.02},
    component_expr=_modigliani_component,
    pins=(
        SpecPin(
            label="single_pair",
            inputs={"returns": (0.05,), "benchmark": (0.04,)},
            expected=(None,),
            reason="a single complete pair leaves the embedded Sharpe and benchmark volatility undefined, so null "
            "(tests/metrics/test_modigliani_risk_adjusted_performance.py::test_single_pair)",
        ),
        SpecPin(
            label="constant_portfolio_is_inf",
            inputs={"returns": (0.01, 0.01, 0.01), "benchmark": (0.02, -0.01, 0.03)},
            expected=(math.inf,),
            reason="a constant portfolio has zero dispersion, so the embedded Sharpe is +inf, which propagates to "
            "+inf — the exact-zero core of the regimes the old suite's two-clause filter (well-spread portfolio, "
            "bounded embedded Sharpe) excluded; both onsets are unreachable by the fuzz, so no filter is declared "
            "(tests/metrics/test_modigliani_risk_adjusted_performance.py::test_constant_portfolio_is_inf)",
        ),
    ),
)
