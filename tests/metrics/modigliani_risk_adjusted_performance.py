"""Spec for ``pomata.metrics.modigliani_risk_adjusted_performance`` — reducing, M-squared, degree-1 at rf=0."""

import math

import polars as pl
from tests.metrics.oracles import modigliani_risk_adjusted_performance_reference
from tests.support.spec import ScaleAxis, Shape, Spec, SpecPin

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
    # Degree-1 homogeneous in a joint returns/benchmark rescale at risk_free_rate=0 (the spec's params): M-squared is
    # then sharpe(returns) * volatility(benchmark), a degree-0 ratio times a degree-1 dispersion — the same
    # default-scoped axis convention as downside_deviation's "at threshold=0". A non-zero rate breaks it.
    scale=(ScaleAxis(roles=("returns", "benchmark"), degree=1),),
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
            reason="a single complete pair leaves the embedded Sharpe and benchmark volatility undefined, so null",
        ),
        SpecPin(
            label="constant_portfolio_is_inf",
            inputs={"returns": (0.01, 0.01, 0.01), "benchmark": (0.02, -0.01, 0.03)},
            expected=(math.inf,),
            reason="a constant portfolio has zero dispersion, so the embedded Sharpe is +inf, which propagates to "
            "+inf — the exact-zero core of the near-constant regime; both onsets (zero portfolio dispersion, an "
            "unbounded embedded Sharpe) are unreachable by the fuzz, so no conditioning filter is declared",
        ),
    ),
)
