"""Declaration for ``pomata.metrics.modigliani_risk_adjusted_performance`` — reducing, M-squared, degree-1 at rf=0."""

import math

import polars as pl

from pomata.metrics import modigliani_risk_adjusted_performance, sharpe_ratio, volatility
from tests.metrics.enums import Annualization, BehaviorNan, BehaviorNull, Degenerate
from tests.metrics.harness import suite_metrics
from tests.metrics.oracles import reference_modigliani_risk_adjusted_performance
from tests.support.declaration import Golden, Pin, ScaleAxis


def _modigliani_component() -> pl.Expr:
    """M-squared recomposed from the public ``sharpe_ratio`` and ``volatility`` factories at the default params."""
    return 0.0 + sharpe_ratio(pl.col("returns"), periods_per_year=252, risk_free_rate=0.0) * volatility(
        pl.col("benchmark"), periods_per_year=252
    )


MODIGLIANI_RISK_ADJUSTED_PERFORMANCE = suite_metrics(
    factory=modigliani_risk_adjusted_performance,
    inputs=("returns", "benchmark"),
    params={"periods_per_year": 252, "risk_free_rate": 0.0},
    null=BehaviorNull.SKIPPED,
    nan=BehaviorNan.POISONS,
    annualization=Annualization.LINEAR,
    degenerate=Degenerate.RATIO_SIGNED_INF_OR_NAN,
    oracle=reference_modigliani_risk_adjusted_performance,
    recomposition=_modigliani_component,
    scaling=(ScaleAxis(roles=("returns", "benchmark"), degree=1),),
    raises=(
        ({"periods_per_year": 0}, r"periods_per_year must be >= 1"),
        ({"risk_free_rate": math.nan}, r"risk_free_rate must be a finite number"),
        ({"risk_free_rate": -2.0}, r"risk_free_rate must be >= -1"),
        ({"risk_free_rate": math.inf}, r"risk_free_rate must be a finite number"),
        ({"risk_free_rate": -math.inf}, r"risk_free_rate must be a finite number"),
    ),
    golden=Golden(
        inputs={
            "returns": (0.012, -0.008, 0.02, -0.015, 0.005, 0.0, -0.02, 0.018),
            "benchmark": (0.01, -0.006, 0.018, -0.012, 0.004, 0.002, -0.018, 0.015),
        },
        output=(0.3274,),
        params={"risk_free_rate": 0.02},
    ),
    pins=(
        Pin(
            label="single_pair",
            inputs={"returns": (0.05,), "benchmark": (0.04,)},
            expected=(None,),
            reason="a single complete pair leaves the embedded Sharpe and benchmark volatility undefined, so null",
        ),
        Pin(
            label="constant_portfolio_is_inf",
            inputs={"returns": (0.01, 0.01, 0.01), "benchmark": (0.02, -0.01, 0.03)},
            expected=(math.inf,),
            reason="a constant portfolio has zero dispersion by the exact pin, so the embedded Sharpe is "
            "+inf, which propagates to +inf; no conditioning filter is declared: the composed oracle "
            "mirrors the same exact min == max constancy detection on both legs, so the two sides "
            "agree in kind on every constant series",
        ),
    ),
)
