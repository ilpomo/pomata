"""Spec for ``pomata.metrics.sterling_ratio`` — reducing, excess CAGR per cushioned average drawdown, scale-exempt."""

import math

import polars as pl
from tests.metrics.oracles import sterling_ratio_reference
from tests.support.spec import ScaleExempt, Shape, Spec, SpecPin

from pomata.metrics import cagr, pain_index, sterling_ratio

STERLING_RATIO = Spec(
    factory=sterling_ratio,
    inputs=("equity_curve",),
    params={"periods_per_year": 252, "risk_free_rate": 0.0, "excess": 0.10},
    shape=Shape.REDUCING,
    raises=(
        ({"periods_per_year": 0}, r"periods_per_year must be >= 1"),
        ({"risk_free_rate": math.nan}, r"risk_free_rate must be a finite number"),
        ({"risk_free_rate": math.inf}, r"risk_free_rate must be a finite number"),
        ({"risk_free_rate": -math.inf}, r"risk_free_rate must be a finite number"),
        ({"excess": math.nan}, r"excess must be a finite number"),
        ({"excess": math.inf}, r"excess must be a finite number"),
        ({"excess": -math.inf}, r"excess must be a finite number"),
    ),
    oracle=sterling_ratio_reference,
    # A normalized growth factor over a scale-invariant average drawdown — neither homogeneous nor invariant
    #
    scale=ScaleExempt(
        reason="a normalized growth factor over a scale-invariant average drawdown — neither homogeneous nor invariant"
    ),
    golden_input={"equity_curve": (1.1, 1.05, 1.2, 1.15, 1.3, 1.25, 1.4)},
    golden_output=(0.4175,),
    golden_params={"periods_per_year": 1},
    component_expr=lambda: (
        (cagr(pl.col("equity_curve"), periods_per_year=252) - 0.0) / (pain_index(pl.col("equity_curve")) + 0.10)
    ),
    pins=(
        SpecPin(
            label="no_drawdown_is_zero",
            inputs={"equity_curve": (1.0,)},
            expected=(0.0,),
            reason="a flat single-period growth has zero drawdown and zero excess growth, so the ratio is exactly 0 "
            "(the excess cushion keeps the denominator finite) "
            "",
        ),
    ),
)
