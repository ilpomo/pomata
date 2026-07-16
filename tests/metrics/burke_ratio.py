"""Spec for ``pomata.metrics.burke_ratio`` — reducing, excess CAGR per unit of drawdown energy, scale-exempt."""

import math

import polars as pl
from tests.metrics.oracles import burke_ratio_reference
from tests.support.spec import ScaleExempt, Shape, Spec, SpecPin

from pomata.metrics import burke_ratio, cagr, drawdown

BURKE_RATIO = Spec(
    factory=burke_ratio,
    inputs=("equity_curve",),
    params={"periods_per_year": 252, "risk_free_rate": 0.0},
    shape=Shape.REDUCING,
    raises=(
        ({"periods_per_year": 0}, r"periods_per_year must be >= 1"),
        ({"risk_free_rate": math.nan}, r"risk_free_rate must be a finite number"),
        ({"risk_free_rate": math.inf}, r"risk_free_rate must be a finite number"),
        ({"risk_free_rate": -math.inf}, r"risk_free_rate must be a finite number"),
    ),
    oracle=burke_ratio_reference,
    # A normalized growth factor (CAGR) over a scale-invariant drawdown energy — neither homogeneous nor invariant
    #
    scale=ScaleExempt(
        reason="a normalized growth factor (CAGR) over a scale-invariant drawdown energy — neither homogeneous "
        "nor invariant"
    ),
    golden_params={"periods_per_year": 1},
    golden_input={"equity_curve": (1.1, 1.05, 1.2, 1.15, 1.3, 1.25, 1.4)},
    golden_output=(0.6776,),
    component_expr=lambda: (
        (cagr(pl.col("equity_curve"), periods_per_year=252) - 0.0)
        / (drawdown(pl.col("equity_curve")) ** 2).sum().sqrt()
    ),
    pins=(
        SpecPin(
            label="single_row",
            inputs={"equity_curve": (1.0,)},
            expected=(math.nan,),
            reason="one observation has zero growth and zero drawdown energy, so the ratio is 0/0, i.e. NaN ",
        ),
        SpecPin(
            label="no_drawdown_is_inf",
            inputs={"equity_curve": (1.0, 1.1, 1.21)},
            expected=(math.inf,),
            reason="a monotonically rising curve has zero drawdown energy with positive growth, so the ratio is +inf ",
            params_override={"periods_per_year": 1},
        ),
        SpecPin(
            label="no_drawdown_zero_growth_is_nan",
            inputs={"equity_curve": (1.0, 1.0, 1.0)},
            expected=(math.nan,),
            reason="a flat multi-row curve has zero drawdown energy and zero excess growth, so the ratio is a 0/0, "
            "i.e. NaN — the degenerate-denominator NaN beside the +inf pin",
        ),
    ),
)
