"""Spec for ``pomata.metrics.calmar_ratio`` — reducing, CAGR per unit of maximum drawdown, scale-exempt."""

import math

import polars as pl
from tests.metrics.oracles import calmar_ratio_reference
from tests_new.support.spec import ScaleExempt, Shape, Spec, SpecPin

from pomata.metrics import cagr, calmar_ratio, max_drawdown

CALMAR_RATIO = Spec(
    factory=calmar_ratio,
    inputs=("equity_curve",),
    params={"periods_per_year": 252},
    shape=Shape.REDUCING,
    raises=(({"periods_per_year": 0}, r"periods_per_year must be >= 1"),),
    oracle=calmar_ratio_reference,
    # A normalized growth-factor curve run through CAGR over a scale-invariant drawdown magnitude — neither
    # homogeneous nor invariant (tests/metrics/test_calmar_ratio.py module docstring).
    scale=ScaleExempt(
        reason="a normalized growth-factor curve run through CAGR over a scale-invariant drawdown magnitude — "
        "neither homogeneous nor invariant"
    ),
    golden_input={"equity_curve": (1.1, 1.05, 1.2, 1.15, 1.3, 1.25, 1.4)},
    golden_output=(1.0833,),
    golden_params={"periods_per_year": 1},
    component_expr=lambda: (
        cagr(pl.col("equity_curve"), periods_per_year=252) / max_drawdown(pl.col("equity_curve")).abs()
    ),
    pins=(
        SpecPin(
            label="single_row",
            inputs={"equity_curve": (1.0,)},
            expected=(math.nan,),
            reason="a one-element series has zero growth and zero drawdown, so the ratio is 0/0, i.e. NaN "
            "(tests/metrics/test_calmar_ratio.py::TestCalmarRatioEdge::test_single_row)",
        ),
        SpecPin(
            label="no_drawdown_is_inf",
            inputs={"equity_curve": (1.0, 1.1, 1.21)},
            expected=(math.inf,),
            reason="a monotonically rising curve has zero maximum drawdown with positive growth, so the ratio is +inf "
            "(tests/metrics/test_calmar_ratio.py::TestCalmarRatioEdge::test_no_drawdown_is_inf)",
            params_override={"periods_per_year": 1},
        ),
    ),
)
