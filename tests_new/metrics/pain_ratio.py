"""Spec for ``pomata.metrics.pain_ratio`` — reducing, excess CAGR per unit of pain index, scale-exempt."""

import math

import polars as pl
from tests_new.metrics.oracles import pain_ratio_reference
from tests_new.support.spec import ScaleExempt, Shape, Spec, SpecPin

from pomata.metrics import cagr, pain_index, pain_ratio

PAIN_RATIO = Spec(
    factory=pain_ratio,
    inputs=("equity_curve",),
    params={"periods_per_year": 252, "risk_free_rate": 0.0},
    shape=Shape.REDUCING,
    raises=(
        ({"periods_per_year": 0}, r"periods_per_year must be >= 1"),
        ({"risk_free_rate": math.nan}, r"risk_free_rate must be a finite number"),
        ({"risk_free_rate": math.inf}, r"risk_free_rate must be a finite number"),
        ({"risk_free_rate": -math.inf}, r"risk_free_rate must be a finite number"),
    ),
    oracle=pain_ratio_reference,
    # A normalized growth-factor curve (excess CAGR over the average-drawdown pain index) — neither homogeneous nor
    # invariant (tests/metrics/test_pain_ratio.py module docstring).
    scale=ScaleExempt(
        reason="a normalized growth-factor curve (excess CAGR over the average-drawdown pain index) — neither "
        "homogeneous nor invariant"
    ),
    golden_input={"equity_curve": (1.1, 1.05, 1.2, 1.15, 1.3, 1.25, 1.4)},
    golden_output=(2.7447,),
    golden_params={"periods_per_year": 1},
    component_expr=lambda: (
        (cagr(pl.col("equity_curve"), periods_per_year=252) - 0.0) / pain_index(pl.col("equity_curve"))
    ),
    pins=(
        SpecPin(
            label="single_row",
            inputs={"equity_curve": (1.0,)},
            expected=(math.nan,),
            reason="one observation has zero excess growth and a zero pain index, so the ratio is the 0/0 NaN branch "
            "(tests/metrics/test_pain_ratio.py::test_single_row)",
        ),
        SpecPin(
            label="no_drawdown_is_inf",
            inputs={"equity_curve": (1.0, 1.1, 1.21)},
            expected=(math.inf,),
            reason="a monotonically rising curve has a zero pain index with positive excess growth, so the ratio "
            "is +inf (tests/metrics/test_pain_ratio.py::test_no_drawdown_is_inf)",
            params_override={"periods_per_year": 1},
        ),
    ),
)
