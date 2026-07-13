"""Spec for ``pomata.metrics.ulcer_performance_ratio`` — reducing, excess CAGR per unit of ulcer index, scale-exempt."""

import math

import polars as pl
from tests_new.metrics.oracles import ulcer_performance_ratio_reference
from tests_new.support.spec import ScaleExempt, Shape, Spec, SpecPin

from pomata.metrics import cagr, ulcer_index, ulcer_performance_ratio

ULCER_PERFORMANCE_RATIO = Spec(
    factory=ulcer_performance_ratio,
    inputs=("equity_curve",),
    params={"periods_per_year": 252, "risk_free_rate": 0.0},
    shape=Shape.REDUCING,
    raises=(
        ({"periods_per_year": 0}, r"periods_per_year must be >= 1"),
        ({"risk_free_rate": math.nan}, r"risk_free_rate must be a finite number"),
        ({"risk_free_rate": math.inf}, r"risk_free_rate must be a finite number"),
        ({"risk_free_rate": -math.inf}, r"risk_free_rate must be a finite number"),
    ),
    oracle=ulcer_performance_ratio_reference,
    # A normalized growth-factor curve run through CAGR over a scale-invariant ulcer index — neither invariant nor
    # homogeneous (tests/metrics/test_ulcer_performance_ratio.py module docstring).
    scale=ScaleExempt(
        reason="a normalized growth-factor curve run through CAGR over a scale-invariant ulcer index — neither "
        "invariant nor homogeneous"
    ),
    golden_input={"equity_curve": (1.1, 1.05, 1.2, 1.15, 1.3, 1.25, 1.4)},
    golden_output=(1.7927,),
    golden_params={"periods_per_year": 1},
    component_expr=lambda: (
        (cagr(pl.col("equity_curve"), periods_per_year=252) - 0.0) / ulcer_index(pl.col("equity_curve"))
    ),
    pins=(
        SpecPin(
            label="single_row",
            inputs={"equity_curve": (1.0,)},
            expected=(math.nan,),
            reason="one observation has zero excess growth and a zero ulcer index, so the ratio is 0/0, i.e. NaN "
            "(tests/metrics/test_ulcer_performance_ratio.py::test_single_row)",
        ),
        SpecPin(
            label="no_drawdown_is_inf",
            inputs={"equity_curve": (1.0, 1.1, 1.21)},
            expected=(math.inf,),
            reason="a monotonically rising curve has a zero ulcer index with positive excess growth, so the ratio "
            "is +inf (tests/metrics/test_ulcer_performance_ratio.py::test_no_drawdown_is_inf)",
            params_override={"periods_per_year": 1},
        ),
    ),
)
