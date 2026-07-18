"""
Declaration for ``pomata.metrics.ulcer_performance_ratio`` — reducing, excess CAGR per unit of ulcer index, scale-
exempt.
"""

import math

from pomata.metrics import ulcer_performance_ratio
from tests_new.metrics.enums import Annualization, BehaviorNan, BehaviorNull, Degenerate
from tests_new.metrics.harness import suite_metrics
from tests_new.metrics.oracles import reference_ulcer_performance_ratio
from tests_new.support.declaration import Golden, Pin, ScaleExempt

ULCER_PERFORMANCE_RATIO = suite_metrics(
    factory=ulcer_performance_ratio,
    inputs=("equity_curve",),
    params={"periods_per_year": 252, "risk_free_rate": 0.0},
    null=BehaviorNull.SKIPPED,
    nan=BehaviorNan.POISONS,
    annualization=Annualization.GEOMETRIC,
    degenerate=Degenerate.RATIO_SIGNED_INF_OR_NAN,
    oracle=reference_ulcer_performance_ratio,
    scaling=ScaleExempt(
        reason="a normalized growth-factor curve run through CAGR over a scale-invariant ulcer index — "
        "neither invariant nor homogeneous"
    ),
    raises=(
        ({"periods_per_year": 0}, r"periods_per_year must be >= 1"),
        ({"risk_free_rate": math.nan}, r"risk_free_rate must be a finite number"),
        ({"risk_free_rate": math.inf}, r"risk_free_rate must be a finite number"),
        ({"risk_free_rate": -math.inf}, r"risk_free_rate must be a finite number"),
    ),
    golden=Golden(
        inputs={"equity_curve": (1.1, 1.05, 1.2, 1.15, 1.3, 1.25, 1.4)},
        output=(1.7927,),
        params={"periods_per_year": 1},
    ),
    pins=(
        Pin(
            label="single_row",
            inputs={"equity_curve": (1.0,)},
            expected=(math.nan,),
            reason="one observation has zero excess growth and a zero ulcer index, so the ratio is 0/0, i.e. NaN",
        ),
        Pin(
            label="no_drawdown_is_inf",
            inputs={"equity_curve": (1.0, 1.1, 1.21)},
            expected=(math.inf,),
            reason="a monotonically rising curve has a zero ulcer index with positive excess growth, so the "
            "ratio is +inf",
            params_override={"periods_per_year": 1},
        ),
        Pin(
            label="no_drawdown_zero_growth_is_nan",
            inputs={"equity_curve": (1.0, 1.0, 1.0)},
            expected=(math.nan,),
            reason="a flat multi-row curve has a zero ulcer index and zero excess growth, so the ratio is a "
            "0/0, i.e. NaN — the degenerate-denominator NaN beside the +inf pin",
        ),
    ),
)
