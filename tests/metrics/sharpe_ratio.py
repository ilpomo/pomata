"""
Declaration for ``pomata.metrics.sharpe_ratio`` — reducing, annualized, null-skipping, NaN-poisoning, scale-
invariant.
"""

import math

from pomata.metrics import sharpe_ratio
from tests.metrics.enums import Annualization, BehaviorNan, BehaviorNull, Degenerate
from tests.metrics.harness import suite_metrics
from tests.metrics.oracles import reference_sharpe_ratio
from tests.support.declaration import Golden, Pin, ScaleAxis

SHARPE_RATIO = suite_metrics(
    factory=sharpe_ratio,
    inputs=("returns",),
    params={"periods_per_year": 252, "risk_free_rate": 0.0},
    null=BehaviorNull.SKIPPED,
    nan=BehaviorNan.POISONS,
    annualization=Annualization.SQRT_TIME,
    degenerate=Degenerate.RATIO_SIGNED_INF_OR_NAN,
    oracle=reference_sharpe_ratio,
    scaling=(ScaleAxis(roles=("returns",), degree=0),),
    raises=(
        ({"periods_per_year": 0}, r"periods_per_year must be >= 1"),
        ({"risk_free_rate": math.nan}, r"risk_free_rate must be a finite number"),
        ({"risk_free_rate": -2.0}, r"risk_free_rate must be >= -1"),
        ({"risk_free_rate": math.inf}, r"risk_free_rate must be a finite number"),
        ({"risk_free_rate": -math.inf}, r"risk_free_rate must be a finite number"),
    ),
    golden=Golden(inputs={"returns": (0.03, -0.01, 0.02, -0.015, 0.01, 0.005, -0.02)}, output=(2.4285,)),
    pins=(
        Pin(
            label="single_row",
            inputs={"returns": (0.05,)},
            expected=(None,),
            reason="one observation has no sample dispersion, so the ratio is null",
        ),
        Pin(
            label="zero_volatility",
            inputs={"returns": (0.01, 0.01, 0.01, 0.01)},
            expected=(math.inf,),
            reason="a constant series has zero dispersion with a positive mean, so the ratio is +inf",
        ),
        Pin(
            label="zero_excess_is_nan",
            inputs={"returns": (0.0, 0.0, 0.0, 0.0)},
            expected=(math.nan,),
            reason="an all-zero series at a zero risk-free rate has zero mean AND zero dispersion, so the "
            "ratio is the 0/0 NaN — the exact-zero core of the constant regime; no conditioning "
            "filter is declared: the oracle detects an exactly-constant excess via min == max, "
            "mirroring the kernel's exact zero-dispersion pin, so the two sides agree in kind on "
            "every constant series",
        ),
    ),
)
