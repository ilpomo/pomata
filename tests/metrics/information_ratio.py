"""
Declaration for ``pomata.metrics.information_ratio`` — reducing, active return over tracking error, scale-invariant.
"""

import math

from pomata.metrics import information_ratio
from tests.metrics.enums import Annualization, BehaviorNan, BehaviorNull, Degenerate
from tests.metrics.harness import suite_metrics
from tests.metrics.oracles import reference_information_ratio
from tests.support.declaration import Golden, Pin, ScaleAxis

INFORMATION_RATIO = suite_metrics(
    factory=information_ratio,
    inputs=("returns", "benchmark"),
    params={"periods_per_year": 252},
    null=BehaviorNull.SKIPPED,
    nan=BehaviorNan.POISONS,
    annualization=Annualization.SQRT_TIME,
    degenerate=Degenerate.RATIO_SIGNED_INF_OR_NAN,
    oracle=reference_information_ratio,
    scaling=(ScaleAxis(roles=("returns", "benchmark"), degree=0),),
    raises=(({"periods_per_year": 0}, r"periods_per_year must be >= 1"),),
    golden=Golden(
        inputs={
            "returns": (0.012, -0.008, 0.02, -0.015, 0.005, 0.0, -0.02, 0.018),
            "benchmark": (0.01, -0.006, 0.018, -0.012, 0.004, 0.002, -0.018, 0.015),
        },
        output=(-0.842,),
    ),
    pins=(
        Pin(
            label="single_pair",
            inputs={"returns": (0.05,), "benchmark": (0.04,)},
            expected=(None,),
            reason="a single complete pair yields null — the tracking error needs two observations",
        ),
        Pin(
            label="zero_tracking_error_is_inf",
            inputs={"returns": (0.01, 0.01, 0.01), "benchmark": (0.0, 0.0, 0.0)},
            expected=(math.inf,),
            reason="a constant active series has zero tracking error with a positive mean, so the ratio is +inf",
        ),
        Pin(
            label="zero_active_is_nan",
            inputs={"returns": (0.01, 0.02, 0.03), "benchmark": (0.01, 0.02, 0.03)},
            expected=(math.nan,),
            reason="identical legs give an exactly-zero active series: zero mean over zero tracking error is "
            "the 0/0 NaN, resolved by the exact dispersion guard on both sides — the oracle detects a "
            "constant active series via min == max, mirroring the kernel's exact zero-dispersion pin; "
            "no conditioning filter is declared",
        ),
    ),
)
