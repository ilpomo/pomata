"""Declaration for ``pomata.metrics.capture_downside_ratio`` — reducing, down-market capture, scale-exempt."""

import math

from pomata.metrics import capture_downside_ratio
from tests.metrics.enums import Annualization, BehaviorNan, BehaviorNull, Degenerate
from tests.metrics.harness import suite_metrics
from tests.metrics.oracles import reference_capture_downside_ratio
from tests.support.declaration import Golden, Pin, ScaleExempt

CAPTURE_DOWNSIDE_RATIO = suite_metrics(
    factory=capture_downside_ratio,
    inputs=("returns", "benchmark"),
    params={"periods_per_year": 252},
    null=BehaviorNull.SKIPPED,
    nan=BehaviorNan.POISONS,
    annualization=Annualization.GEOMETRIC,
    degenerate=Degenerate.RATIO_SIGNED_INF_OR_NAN,
    oracle=reference_capture_downside_ratio,
    scaling=ScaleExempt(reason="a ratio of two annualized geometric returns — neither scale-invariant nor homogeneous"),
    raises=(({"periods_per_year": 0}, r"periods_per_year must be >= 1"),),
    golden=Golden(
        inputs={
            "returns": (0.012, -0.008, 0.02, -0.015, 0.005, 0.0, -0.02, 0.018),
            "benchmark": (0.01, -0.006, 0.018, -0.012, 0.004, 0.002, -0.018, 0.015),
        },
        output=(1.0224,),
    ),
    pins=(
        Pin(
            label="no_down_market_is_null",
            inputs={"returns": (0.01, 0.02, 0.03), "benchmark": (0.01, 0.02, 0.03)},
            expected=(None,),
            reason="with no negative-benchmark period the ratio is undefined, so the result is null ",
        ),
        Pin(
            label="return_leg_wiped_out_is_nan",
            inputs={"returns": (0.02, -1.5, 0.01), "benchmark": (-0.01, -0.02, -0.03)},
            expected=(math.nan,),
            reason="a selected portfolio return <= -1 wipes that leg out of the geometric-growth domain, a loud NaN",
        ),
        Pin(
            label="benchmark_leg_wiped_out_is_nan",
            inputs={"returns": (0.02, -0.03, 0.01), "benchmark": (-0.01, -1.2, -0.03)},
            expected=(math.nan,),
            reason="a selected benchmark value <= -1 wipes that leg out of the geometric-growth domain, a loud NaN",
        ),
    ),
)
