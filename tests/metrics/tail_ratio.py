"""Declaration for ``pomata.metrics.tail_ratio`` — reducing, the right-tail quantile over the left-tail magnitude."""

import math

from pomata.metrics import tail_ratio
from tests.metrics.enums import Annualization, BehaviorNan, BehaviorNull, Degenerate
from tests.metrics.harness import suite_metrics
from tests.metrics.oracles import reference_tail_ratio
from tests.support.declaration import Golden, Pin, ScaleAxis

TAIL_RATIO = suite_metrics(
    factory=tail_ratio,
    inputs=("returns",),
    params={},
    null=BehaviorNull.SKIPPED,
    nan=BehaviorNan.POISONS,
    annualization=Annualization.NONE,
    degenerate=Degenerate.RATIO_SIGNED_INF_OR_NAN,
    oracle=reference_tail_ratio,
    scaling=(ScaleAxis(roles=("returns",), degree=0),),
    golden=Golden(inputs={"returns": (0.02, -0.04, 0.01, -0.06, 0.03)}, output=(0.5,)),
    pins=(
        Pin(
            label="single_row",
            inputs={"returns": (0.05,)},
            expected=(1.0,),
            reason="a one-element series has equal tails, so the ratio is 1.0",
        ),
        Pin(
            label="constant_is_one",
            inputs={"returns": (0.01, 0.01, 0.01)},
            expected=(1.0,),
            reason="a constant series has equal 5th/95th percentiles, so the ratio is 1.0 ",
        ),
        Pin(
            label="zero_left_tail_is_inf",
            inputs={"returns": (0.0, 0.0, 0.0, 0.0, 0.02)},
            expected=(math.inf,),
            reason="a zero 5th-percentile against a non-zero 95th gives +inf ",
        ),
        Pin(
            label="all_zero_is_nan",
            inputs={"returns": (0.0, 0.0, 0.0)},
            expected=(math.nan,),
            reason="an all-zero series gives 0/0 at both tails, so the ratio is NaN ",
        ),
    ),
)
