"""Declaration for ``pomata.metrics.tail_ratio_rolling`` — the rolling right-tail over left-tail quantile ratio."""

import math

from pomata.metrics import tail_ratio_rolling
from tests_new.metrics.enums import BehaviorNan, BehaviorNull
from tests_new.metrics.harness import suite_metrics
from tests_new.metrics.oracles import reference_tail_ratio_rolling
from tests_new.metrics.tail_ratio import TAIL_RATIO
from tests_new.support.declaration import Golden, Pin, ScaleAxis

TAIL_RATIO_ROLLING = suite_metrics(
    factory=tail_ratio_rolling,
    inputs=("returns",),
    params={"window": 5},
    null=BehaviorNull.IN_WINDOW_IS_NULL,
    nan=BehaviorNan.PROPAGATES,
    rolling_of=TAIL_RATIO,
    window="window",
    warmup=4,
    oracle=reference_tail_ratio_rolling,
    scaling=(ScaleAxis(roles=("returns",), degree=0),),
    raises=(({"window": 0}, r"window must be >= 1"),),
    golden=Golden(
        inputs={"returns": (0.01, -0.02, 0.03, -0.01, 0.02, 0.0, -0.015)},
        output=(None, None, None, None, 1.5556, 1.5556, 2.0),
    ),
    pins=(
        Pin(
            label="zero_left_tail_window_is_inf",
            inputs={"returns": (0.0, 0.0, 0.0, 0.0, 0.02)},
            expected=(None, None, None, None, math.inf),
            reason="a window with a zero 5th-percentile and a non-zero 95th gives +inf ",
            params_override={"window": 5},
        ),
        Pin(
            label="all_zero_window_is_nan",
            inputs={"returns": (0.0, 0.0, 0.0)},
            expected=(None, None, math.nan),
            reason="an all-zero window gives 0/0, so the ratio is NaN ",
            params_override={"window": 3},
        ),
    ),
)
