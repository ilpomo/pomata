"""Spec for ``pomata.metrics.tail_ratio_rolling`` — the rolling right-tail over left-tail quantile ratio,
scale-invariant.
"""

import math

from tests.metrics.oracles import tail_ratio_rolling_reference
from tests_new.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.metrics import tail_ratio_rolling

TAIL_RATIO_ROLLING = Spec(
    factory=tail_ratio_rolling,
    inputs=("returns",),
    params={"window": 5},
    shape=Shape.SERIES,
    warmup=4,
    raises=(({"window": 0}, r"window must be >= 1"),),
    oracle=tail_ratio_rolling_reference,
    # A ratio of two quantiles per window is scale-invariant (test_tail_ratio_rolling.py::test_scale_invariance).
    scale=(ScaleAxis(roles=("returns",), degree=0),),
    golden_input={"returns": (0.01, -0.02, 0.03, -0.01, 0.02, 0.0, -0.015)},
    golden_output=(None, None, None, None, 1.5556, 1.5556, 2.0),
    pins=(
        SpecPin(
            label="zero_left_tail_window_is_inf",
            inputs={"returns": (0.0, 0.0, 0.0, 0.0, 0.02)},
            expected=(None, None, None, None, math.inf),
            reason="a window with a zero 5th-percentile and a non-zero 95th gives +inf "
            "(test_tail_ratio_rolling.py::test_zero_left_tail_window_is_inf)",
            params_override={"window": 5},
        ),
        SpecPin(
            label="all_zero_window_is_nan",
            inputs={"returns": (0.0, 0.0, 0.0)},
            expected=(None, None, math.nan),
            reason="an all-zero window gives 0/0, so the ratio is NaN "
            "(test_tail_ratio_rolling.py::test_all_zero_window_is_nan)",
            params_override={"window": 3},
        ),
    ),
)
