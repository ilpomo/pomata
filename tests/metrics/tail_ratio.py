"""Spec for ``pomata.metrics.tail_ratio`` — reducing, the right-tail quantile over the left-tail magnitude,
scale-invariant.
"""

import math

from tests.metrics.oracles import tail_ratio_reference
from tests.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.metrics import tail_ratio

TAIL_RATIO = Spec(
    factory=tail_ratio,
    inputs=("returns",),
    params={},
    shape=Shape.REDUCING,
    oracle=tail_ratio_reference,
    # A ratio of two quantiles is scale-invariant
    scale=(ScaleAxis(roles=("returns",), degree=0),),
    golden_input={"returns": (0.02, -0.04, 0.01, -0.06, 0.03)},
    golden_output=(0.5,),
    pins=(
        SpecPin(
            label="single_row",
            inputs={"returns": (0.05,)},
            expected=(1.0,),
            reason="a one-element series has equal tails, so the ratio is 1.0",
        ),
        SpecPin(
            label="constant_is_one",
            inputs={"returns": (0.01, 0.01, 0.01)},
            expected=(1.0,),
            reason="a constant series has equal 5th/95th percentiles, so the ratio is 1.0 ",
        ),
        SpecPin(
            label="zero_left_tail_is_inf",
            inputs={"returns": (0.0, 0.0, 0.0, 0.0, 0.02)},
            expected=(math.inf,),
            reason="a zero 5th-percentile against a non-zero 95th gives +inf ",
        ),
        SpecPin(
            label="all_zero_is_nan",
            inputs={"returns": (0.0, 0.0, 0.0)},
            expected=(math.nan,),
            reason="an all-zero series gives 0/0 at both tails, so the ratio is NaN ",
        ),
    ),
)
