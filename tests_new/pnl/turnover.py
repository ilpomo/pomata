"""Declaration for ``pomata.pnl.turnover`` — the absolute one-bar weight change, propagating, degree-1."""

import math

from pomata.pnl import turnover
from tests_new.pnl.enums import BehaviorNan, BehaviorNull, ConventionSign, SpaceCost
from tests_new.pnl.harness import suite_pnl
from tests_new.pnl.oracles import reference_turnover
from tests_new.support.declaration import Golden, Pin, ScaleAxis

TURNOVER = suite_pnl(
    factory=turnover,
    inputs=("weight",),
    params={},
    null=BehaviorNull.PROPAGATES,
    nan=BehaviorNan.PROPAGATES,
    space=SpaceCost.RETURNS,
    sign=ConventionSign.LONG_SHORT,
    oracle=reference_turnover,
    # The absolute weight change |w_t - w_{t-1}| is degree-1 homogeneous.
    scaling=(ScaleAxis(roles=("weight",), degree=1),),
    golden=Golden(
        inputs={"weight": (0.5, 1.0, -0.5, -0.5, 0.0)},
        output=(0.5, 0.5, 1.5, 0.0, 0.5),
    ),
    pins=(
        Pin(
            label="single_row",
            inputs={"weight": (0.7,)},
            expected=(0.7,),
            reason="a one-element series resolves to |weight_0| = 0.7 (the entry trade off a flat start), not null",
        ),
        Pin(
            label="null_takes_precedence_over_nan",
            inputs={"weight": (0.5, None, math.nan, 1.0)},
            expected=(0.5, None, None, math.nan),
            reason="the difference row where a NaN weight meets the previous row's null yields null (null wins), while "
            "the next difference off the NaN is NaN",
        ),
        Pin(
            label="consecutive_infinities_make_nan",
            inputs={"weight": (math.inf, math.inf, 1.0, -math.inf)},
            expected=(math.inf, math.nan, math.inf, math.inf),
            reason="a single inf carries |inf| forward, while two consecutive equal-sign infinities make inf - inf = "
            "NaN; the property tiers set allow_infinity=False",
        ),
    ),
)
