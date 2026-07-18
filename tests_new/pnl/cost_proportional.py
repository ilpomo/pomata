"""Declaration for ``pomata.pnl.cost_proportional`` — the bps-of-weight-turnover fee, propagating, degree-1."""

import math

from pomata.pnl import cost_proportional
from tests_new.pnl.enums import BehaviorNan, BehaviorNull, ConventionSign, SpaceCost
from tests_new.pnl.harness import suite_pnl
from tests_new.pnl.oracles import reference_cost_proportional
from tests_new.support.declaration import Golden, Pin, ScaleAxis

COST_PROPORTIONAL = suite_pnl(
    factory=cost_proportional,
    inputs=("weight",),
    params={"rate": 0.001},
    null=BehaviorNull.PROPAGATES,
    nan=BehaviorNan.PROPAGATES,
    space=SpaceCost.RETURNS,
    sign=ConventionSign.LONG_SHORT,
    oracle=reference_cost_proportional,
    # Degree-1 homogeneous in the weight (it scales turnover by a fixed rate).
    scaling=(ScaleAxis(roles=("weight",), degree=1),),
    raises=(
        ({"rate": -0.001}, r"rate must be a finite number >= 0"),
        ({"rate": math.nan}, r"rate must be a finite number >= 0"),
        ({"rate": math.inf}, r"rate must be a finite number >= 0"),
        ({"rate": -math.inf}, r"rate must be a finite number >= 0"),
    ),
    golden=Golden(
        inputs={"weight": (0.5, 1.0, -0.5, -0.5, 0.0)},
        output=(0.0005, 0.0005, 0.0015, 0.0, 0.0005),
    ),
    pins=(
        Pin(
            label="single_row",
            inputs={"weight": (0.5,)},
            expected=(0.0005,),
            reason="a one-element series resolves to |weight| * rate = 0.5 * 0.001 = 0.0005 on the entry trade",
        ),
        Pin(
            label="null_takes_precedence_over_nan",
            inputs={"weight": (0.5, None, math.nan, 1.0)},
            expected=(0.0005, None, None, math.nan),
            reason="the turnover row where a NaN weight meets the previous row's null yields null (null wins), while "
            "the next turnover off the NaN is NaN",
        ),
        Pin(
            label="consecutive_infinities_make_nan",
            inputs={"weight": (math.inf, math.inf, 1.0, -math.inf)},
            expected=(math.inf, math.nan, math.inf, math.inf),
            reason="two consecutive equal-sign infinities make inf - inf = NaN turnover at the second bar; the "
            "property tiers set allow_infinity=False",
        ),
    ),
)
