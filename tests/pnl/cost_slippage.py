"""Declaration for ``pomata.pnl.cost_slippage`` — the half-spread crossing charge, turnover-scaled, propagating."""

import math

from pomata.pnl import cost_slippage
from tests.pnl.enums import BehaviorNan, BehaviorNull, ConventionSign, SpaceCost
from tests.pnl.harness import suite_pnl
from tests.pnl.oracles import reference_cost_slippage
from tests.support.declaration import Golden, Pin, ScaleAxis

COST_SLIPPAGE = suite_pnl(
    factory=cost_slippage,
    inputs=("weight",),
    params={"half_spread": 0.002},
    null=BehaviorNull.PROPAGATES,
    nan=BehaviorNan.PROPAGATES,
    space=SpaceCost.RETURNS,
    sign=ConventionSign.LONG_SHORT,
    oracle=reference_cost_slippage,
    # Degree-1 homogeneous in the weight (it scales turnover by a fixed half-spread).
    scaling=(ScaleAxis(roles=("weight",), degree=1),),
    raises=(
        ({"half_spread": -0.002}, r"half_spread must be a finite number >= 0"),
        ({"half_spread": math.nan}, r"half_spread must be a finite number >= 0"),
        ({"half_spread": math.inf}, r"half_spread must be a finite number >= 0"),
        ({"half_spread": -math.inf}, r"half_spread must be a finite number >= 0"),
    ),
    golden=Golden(
        inputs={"weight": (0.5, 1.0, -0.5, -0.5, 0.0)},
        output=(0.001, 0.001, 0.003, 0.0, 0.001),
    ),
    pins=(
        Pin(
            label="single_row",
            inputs={"weight": (0.5,)},
            expected=(0.001,),
            reason="a one-element series resolves to |weight| * half_spread = 0.5 * 0.002 = 0.001 on the entry trade",
        ),
        Pin(
            label="null_takes_precedence_over_nan",
            inputs={"weight": (0.5, None, math.nan, 1.0)},
            expected=(0.001, None, None, math.nan),
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
