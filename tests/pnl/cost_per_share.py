"""Declaration for ``pomata.pnl.cost_per_share`` — the per-unit commission, turnover-scaled, propagating, degree-1."""

import math

from pomata.pnl import cost_per_share
from tests.pnl.enums import BehaviorNan, BehaviorNull, ConventionSign, SpaceCost
from tests.pnl.harness import suite_pnl
from tests.pnl.oracles import reference_cost_per_share
from tests.support.declaration import Golden, Pin, ScaleAxis

COST_PER_SHARE = suite_pnl(
    factory=cost_per_share,
    inputs=("quantity",),
    params={"fee": 0.01},
    null=BehaviorNull.PROPAGATES,
    nan=BehaviorNan.PROPAGATES,
    space=SpaceCost.CASH,
    sign=ConventionSign.LONG_SHORT,
    oracle=reference_cost_per_share,
    # Degree-1 homogeneous in quantity (it scales turnover by a fixed fee).
    scaling=(ScaleAxis(roles=("quantity",), degree=1),),
    raises=(
        ({"fee": -0.01}, r"fee must be a finite number >= 0"),
        ({"fee": math.nan}, r"fee must be a finite number >= 0"),
        ({"fee": math.inf}, r"fee must be a finite number >= 0"),
        ({"fee": -math.inf}, r"fee must be a finite number >= 0"),
    ),
    golden=Golden(
        inputs={"quantity": (10.0, 10.0, -5.0, -5.0, 20.0)},
        output=(0.1, 0.0, 0.15, 0.0, 0.25),
    ),
    pins=(
        Pin(
            label="single_row",
            inputs={"quantity": (10.0,)},
            expected=(0.1,),
            reason="a one-element series resolves to |quantity| * fee = 10 * 0.01 = 0.1 on the entry trade",
        ),
        Pin(
            label="null_takes_precedence_over_nan",
            inputs={"quantity": (10.0, None, math.nan, 20.0)},
            expected=(0.1, None, None, math.nan),
            reason="the traded row where a NaN quantity meets the previous row's null yields null (null wins), while "
            "the next trade off the NaN is NaN",
        ),
        Pin(
            label="consecutive_infinities",
            inputs={"quantity": (math.inf, math.inf, 1.0, -math.inf)},
            expected=(math.inf, math.nan, math.inf, math.inf),
            reason="two consecutive equal-sign infinities make inf - inf = NaN turnover at the second bar; the "
            "property tiers set allow_infinity=False, so only this pin reaches the branch",
        ),
    ),
)
