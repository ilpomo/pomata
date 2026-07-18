"""Declaration for ``pomata.pnl.cost_fixed`` — the flat per-trade fee, elementwise, propagating, scale-invariant."""

import math

from pomata.pnl import cost_fixed
from tests_new.pnl.enums import BehaviorNan, BehaviorNull, ConventionSign, SpaceCost
from tests_new.pnl.harness import suite_pnl
from tests_new.pnl.oracles import reference_cost_fixed
from tests_new.support.declaration import Golden, Pin, ScaleAxis

COST_FIXED = suite_pnl(
    factory=cost_fixed,
    inputs=("quantity",),
    params={"fee": 1.0},
    null=BehaviorNull.PROPAGATES,
    nan=BehaviorNan.PROPAGATES,
    space=SpaceCost.CASH,
    sign=ConventionSign.LONG_SHORT,
    oracle=reference_cost_fixed,
    # Scaling the quantity by a positive constant leaves the trade-bar set unchanged, so the fee schedule is invariant,
    # degree 0.
    scaling=(ScaleAxis(roles=("quantity",), degree=0),),
    raises=(
        ({"fee": -1.0}, r"fee must be a finite number >= 0"),
        ({"fee": math.nan}, r"fee must be a finite number >= 0"),
        ({"fee": math.inf}, r"fee must be a finite number >= 0"),
        ({"fee": -math.inf}, r"fee must be a finite number >= 0"),
    ),
    golden=Golden(
        inputs={"quantity": (10.0, 10.0, -5.0, -5.0, 20.0)},
        output=(1.0, 0.0, 1.0, 0.0, 1.0),
    ),
    pins=(
        Pin(
            label="single_row",
            inputs={"quantity": (10.0,)},
            expected=(1.0,),
            reason="a one-element series charges the fee on the entry trade, not null",
        ),
        Pin(
            label="null_takes_precedence_over_nan",
            inputs={"quantity": (10.0, None, math.nan, 20.0)},
            expected=(1.0, None, None, math.nan),
            reason="the traded row where a NaN quantity meets the previous row's null yields null (null wins), while "
            "the next trade off the NaN is NaN",
        ),
        Pin(
            label="consecutive_infinities",
            inputs={"quantity": (math.inf, math.inf, 1.0, -math.inf)},
            expected=(1.0, math.nan, 1.0, 1.0),
            reason="two consecutive equal-sign infinities make the turnover inf - inf = NaN and the masked fee NaN; "
            "the property tiers set allow_infinity=False",
        ),
    ),
)
