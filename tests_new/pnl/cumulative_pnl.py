"""Declaration for ``pomata.pnl.cumulative_pnl`` — the additive running total, bridged nulls, latched NaNs, degree-1."""

import math

from pomata.pnl import cumulative_pnl
from tests_new.pnl.enums import BehaviorNan, BehaviorNull, ConventionSign, SpaceCost
from tests_new.pnl.harness import suite_pnl
from tests_new.pnl.oracles import reference_cumulative_pnl
from tests_new.support.declaration import Golden, Pin, ScaleAxis

CUMULATIVE_PNL = suite_pnl(
    factory=cumulative_pnl,
    inputs=("returns",),
    params={},
    null=BehaviorNull.BRIDGED,
    nan=BehaviorNan.LATCHES,
    space=SpaceCost.CASH,
    sign=ConventionSign.LONG_SHORT,
    oracle=reference_cumulative_pnl,
    # The linear (additive) twin of equity_curve: degree-1 homogeneous, not scale-exempt.
    scaling=(ScaleAxis(roles=("returns",), degree=1),),
    golden=Golden(
        inputs={"returns": (0.1, -0.05, 0.2, 0.1)},
        output=(0.1, 0.05, 0.25, 0.35),
    ),
    pins=(
        Pin(
            label="single_row",
            inputs={"returns": (0.1,)},
            expected=(0.1,),
            reason="a one-element series resolves to that single return with no warm-up",
        ),
        Pin(
            label="warmup_leading_null",
            inputs={"returns": (None, 0.1, 0.2, -0.05)},
            expected=(None, 0.1, 0.3, 0.25),
            reason="a leading warm-up null stays null and the running total begins at the first defined return; the "
            "function declares no warm-up window, so no generic warm-up rung exercises a leading null",
        ),
        Pin(
            label="infinite_pnl_latches_nan_on_cancellation",
            inputs={"returns": (math.inf, 0.1, -math.inf, 0.2)},
            expected=(math.inf, math.inf, math.nan, math.nan),
            reason="the running total holds +inf until the opposite infinity cancels it to inf - inf = NaN, which "
            "then contaminates every later row; the property tiers set allow_infinity=False",
        ),
    ),
)
