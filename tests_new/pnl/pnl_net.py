"""Declaration for ``pomata.pnl.pnl_net`` — the gross-minus-cost difference, elementwise, jointly degree-1."""

import math

from pomata.pnl import pnl_net
from tests_new.pnl.enums import BehaviorNan, BehaviorNull, ConventionSign, SpaceCost
from tests_new.pnl.harness import suite_pnl
from tests_new.pnl.oracles import reference_pnl_net
from tests_new.support.declaration import Golden, Pin, ScaleAxis

PNL_NET = suite_pnl(
    factory=pnl_net,
    inputs=("pnl_gross", "cost"),
    params={},
    null=BehaviorNull.PROPAGATES,
    nan=BehaviorNan.PROPAGATES,
    space=SpaceCost.CASH,
    sign=ConventionSign.LONG_SHORT,
    oracle=reference_pnl_net,
    # The difference is degree-1 homogeneous when both inputs are scaled together.
    scaling=(ScaleAxis(roles=("pnl_gross", "cost"), degree=1),),
    golden=Golden(
        inputs={
            "pnl_gross": (20.0, 5.0, -15.0, -20.0, 8.0),
            "cost": (2.0, 0.0, 3.0, 0.0, 1.0),
        },
        output=(18.0, 5.0, -18.0, -20.0, 7.0),
    ),
    pins=(
        Pin(
            label="single_row",
            inputs={"pnl_gross": (20.0,), "cost": (2.0,)},
            expected=(18.0,),
            reason="a one-row series resolves to the single difference 20 - 2 = 18",
        ),
        Pin(
            label="null_takes_precedence_over_nan",
            inputs={"pnl_gross": (None, 20.0), "cost": (math.nan, 2.0)},
            expected=(None, 18.0),
            reason="a null in one input against a NaN in the other at the same row yields null (null wins over NaN)",
        ),
        Pin(
            label="consecutive_infinities_make_nan",
            inputs={"pnl_gross": (math.inf, 5.0), "cost": (math.inf, 1.0)},
            expected=(math.nan, 4.0),
            reason="a same-sign infinite gross and cost cancel to inf - inf = NaN; the property tiers set "
            "allow_infinity=False",
        ),
    ),
)
