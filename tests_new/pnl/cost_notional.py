"""Declaration for ``pomata.pnl.cost_notional`` — the bps-of-traded-notional fee, turnover-scaled, propagating."""

import math

from pomata.pnl import cost_notional
from tests_new.pnl.enums import BehaviorNan, BehaviorNull, ConventionSign, SpaceCost
from tests_new.pnl.harness import suite_pnl
from tests_new.pnl.oracles import reference_cost_notional
from tests_new.support.declaration import Golden, Pin, ScaleAxis

COST_NOTIONAL = suite_pnl(
    factory=cost_notional,
    inputs=("quantity", "price"),
    params={"rate": 0.001},
    null=BehaviorNull.PROPAGATES,
    nan=BehaviorNan.PROPAGATES,
    space=SpaceCost.CASH,
    sign=ConventionSign.LONG_SHORT,
    oracle=reference_cost_notional,
    # Degree-1 homogeneous in quantity; only the quantity axis is exercised.
    scaling=(ScaleAxis(roles=("quantity",), degree=1),),
    raises=(
        ({"rate": -0.001}, r"rate must be a finite number >= 0"),
        ({"rate": math.nan}, r"rate must be a finite number >= 0"),
        ({"rate": math.inf}, r"rate must be a finite number >= 0"),
        ({"rate": -math.inf}, r"rate must be a finite number >= 0"),
    ),
    golden=Golden(
        inputs={
            "quantity": (10.0, 10.0, -5.0, -5.0, 20.0),
            "price": (100.0, 102.0, 101.0, 104.0, 103.0),
        },
        output=(1.0, 0.0, 1.515, 0.0, 2.575),
    ),
    pins=(
        Pin(
            label="single_row",
            inputs={"quantity": (10.0,), "price": (100.0,)},
            expected=(1.0,),
            reason="the first row charges on the entry trade |quantity| * price * rate = 10 * 100 * 0.001 = 1.0",
        ),
        Pin(
            label="null_takes_precedence_over_nan",
            inputs={"quantity": (10.0, None), "price": (100.0, math.nan)},
            expected=(1.0, None),
            reason="a null in quantity against a NaN in price at the same row yields null (null wins)",
        ),
        Pin(
            label="consecutive_infinities_make_nan",
            inputs={"quantity": (math.inf, math.inf, 1.0, -math.inf), "price": (100.0, 100.0, 100.0, 100.0)},
            expected=(math.inf, math.nan, math.inf, math.inf),
            reason="two consecutive equal-sign infinite quantities make inf - inf = NaN turnover at the second bar; "
            "the property tiers set allow_infinity=False",
        ),
    ),
)
