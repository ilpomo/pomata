"""Declaration for ``pomata.pnl.equity_curve`` — compounding cumulation, bridged nulls, latched NaNs, scale-exempt."""

import math

from pomata.pnl import equity_curve
from tests.pnl.enums import BehaviorNan, BehaviorNull, ConventionSign, SpaceCost
from tests.pnl.harness import suite_pnl
from tests.pnl.oracles import reference_equity_curve
from tests.support.declaration import Golden, Pin, ScaleExempt

EQUITY_CURVE = suite_pnl(
    factory=equity_curve,
    inputs=("returns",),
    params={},
    null=BehaviorNull.BRIDGED,
    nan=BehaviorNan.LATCHES,
    space=SpaceCost.RETURNS,
    sign=ConventionSign.LONG_SHORT,
    oracle=reference_equity_curve,
    # A nonlinear compounding transform — neither scale-invariant nor homogeneous.
    scaling=ScaleExempt(reason="nonlinear compounding: neither scale-invariant nor homogeneous"),
    golden=Golden(
        inputs={"returns": (0.1, -0.05, 0.2, 0.1)},
        output=(1.1, 1.045, 1.254, 1.3794),
    ),
    pins=(
        Pin(
            label="single_row",
            inputs={"returns": (0.1,)},
            expected=(1.1,),
            reason="a one-element series resolves to 1 + return with no warm-up of its own",
        ),
        Pin(
            label="leading_null_passthrough",
            inputs={"returns": (None, 0.1, 0.2, -0.05)},
            expected=(None, 1.1, 1.32, 1.254),
            reason="a leading warm-up null stays null and the compounded curve begins at the first defined return; "
            "the function declares no warm-up window, so no generic warm-up rung exercises a leading null",
        ),
        Pin(
            label="infinite_return_flips_the_curve_sign",
            inputs={"returns": (math.inf, 0.1, -math.inf, 0.2)},
            expected=(math.inf, math.inf, -math.inf, -math.inf),
            reason="a +inf return inflates the compounded curve to +inf and a later -inf factor flips it to -inf, "
            "which then persists; the property tiers set allow_infinity=False",
        ),
    ),
)
