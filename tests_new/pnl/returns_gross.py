"""Declaration for ``pomata.pnl.returns_gross`` — the weight-times-asset-return product, propagating, degree-1."""

import math

from pomata.pnl import returns_gross
from tests_new.pnl.enums import BehaviorNan, BehaviorNull, ConventionSign, SpaceCost
from tests_new.pnl.harness import suite_pnl
from tests_new.pnl.oracles import reference_returns_gross
from tests_new.support.declaration import Golden, Pin, ScaleAxis

RETURNS_GROSS = suite_pnl(
    factory=returns_gross,
    inputs=("weight", "asset_returns"),
    params={},
    null=BehaviorNull.PROPAGATES,
    nan=BehaviorNan.PROPAGATES,
    space=SpaceCost.RETURNS,
    sign=ConventionSign.LONG_SHORT,
    oracle=reference_returns_gross,
    # Degree-1 homogeneous in the weight; only the weight axis is exercised.
    scaling=(ScaleAxis(roles=("weight",), degree=1),),
    golden=Golden(
        inputs={
            "weight": (1.0, 0.5, -1.0, -1.0, 0.5),
            "asset_returns": (0.02, -0.01, 0.03, -0.02, 0.04),
        },
        output=(0.02, -0.005, -0.03, 0.02, 0.02),
    ),
    pins=(
        Pin(
            label="single_row",
            inputs={"weight": (0.5,), "asset_returns": (0.04,)},
            expected=(0.02,),
            reason="a one-row series resolves to the single product 0.5 * 0.04 = 0.02",
        ),
        Pin(
            label="null_takes_precedence_over_nan",
            inputs={"weight": (None, 0.5), "asset_returns": (math.nan, 0.04)},
            expected=(None, 0.02),
            reason="a null in one input against a NaN in the other at the same row yields null (null wins over NaN)",
        ),
        Pin(
            label="infinite_legs_sign_the_return",
            inputs={"weight": (math.inf, -1.0, -math.inf), "asset_returns": (0.1, math.inf, -0.2)},
            expected=(math.inf, -math.inf, math.inf),
            reason="the gross return keeps the sign of weight * asset_returns even at infinite magnitude; the "
            "property tiers set allow_infinity=False",
        ),
    ),
)
