"""Spec for ``pomata.pnl.returns_net`` — the gross-return-minus-cost difference, propagating, jointly degree-1."""

import math

from tests.pnl.oracles import returns_net_reference
from tests.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.pnl import returns_net

RETURNS_NET = Spec(
    factory=returns_net,
    inputs=("returns_gross", "cost"),
    params={},
    shape=Shape.SERIES,
    oracle=returns_net_reference,
    # The difference is degree-1 homogeneous when both inputs are scaled together.
    scale=(ScaleAxis(roles=("returns_gross", "cost"), degree=1),),
    golden_input={
        "returns_gross": (0.05, -0.02, 0.03, 0.01, 0.0),
        "cost": (0.0005, 0.0015, 0.0005, 0.0, 0.0005),
    },
    golden_output=(0.0495, -0.0215, 0.0295, 0.01, -0.0005),
    pins=(
        SpecPin(
            label="single_row",
            inputs={"returns_gross": (0.05,), "cost": (0.0005,)},
            expected=(0.0495,),
            reason="a one-row series resolves to the single difference 0.05 - 0.0005 = 0.0495",
        ),
        SpecPin(
            label="null_takes_precedence_over_nan",
            inputs={"returns_gross": (None, 0.05), "cost": (math.nan, 0.0005)},
            expected=(None, 0.0495),
            reason="a null in one input against a NaN in the other at the same row yields null (null wins over NaN)",
        ),
        SpecPin(
            label="consecutive_infinities_make_nan",
            inputs={"returns_gross": (math.inf, 0.05), "cost": (math.inf, 0.01)},
            expected=(math.nan, 0.04),
            reason="a same-sign infinite gross return and cost cancel to inf - inf = NaN; the property tiers set "
            "allow_infinity=False",
        ),
    ),
)
