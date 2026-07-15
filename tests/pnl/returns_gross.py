"""Spec for ``pomata.pnl.returns_gross`` — the weight-times-asset-return per-leg product, propagating, degree-1."""

import math

from tests.pnl.oracles import returns_gross_reference
from tests.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.pnl import returns_gross

RETURNS_GROSS = Spec(
    factory=returns_gross,
    inputs=("weight", "asset_returns"),
    params={},
    shape=Shape.SERIES,
    oracle=returns_gross_reference,
    # Degree-1 homogeneous in the weight; only the weight axis is exercised.
    scale=(ScaleAxis(roles=("weight",), degree=1),),
    golden_input={
        "weight": (1.0, 0.5, -1.0, -1.0, 0.5),
        "asset_returns": (0.02, -0.01, 0.03, -0.02, 0.04),
    },
    golden_output=(0.02, -0.005, -0.03, 0.02, 0.02),
    pins=(
        SpecPin(
            label="single_row",
            inputs={"weight": (0.5,), "asset_returns": (0.04,)},
            expected=(0.02,),
            reason="a one-row series resolves to the single product 0.5 * 0.04 = 0.02",
        ),
        SpecPin(
            label="null_takes_precedence_over_nan",
            inputs={"weight": (None, 0.5), "asset_returns": (math.nan, 0.04)},
            expected=(None, 0.02),
            reason="a null in one input against a NaN in the other at the same row yields null (null wins over NaN)",
        ),
        SpecPin(
            label="infinite_legs_sign_the_return",
            inputs={"weight": (math.inf, -1.0, -math.inf), "asset_returns": (0.1, math.inf, -0.2)},
            expected=(math.inf, -math.inf, math.inf),
            reason="the gross return keeps the sign of weight * asset_returns even at infinite magnitude; the "
            "property tiers set allow_infinity=False",
        ),
    ),
)
