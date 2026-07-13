"""Spec for ``pomata.pnl.cost_proportional`` — the bps-of-weight-turnover fee, propagating, degree-1."""

import math

from tests.pnl.oracles import cost_proportional_reference
from tests.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.pnl import cost_proportional

COST_PROPORTIONAL = Spec(
    factory=cost_proportional,
    inputs=("weight",),
    params={"rate": 0.001},
    shape=Shape.SERIES,
    raises=(
        ({"rate": -0.001}, r"rate must be a finite number >= 0"),
        ({"rate": math.nan}, r"rate must be a finite number >= 0"),
        ({"rate": math.inf}, r"rate must be a finite number >= 0"),
        ({"rate": -math.inf}, r"rate must be a finite number >= 0"),
    ),
    oracle=cost_proportional_reference,
    # Degree-1 homogeneous in the weight (it scales turnover by a fixed rate).
    scale=(ScaleAxis(roles=("weight",), degree=1),),
    golden_input={"weight": (0.5, 1.0, -0.5, -0.5, 0.0)},
    golden_output=(0.0005, 0.0005, 0.0015, 0.0, 0.0005),
    pins=(
        SpecPin(
            label="single_row",
            inputs={"weight": (0.5,)},
            expected=(0.0005,),
            reason="a one-element series resolves to |weight| * rate = 0.5 * 0.001 = 0.0005 on the entry trade",
        ),
        SpecPin(
            label="null_takes_precedence_over_nan",
            inputs={"weight": (0.5, None, math.nan, 1.0)},
            expected=(0.0005, None, None, math.nan),
            reason="the turnover row where a NaN weight meets the previous row's null yields null (null wins), while "
            "the next turnover off the NaN is NaN",
        ),
        SpecPin(
            label="consecutive_infinities_make_nan",
            inputs={"weight": (math.inf, math.inf, 1.0, -math.inf)},
            expected=(math.inf, math.nan, math.inf, math.inf),
            reason="two consecutive equal-sign infinities make inf - inf = NaN turnover at the second bar; the "
            "property tiers set allow_infinity=False",
        ),
    ),
)
