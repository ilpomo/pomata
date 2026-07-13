"""Spec for ``pomata.pnl.cost_fixed`` — the flat per-trade fee, elementwise, propagating, scale-invariant."""

import math

from tests.pnl.oracles import cost_fixed_reference
from tests.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.pnl import cost_fixed

COST_FIXED = Spec(
    factory=cost_fixed,
    inputs=("quantity",),
    params={"fee": 1.0},
    shape=Shape.SERIES,
    raises=(
        ({"fee": -1.0}, r"fee must be a finite number >= 0"),
        ({"fee": math.nan}, r"fee must be a finite number >= 0"),
        ({"fee": math.inf}, r"fee must be a finite number >= 0"),
        ({"fee": -math.inf}, r"fee must be a finite number >= 0"),
    ),
    oracle=cost_fixed_reference,
    # Scaling the quantity by a positive constant leaves the trade-bar set unchanged, so the fee schedule is invariant,
    # degree 0
    scale=(ScaleAxis(roles=("quantity",), degree=0),),
    golden_input={"quantity": (10.0, 10.0, -5.0, -5.0, 20.0)},
    golden_output=(1.0, 0.0, 1.0, 0.0, 1.0),
    pins=(
        SpecPin(
            label="single_row",
            inputs={"quantity": (10.0,)},
            expected=(1.0,),
            reason="a one-element series charges the fee on the entry trade, not null",
        ),
        SpecPin(
            label="null_takes_precedence_over_nan",
            inputs={"quantity": (10.0, None, math.nan, 20.0)},
            expected=(1.0, None, None, math.nan),
            reason="the traded row where a NaN quantity meets the previous row's null yields null (null wins), while "
            "the next trade off the NaN is NaN",
        ),
        SpecPin(
            label="consecutive_infinities",
            inputs={"quantity": (math.inf, math.inf, 1.0, -math.inf)},
            expected=(1.0, math.nan, 1.0, 1.0),
            reason="two consecutive equal-sign infinities make the turnover inf - inf = NaN and the masked fee NaN; "
            "the property tiers set allow_infinity=False",
        ),
    ),
)
