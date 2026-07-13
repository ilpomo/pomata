"""Spec for ``pomata.pnl.cost_slippage`` — the half-spread crossing charge, turnover-scaled, propagating, degree-1."""

import math

from tests_new.pnl.oracles import cost_slippage_reference
from tests_new.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.pnl import cost_slippage

COST_SLIPPAGE = Spec(
    factory=cost_slippage,
    inputs=("weight",),
    params={"half_spread": 0.002},
    shape=Shape.SERIES,
    raises=(
        ({"half_spread": -0.002}, r"half_spread must be a finite number >= 0"),
        ({"half_spread": math.nan}, r"half_spread must be a finite number >= 0"),
        ({"half_spread": math.inf}, r"half_spread must be a finite number >= 0"),
        ({"half_spread": -math.inf}, r"half_spread must be a finite number >= 0"),
    ),
    oracle=cost_slippage_reference,
    # Degree-1 homogeneous in the weight (it scales turnover by a fixed half-spread) (tests/pnl/test_cost_slippage.py
    # ::test_scale_homogeneity).
    scale=(ScaleAxis(roles=("weight",), degree=1),),
    golden_input={"weight": (0.5, 1.0, -0.5, -0.5, 0.0)},
    golden_output=(0.001, 0.001, 0.003, 0.0, 0.001),
    pins=(
        SpecPin(
            label="single_row",
            inputs={"weight": (0.5,)},
            expected=(0.001,),
            reason="a one-element series resolves to |weight| * half_spread = 0.5 * 0.002 = 0.001 on the entry trade "
            "(tests/pnl/test_cost_slippage.py::test_single_row)",
        ),
        SpecPin(
            label="null_takes_precedence_over_nan",
            inputs={"weight": (0.5, None, math.nan, 1.0)},
            expected=(0.001, None, None, math.nan),
            reason="the turnover row where a NaN weight meets the previous row's null yields null (null wins), while "
            "the next turnover off the NaN is NaN (tests/pnl/test_cost_slippage.py"
            "::test_null_takes_precedence_over_nan)",
        ),
        SpecPin(
            label="consecutive_infinities_make_nan",
            inputs={"weight": (math.inf, math.inf, 1.0, -math.inf)},
            expected=(math.inf, math.nan, math.inf, math.inf),
            reason="two consecutive equal-sign infinities make inf - inf = NaN turnover at the second bar; the "
            "property tiers set allow_infinity=False (tests/pnl/test_cost_slippage.py"
            "::test_consecutive_infinities_make_nan)",
        ),
    ),
)
