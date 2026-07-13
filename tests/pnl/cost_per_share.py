"""Spec for ``pomata.pnl.cost_per_share`` — the per-unit commission, turnover-scaled, propagating, degree-1."""

import math

from tests.pnl.oracles import cost_per_share_reference
from tests.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.pnl import cost_per_share

COST_PER_SHARE = Spec(
    factory=cost_per_share,
    inputs=("quantity",),
    params={"fee": 0.01},
    shape=Shape.SERIES,
    raises=(
        ({"fee": -0.01}, r"fee must be a finite number >= 0"),
        ({"fee": math.nan}, r"fee must be a finite number >= 0"),
        ({"fee": math.inf}, r"fee must be a finite number >= 0"),
        ({"fee": -math.inf}, r"fee must be a finite number >= 0"),
    ),
    oracle=cost_per_share_reference,
    # Degree-1 homogeneous in quantity (it scales turnover by a fixed fee) (tests/pnl/test_cost_per_share.py
    # ::test_scale_homogeneity).
    scale=(ScaleAxis(roles=("quantity",), degree=1),),
    golden_input={"quantity": (10.0, 10.0, -5.0, -5.0, 20.0)},
    golden_output=(0.1, 0.0, 0.15, 0.0, 0.25),
    pins=(
        SpecPin(
            label="single_row",
            inputs={"quantity": (10.0,)},
            expected=(0.1,),
            reason="a one-element series resolves to |quantity| * fee = 10 * 0.01 = 0.1 on the entry trade "
            "(tests/pnl/test_cost_per_share.py::test_single_row)",
        ),
        SpecPin(
            label="null_takes_precedence_over_nan",
            inputs={"quantity": (10.0, None, math.nan, 20.0)},
            expected=(0.1, None, None, math.nan),
            reason="the traded row where a NaN quantity meets the previous row's null yields null (null wins), while "
            "the next trade off the NaN is NaN (tests/pnl/test_cost_per_share.py::test_null_takes_precedence_over_nan)",
        ),
        SpecPin(
            label="consecutive_infinities",
            inputs={"quantity": (math.inf, math.inf, 1.0, -math.inf)},
            expected=(math.inf, math.nan, math.inf, math.inf),
            reason="two consecutive equal-sign infinities make inf - inf = NaN turnover at the second bar; the "
            "property tiers set allow_infinity=False (tests/pnl/test_cost_per_share.py"
            "::test_consecutive_infinities_make_nan)",
        ),
    ),
)
