"""Spec for ``pomata.pnl.cost_notional`` — the bps-of-traded-notional fee, turnover-scaled, propagating, degree-1."""

import math

from tests.pnl.oracles import cost_notional_reference
from tests_new.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.pnl import cost_notional

COST_NOTIONAL = Spec(
    factory=cost_notional,
    inputs=("quantity", "price"),
    params={"rate": 0.001},
    shape=Shape.SERIES,
    raises=(
        ({"rate": -0.001}, r"rate must be a finite number >= 0"),
        ({"rate": math.nan}, r"rate must be a finite number >= 0"),
        ({"rate": math.inf}, r"rate must be a finite number >= 0"),
        ({"rate": -math.inf}, r"rate must be a finite number >= 0"),
    ),
    oracle=cost_notional_reference,
    # Degree-1 homogeneous in quantity; the old suite exercises only the quantity axis (tests/pnl/
    # test_cost_notional.py::test_scale_homogeneity_in_quantity).
    scale=(ScaleAxis(roles=("quantity",), degree=1),),
    golden_input={
        "quantity": (10.0, 10.0, -5.0, -5.0, 20.0),
        "price": (100.0, 102.0, 101.0, 104.0, 103.0),
    },
    golden_output=(1.0, 0.0, 1.515, 0.0, 2.575),
    pins=(
        SpecPin(
            label="single_row",
            inputs={"quantity": (10.0,), "price": (100.0,)},
            expected=(1.0,),
            reason="the first row charges on the entry trade |quantity| * price * rate = 10 * 100 * 0.001 = 1.0 "
            "(tests/pnl/test_cost_notional.py::test_single_row)",
        ),
        SpecPin(
            label="null_takes_precedence_over_nan",
            inputs={"quantity": (10.0, None), "price": (100.0, math.nan)},
            expected=(1.0, None),
            reason="a null in quantity against a NaN in price at the same row yields null (null wins) (tests/pnl/"
            "test_cost_notional.py::test_null_takes_precedence_over_nan)",
        ),
        SpecPin(
            label="consecutive_infinities_make_nan",
            inputs={"quantity": (math.inf, math.inf, 1.0, -math.inf), "price": (100.0, 100.0, 100.0, 100.0)},
            expected=(math.inf, math.nan, math.inf, math.inf),
            reason="two consecutive equal-sign infinite quantities make inf - inf = NaN turnover at the second bar; "
            "the property tiers set allow_infinity=False (tests/pnl/test_cost_notional.py"
            "::test_consecutive_infinities_make_nan)",
        ),
    ),
)
