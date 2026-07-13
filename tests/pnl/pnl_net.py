"""Spec for ``pomata.pnl.pnl_net`` — the gross-minus-cost difference, elementwise, propagating, jointly degree-1."""

import math

from tests.pnl.oracles import pnl_net_reference
from tests.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.pnl import pnl_net

PNL_NET = Spec(
    factory=pnl_net,
    inputs=("pnl_gross", "cost"),
    params={},
    shape=Shape.SERIES,
    oracle=pnl_net_reference,
    # The difference is degree-1 homogeneous when both inputs are scaled together (tests/pnl/test_pnl_net.py
    # ::test_scale_homogeneity).
    scale=(ScaleAxis(roles=("pnl_gross", "cost"), degree=1),),
    golden_input={
        "pnl_gross": (20.0, 5.0, -15.0, -20.0, 8.0),
        "cost": (2.0, 0.0, 3.0, 0.0, 1.0),
    },
    golden_output=(18.0, 5.0, -18.0, -20.0, 7.0),
    pins=(
        SpecPin(
            label="single_row",
            inputs={"pnl_gross": (20.0,), "cost": (2.0,)},
            expected=(18.0,),
            reason="a one-row series resolves to the single difference 20 - 2 = 18 (tests/pnl/test_pnl_net.py"
            "::test_single_row)",
        ),
        SpecPin(
            label="null_takes_precedence_over_nan",
            inputs={"pnl_gross": (None, 20.0), "cost": (math.nan, 2.0)},
            expected=(None, 18.0),
            reason="a null in one input against a NaN in the other at the same row yields null (null wins over NaN) "
            "(tests/pnl/test_pnl_net.py::test_null_takes_precedence_over_nan)",
        ),
        SpecPin(
            label="consecutive_infinities_make_nan",
            inputs={"pnl_gross": (math.inf, 5.0), "cost": (math.inf, 1.0)},
            expected=(math.nan, 4.0),
            reason="a same-sign infinite gross and cost cancel to inf - inf = NaN; the property tiers set "
            "allow_infinity=False (tests/pnl/test_pnl_net.py::test_consecutive_infinities_make_nan)",
        ),
    ),
)
