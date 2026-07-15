"""Spec for ``pomata.pnl.dividend`` — the position-times-dividend cash flow, elementwise, propagating, degree-1."""

import math

from tests.pnl.oracles import dividend_reference
from tests.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.pnl import dividend

DIVIDEND = Spec(
    factory=dividend,
    inputs=("quantity", "dividend_per_share"),
    params={},
    shape=Shape.SERIES,
    oracle=dividend_reference,
    # Degree-1 homogeneous in the position; only the quantity axis is exercised.
    scale=(ScaleAxis(roles=("quantity",), degree=1),),
    golden_input={
        "quantity": (100.0, 100.0, 100.0, 0.0, -50.0),
        "dividend_per_share": (0.0, 0.0, 0.5, 0.0, 0.5),
    },
    golden_output=(0.0, 0.0, 50.0, 0.0, -25.0),
    pins=(
        SpecPin(
            label="single_row",
            inputs={"quantity": (100.0,), "dividend_per_share": (0.5,)},
            expected=(50.0,),
            reason="a one-row series resolves to the single product 100 * 0.5 = 50",
        ),
        SpecPin(
            label="null_takes_precedence_over_nan",
            inputs={"quantity": (None, 100.0), "dividend_per_share": (math.nan, 0.5)},
            expected=(None, 50.0),
            reason="a null in one input against a NaN in the other at the same row yields null (null wins over NaN)",
        ),
        SpecPin(
            label="infinite_flow_signs_with_the_position",
            inputs={"quantity": (math.inf, -2.0, -math.inf), "dividend_per_share": (1.0, math.inf, 0.5)},
            expected=(math.inf, -math.inf, -math.inf),
            reason="the cash flow keeps the sign of quantity * dividend_per_share even at infinite magnitude; the "
            "property tiers set allow_infinity=False",
        ),
    ),
)
