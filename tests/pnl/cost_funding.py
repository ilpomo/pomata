"""Spec for ``pomata.pnl.cost_funding`` — the signed funding charge, an elementwise triple product, degree-1."""

import math

from tests.pnl.oracles import cost_funding_reference
from tests.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.pnl import cost_funding

COST_FUNDING = Spec(
    factory=cost_funding,
    inputs=("quantity", "price", "funding_rate"),
    params={},
    shape=Shape.SERIES,
    oracle=cost_funding_reference,
    # Degree-1 homogeneous in the position; the old suite tests the quantity axis as representative of the symmetric
    # product (tests/pnl/test_cost_funding.py::test_scale_homogeneity_in_quantity).
    scale=(ScaleAxis(roles=("quantity",), degree=1),),
    golden_input={
        "quantity": (10.0, 10.0, -5.0, -5.0, 20.0),
        "price": (100.0, 102.0, 101.0, 104.0, 103.0),
        "funding_rate": (0.0001, 0.0001, 0.0001, -0.0001, 0.0001),
    },
    golden_output=(0.1, 0.102, -0.0505, 0.052, 0.206),
    golden_round=6,
    pins=(
        SpecPin(
            label="single_row",
            inputs={"quantity": (10.0,), "price": (100.0,), "funding_rate": (0.0001,)},
            expected=(0.1,),
            reason="a one-row series resolves to the product 10 * 100 * 0.0001 = 0.1 (tests/pnl/test_cost_funding.py"
            "::test_single_row)",
        ),
        SpecPin(
            label="null_takes_precedence_over_nan",
            inputs={"quantity": (None, 10.0), "price": (math.nan, 100.0), "funding_rate": (0.0001, 0.0001)},
            expected=(None, 0.1),
            reason="a null in one column against a NaN in another of the same row yields null (null wins over NaN) "
            "(tests/pnl/test_cost_funding.py::test_null_takes_precedence_over_nan)",
        ),
        SpecPin(
            label="sign_follows_quantity_and_rate",
            inputs={
                "quantity": (10.0, -5.0, 10.0, -5.0),
                "price": (100.0, 100.0, 100.0, 100.0),
                "funding_rate": (0.0001, 0.0001, -0.0001, -0.0001),
            },
            expected=(0.1, -0.05, -0.1, 0.05),
            reason="the sign(quantity) * sign(funding_rate) convention over the full 2x2 matrix; the fuzz quantity is "
            "positive-only, so the short branch is otherwise untested (tests/pnl/test_cost_funding.py"
            "::test_sign_follows_quantity_and_rate)",
        ),
        SpecPin(
            label="zero_rate_is_free",
            inputs={
                "quantity": (10.0, -5.0, 20.0),
                "price": (100.0, 101.0, 102.0),
                "funding_rate": (0.0, 0.0, 0.0),
            },
            expected=(0.0, 0.0, 0.0),
            reason="an off-funding bar (funding_rate = 0) costs nothing (tests/pnl/test_cost_funding.py"
            "::test_zero_rate_is_free)",
        ),
    ),
)
