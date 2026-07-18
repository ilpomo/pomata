"""Declaration for ``pomata.pnl.cost_funding`` — the signed funding charge, an elementwise triple product, degree-1."""

import math

from pomata.pnl import cost_funding
from tests.pnl.enums import BehaviorNan, BehaviorNull, ConventionSign, SpaceCost
from tests.pnl.harness import suite_pnl
from tests.pnl.oracles import reference_cost_funding
from tests.support.declaration import Golden, Pin, ScaleAxis

COST_FUNDING = suite_pnl(
    factory=cost_funding,
    inputs=("quantity", "price", "funding_rate"),
    params={},
    null=BehaviorNull.PROPAGATES,
    nan=BehaviorNan.PROPAGATES,
    space=SpaceCost.CASH,
    sign=ConventionSign.LONG_SHORT,
    oracle=reference_cost_funding,
    # Degree-1 homogeneous in the position; the quantity axis stands in for the symmetric product.
    scaling=(ScaleAxis(roles=("quantity",), degree=1),),
    golden=Golden(
        inputs={
            "quantity": (10.0, 10.0, -5.0, -5.0, 20.0),
            "price": (100.0, 102.0, 101.0, 104.0, 103.0),
            "funding_rate": (0.0001, 0.0001, 0.0001, -0.0001, 0.0001),
        },
        output=(0.1, 0.102, -0.0505, 0.052, 0.206),
        round_to=6,
    ),
    pins=(
        Pin(
            label="single_row",
            inputs={"quantity": (10.0,), "price": (100.0,), "funding_rate": (0.0001,)},
            expected=(0.1,),
            reason="a one-row series resolves to the product 10 * 100 * 0.0001 = 0.1",
        ),
        Pin(
            label="null_takes_precedence_over_nan",
            inputs={"quantity": (None, 10.0), "price": (math.nan, 100.0), "funding_rate": (0.0001, 0.0001)},
            expected=(None, 0.1),
            reason="a null in one column against a NaN in another of the same row yields null (null wins over NaN)",
        ),
        Pin(
            label="sign_follows_quantity_and_rate",
            inputs={
                "quantity": (10.0, -5.0, 10.0, -5.0),
                "price": (100.0, 100.0, 100.0, 100.0),
                "funding_rate": (0.0001, 0.0001, -0.0001, -0.0001),
            },
            expected=(0.1, -0.05, -0.1, 0.05),
            reason="the sign(quantity) * sign(funding_rate) convention pinned over the full 2x2 sign matrix on "
            "hand-checked values",
        ),
        Pin(
            label="zero_rate_is_free",
            inputs={
                "quantity": (10.0, -5.0, 20.0),
                "price": (100.0, 101.0, 102.0),
                "funding_rate": (0.0, 0.0, 0.0),
            },
            expected=(0.0, 0.0, 0.0),
            reason="an off-funding bar (funding_rate = 0) costs nothing",
        ),
        Pin(
            label="infinite_notional_signs_the_carry",
            inputs={
                "quantity": (math.inf, -2.0, -math.inf),
                "price": (10.0, math.inf, 20.0),
                "funding_rate": (0.01, 0.02, -0.01),
            },
            expected=(math.inf, -math.inf, math.inf),
            reason="the carry keeps the sign of quantity * price * rate even at infinite magnitude; the property "
            "tiers set allow_infinity=False",
        ),
    ),
)
