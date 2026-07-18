"""Declaration for ``pomata.pnl.pnl_gross_inverse`` — coin-settled inverse-contract PnL, one-bar lag, homogeneous."""

import math

from pomata.pnl import pnl_gross_inverse
from tests.pnl.enums import BehaviorNan, BehaviorNull, ConventionSign, SpaceCost, Warmup
from tests.pnl.harness import suite_pnl
from tests.pnl.oracles import reference_pnl_gross_inverse
from tests.support.declaration import Golden, Pin, ScaleAxis

PNL_GROSS_INVERSE = suite_pnl(
    factory=pnl_gross_inverse,
    inputs=("quantity", "price"),
    params={"multiplier": 1.0},
    null=BehaviorNull.PROPAGATES,
    nan=BehaviorNan.PROPAGATES,
    space=SpaceCost.CASH,
    sign=ConventionSign.LONG_SHORT,
    warmup=Warmup.ONE_ROW,
    oracle=reference_pnl_gross_inverse,
    # Degree-1 homogeneous in quantity, degree-(-1) homogeneous in price (the reciprocal payoff).
    scaling=(
        ScaleAxis(roles=("quantity",), degree=1),
        ScaleAxis(roles=("price",), degree=-1),
    ),
    raises=(
        ({"multiplier": 0.0}, r"multiplier must be a finite number > 0"),
        ({"multiplier": -5.0}, r"multiplier must be a finite number > 0"),
        ({"multiplier": math.nan}, r"multiplier must be a finite number > 0"),
        ({"multiplier": math.inf}, r"multiplier must be a finite number > 0"),
        ({"multiplier": -math.inf}, r"multiplier must be a finite number > 0"),
    ),
    golden=Golden(
        inputs={
            "quantity": (1.0, 1.0, -2.0, -2.0, 3.0),
            "price": (100.0, 110.0, 105.0, 120.0, 115.0),
        },
        output=(None, 0.000909, 0.000866, -0.002381, -0.001087),
        round_to=6,
    ),
    pins=(
        Pin(
            label="null_quantity_nan_price",
            inputs={"quantity": (1.0, None, 2.0), "price": (100.0, math.nan, 110.0)},
            expected=(None, None, math.nan),
            reason="a null quantity against a NaN price yields null (null wins); the NaN previous price then "
            "propagates to the next bar",
        ),
        Pin(
            label="nan_quantity_null_price",
            inputs={"quantity": (1.0, math.nan, 2.0), "price": (100.0, None, 110.0)},
            expected=(None, None, None),
            reason="the reverse direction: a NaN quantity against a null price yields null, and the null previous "
            "price also nulls the next bar",
        ),
        Pin(
            label="short_flat_price_signed_zero",
            inputs={"quantity": (-5.0, -5.0), "price": (100.0, 100.0)},
            expected=(None, -0.0),
            reason="a short over a flat price yields IEEE -0.0 (the reciprocal change is an exact +0.0, and a negative "
            "quantity carries the sign bit)",
            signed=True,
        ),
        Pin(
            label="domain_boundaries",
            inputs={"quantity": (1.0, 1.0, 1.0, 1.0), "price": (100.0, 0.0, 50.0, -50.0)},
            expected=(None, -math.inf, math.inf, 0.04),
            reason="the IEEE reciprocal boundaries pinned as data: a zero current price makes the bar -inf, a zero "
            "previous price makes the next bar +inf, a negative price stays finite",
        ),
        Pin(
            label="notional_multiplier_100",
            inputs={
                "quantity": (1.0, 1.0, -2.0, -2.0, 3.0),
                "price": (100.0, 110.0, 105.0, 120.0, 115.0),
            },
            expected=(
                None,
                0.09090909090909097,
                0.08658008658008684,
                -0.2380952380952383,
                -0.10869565217391311,
            ),
            reason="a 100x inverse-contract notional golden, at full precision (a pin has no "
            "rounding step), also subsuming the multiplier-scaling property",
            params_override={"multiplier": 100.0},
        ),
        Pin(
            label="infinite_legs_propagate",
            inputs={"quantity": (1.0, -math.inf, 2.0), "price": (math.inf, 20.0, 5.0)},
            expected=(None, math.inf, -0.30000000000000004),
            reason="an infinite entry price contributes 1 / inf = 0 to the inverse difference and an infinite short "
            "quantity drives the payoff to +inf, while the finite tail row is unaffected; the property tiers set "
            "allow_infinity=False",
        ),
    ),
)
