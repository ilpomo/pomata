"""Declaration for ``pomata.pnl.pnl_gross`` — the linear mark-to-market PnL, one-bar lag, propagating, degree-1."""

import math

from pomata.pnl import pnl_gross
from tests.pnl.enums import BehaviorNan, BehaviorNull, ConventionSign, SpaceCost, Warmup
from tests.pnl.harness import suite_pnl
from tests.pnl.oracles import reference_pnl_gross
from tests.support.declaration import Golden, Pin, ScaleAxis

PNL_GROSS = suite_pnl(
    factory=pnl_gross,
    inputs=("quantity", "price"),
    params={"multiplier": 1.0},
    null=BehaviorNull.PROPAGATES,
    nan=BehaviorNan.PROPAGATES,
    space=SpaceCost.CASH,
    sign=ConventionSign.LONG_SHORT,
    warmup=Warmup.ONE_ROW,
    oracle=reference_pnl_gross,
    # Degree-1 homogeneous in the position and in the price (each scales the currency P&L linearly).
    scaling=(
        ScaleAxis(roles=("quantity",), degree=1),
        ScaleAxis(roles=("price",), degree=1),
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
            "quantity": (10.0, 10.0, -5.0, -5.0, 20.0),
            "price": (100.0, 102.0, 101.0, 104.0, 103.0),
        },
        output=(None, 20.0, 5.0, -15.0, -20.0),
    ),
    pins=(
        Pin(
            label="null_precedence_null_quantity_nan_price",
            inputs={"quantity": (1.0, None, 2.0), "price": (100.0, math.nan, 110.0)},
            expected=(None, None, math.nan),
            reason="a null quantity against a NaN price yields null (null wins); the next bar reads the NaN previous "
            "price and propagates to NaN",
        ),
        Pin(
            label="null_precedence_nan_quantity_null_price",
            inputs={"quantity": (1.0, math.nan, 2.0), "price": (100.0, None, 110.0)},
            expected=(None, None, None),
            reason="the reverse precedence direction: a NaN quantity against a null price yields null, and the null "
            "previous price also nulls the next bar",
        ),
        Pin(
            label="short_on_flat_price_is_signed_zero",
            inputs={"quantity": (-5.0, -5.0), "price": (100.0, 100.0)},
            expected=(None, -0.0),
            reason="a short over a flat price yields IEEE -0.0 (a negative quantity times an exact +0.0 delta carries "
            "the sign bit); assert_matches reads -0.0 == 0.0",
            signed=True,
        ),
        Pin(
            label="consecutive_infinities_make_nan",
            inputs={"quantity": (10.0, 10.0, 10.0, 10.0), "price": (math.inf, math.inf, 1.0, -math.inf)},
            expected=(None, math.nan, -math.inf, -math.inf),
            reason="two consecutive equal-sign infinite prices make the second bar's price change inf - inf = NaN; the "
            "property tiers set allow_infinity=False",
        ),
        Pin(
            label="multiplier_50x_golden",
            inputs={
                "quantity": (10.0, 10.0, -5.0, -5.0, 20.0),
                "price": (100.0, 102.0, 101.0, 104.0, 103.0),
            },
            expected=(None, 1000.0, 250.0, -750.0, -1000.0),
            reason="a 50x futures-multiplier golden, exercising the exact multiplier "
            "arithmetic a single canonical golden cannot",
            params_override={"multiplier": 50.0},
        ),
    ),
)
