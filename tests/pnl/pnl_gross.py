"""Spec for ``pomata.pnl.pnl_gross`` — the linear mark-to-market PnL, one-bar price lag, propagating, degree-1."""

import math

from tests.pnl.oracles import pnl_gross_reference
from tests.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.pnl import pnl_gross

PNL_GROSS = Spec(
    factory=pnl_gross,
    inputs=("quantity", "price"),
    params={"multiplier": 1.0},
    shape=Shape.SERIES,
    warmup=1,
    raises=(
        ({"multiplier": 0.0}, r"multiplier must be a finite number > 0"),
        ({"multiplier": -5.0}, r"multiplier must be a finite number > 0"),
        ({"multiplier": math.nan}, r"multiplier must be a finite number > 0"),
        ({"multiplier": math.inf}, r"multiplier must be a finite number > 0"),
        ({"multiplier": -math.inf}, r"multiplier must be a finite number > 0"),
    ),
    oracle=pnl_gross_reference,
    # Degree-1 homogeneous in the position and in the price (each scales the currency P&L linearly)
    scale=(
        ScaleAxis(roles=("quantity",), degree=1),
        ScaleAxis(roles=("price",), degree=1),
    ),
    golden_input={
        "quantity": (10.0, 10.0, -5.0, -5.0, 20.0),
        "price": (100.0, 102.0, 101.0, 104.0, 103.0),
    },
    golden_output=(None, 20.0, 5.0, -15.0, -20.0),
    pins=(
        SpecPin(
            label="null_precedence_null_quantity_nan_price",
            inputs={"quantity": (1.0, None, 2.0), "price": (100.0, math.nan, 110.0)},
            expected=(None, None, math.nan),
            reason="a null quantity against a NaN price yields null (null wins); the next bar reads the NaN previous "
            "price and propagates to NaN",
        ),
        SpecPin(
            label="null_precedence_nan_quantity_null_price",
            inputs={"quantity": (1.0, math.nan, 2.0), "price": (100.0, None, 110.0)},
            expected=(None, None, None),
            reason="the reverse precedence direction: a NaN quantity against a null price yields null, and the null "
            "previous price also nulls the next bar",
        ),
        SpecPin(
            label="short_on_flat_price_is_signed_zero",
            inputs={"quantity": (-5.0, -5.0), "price": (100.0, 100.0)},
            expected=(None, -0.0),
            reason="a short over a flat price yields IEEE -0.0 (a negative quantity times an exact +0.0 delta carries "
            "the sign bit); assert_matches reads -0.0 == 0.0",
            signed=True,
        ),
        SpecPin(
            label="consecutive_infinities_make_nan",
            inputs={"quantity": (10.0, 10.0, 10.0, 10.0), "price": (math.inf, math.inf, 1.0, -math.inf)},
            expected=(None, math.nan, -math.inf, -math.inf),
            reason="two consecutive equal-sign infinite prices make the second bar's price change inf - inf = NaN; the "
            "property tiers set allow_infinity=False",
        ),
        SpecPin(
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
