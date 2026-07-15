"""Spec for ``pomata.pnl.cost_borrow`` — the borrow charge on the short leg, elementwise, propagating, degree-1."""

import math

from tests.pnl.oracles import cost_borrow_reference
from tests.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.pnl import cost_borrow

COST_BORROW = Spec(
    factory=cost_borrow,
    inputs=("quantity", "price"),
    params={"rate": 0.0001},
    shape=Shape.SERIES,
    raises=(
        ({"rate": -0.0001}, r"rate must be a finite number >= 0"),
        ({"rate": math.nan}, r"rate must be a finite number >= 0"),
        ({"rate": math.inf}, r"rate must be a finite number >= 0"),
        ({"rate": -math.inf}, r"rate must be a finite number >= 0"),
    ),
    oracle=cost_borrow_reference,
    # Degree-1 homogeneous in the short notional; only the quantity axis is exercised.
    scale=(ScaleAxis(roles=("quantity",), degree=1),),
    golden_input={
        "quantity": (100.0, -50.0, -50.0, -20.0, -20.0),
        "price": (10.0, 11.0, 12.0, 13.0, 14.0),
    },
    golden_output=(0.0, 0.055, 0.06, 0.026, 0.028),
    golden_round=6,
    pins=(
        SpecPin(
            label="single_row",
            inputs={"quantity": (-50.0,), "price": (10.0,)},
            expected=(0.05,),
            reason="a one-row short series resolves to max(50, 0) * 10 * 0.0001 = 0.05",
        ),
        SpecPin(
            label="null_takes_precedence_over_nan",
            inputs={"quantity": (None, -50.0), "price": (math.nan, 10.0)},
            expected=(None, 0.05),
            reason="a null in one input against a NaN in the other yields null (null wins over NaN); the flow rungs "
            "poison every input with the same kind, never one null with one NaN",
        ),
        SpecPin(
            label="long_or_flat_is_zero",
            inputs={"quantity": (100.0, 0.0, 50.0), "price": (10.0, 11.0, 12.0)},
            expected=(0.0, 0.0, 0.0),
            reason="a long or flat quantity has zero borrow cost regardless of price",
        ),
        SpecPin(
            label="matches_reference_mixed_long_short",
            inputs={
                "quantity": (100.0, -50.0, -50.0, -20.0, -20.0, 0.0, -80.0, 40.0),
                "price": (10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0),
            },
            expected=(0.0, 0.055, 0.06, 0.026, 0.028, 0.0, 0.128, 0.0),
            reason="a hand-checked mixed long/short/flat series: the short branch max(-quantity, 0) charges exactly "
            "the short bars while longs and flats stay at exactly zero",
        ),
        SpecPin(
            label="infinite_short_notional_charges_inf",
            inputs={"quantity": (math.inf, -2.0, -math.inf), "price": (10.0, math.inf, 20.0)},
            expected=(0.0, math.inf, math.inf),
            reason="an infinite long is free to hold (only the short branch pays borrow) while an infinite short "
            "notional charges an infinite fee; the property tiers set allow_infinity=False",
        ),
    ),
)
