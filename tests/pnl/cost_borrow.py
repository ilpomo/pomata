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
    # Degree-1 homogeneous in the short notional; the old suite only exercises the quantity axis
    # (tests/pnl/test_cost_borrow.py::test_scale_homogeneity_in_quantity).
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
            reason="a one-row short series resolves to max(50, 0) * 10 * 0.0001 = 0.05 (tests/pnl/"
            "test_cost_borrow.py::test_single_row)",
        ),
        SpecPin(
            label="null_takes_precedence_over_nan",
            inputs={"quantity": (None, -50.0), "price": (math.nan, 10.0)},
            expected=(None, 0.05),
            reason="a null in one input against a NaN in the other yields null (null wins over NaN); the flow rungs "
            "poison every input with the same kind, never one null with one NaN (tests/pnl/test_cost_borrow.py"
            "::test_null_takes_precedence_over_nan)",
        ),
        SpecPin(
            label="long_or_flat_is_zero",
            inputs={"quantity": (100.0, 0.0, 50.0), "price": (10.0, 11.0, 12.0)},
            expected=(0.0, 0.0, 0.0),
            reason="a long or flat quantity has zero borrow cost regardless of price (tests/pnl/test_cost_borrow.py"
            "::test_long_or_flat_is_zero)",
        ),
        SpecPin(
            label="matches_reference_mixed_long_short",
            inputs={
                "quantity": (100.0, -50.0, -50.0, -20.0, -20.0, 0.0, -80.0, 40.0),
                "price": (10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0),
            },
            expected=(0.0, 0.055, 0.06, 0.026, 0.028, 0.0, 0.128, 0.0),
            reason="the short branch max(-quantity, 0) is never reached by the probe frame (quantity is always "
            "positive there), so its correctness is pinned on a mixed long/short/flat series (tests/pnl/"
            "test_cost_borrow.py::test_matches_reference)",
        ),
    ),
)
