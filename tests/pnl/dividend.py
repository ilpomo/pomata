"""Declaration for ``pomata.pnl.dividend`` — the position-times-dividend cash flow, elementwise, propagating."""

import math

from pomata.pnl import dividend
from tests.pnl.enums import BehaviorNan, BehaviorNull, ConventionSign, SpaceCost
from tests.pnl.harness import suite_pnl
from tests.pnl.oracles import reference_dividend
from tests.support.declaration import Golden, Pin, ScaleAxis

DIVIDEND = suite_pnl(
    factory=dividend,
    inputs=("quantity", "dividend_per_share"),
    params={},
    null=BehaviorNull.PROPAGATES,
    nan=BehaviorNan.PROPAGATES,
    space=SpaceCost.CASH,
    sign=ConventionSign.LONG_SHORT,
    oracle=reference_dividend,
    # Degree-1 homogeneous in the position; only the quantity axis is exercised.
    scaling=(ScaleAxis(roles=("quantity",), degree=1),),
    golden=Golden(
        inputs={
            "quantity": (100.0, 100.0, 100.0, 0.0, -50.0),
            "dividend_per_share": (0.0, 0.0, 0.5, 0.0, 0.5),
        },
        output=(0.0, 0.0, 50.0, 0.0, -25.0),
    ),
    pins=(
        Pin(
            label="single_row",
            inputs={"quantity": (100.0,), "dividend_per_share": (0.5,)},
            expected=(50.0,),
            reason="a one-row series resolves to the single product 100 * 0.5 = 50",
        ),
        Pin(
            label="null_takes_precedence_over_nan",
            inputs={"quantity": (None, 100.0), "dividend_per_share": (math.nan, 0.5)},
            expected=(None, 50.0),
            reason="a null in one input against a NaN in the other at the same row yields null (null wins over NaN)",
        ),
        Pin(
            label="infinite_flow_signs_with_the_position",
            inputs={"quantity": (math.inf, -2.0, -math.inf), "dividend_per_share": (1.0, math.inf, 0.5)},
            expected=(math.inf, -math.inf, -math.inf),
            reason="the cash flow keeps the sign of quantity * dividend_per_share even at infinite magnitude; the "
            "property tiers set allow_infinity=False",
        ),
    ),
    wikipedia="https://en.wikipedia.org/wiki/Dividend",
    see_also=(
        ("pnl_gross", "The gross position PnL this dividend income is added to."),
        ("cost_borrow", "The equity holding cashflow on the cost side (short-borrow)."),
        ("cost_funding", "The perpetual-swap funding leg, another per-bar holding cashflow."),
    ),
    bullets=(
        ("Null", "a ``null`` quantity makes that row ``null`` (``null`` takes precedence over ``NaN``)."),
        ("NaN", "a ``NaN`` quantity yields ``NaN`` for that row."),
        (
            "Non-finite input",
            "an ``inf`` quantity follows IEEE-754 through the arithmetic, so the flow signs with "
            "``quantity * dividend_per_share`` (the sign included).",
        ),
        (
            "Partitioning",
            "already correct on a multi-series panel: ``.over(...)`` partitions identically and is "
            "therefore optional here.",
        ),
    ),
    returns_body="The dividend cashflow for each row, the same length as the inputs.",
    args_prose={
        "quantity": "Signed position size in units / shares / contracts held over the bar; a long (positive) "
        "receives the dividend, a short (negative) pays it.",
        "dividend_per_share": 'Dividend paid per share for the bar (e.g. ``pl.col("dividend")``); zero on ordinary '
        "bars.",
    },
    intro_basic="Basic usage on a held quantity and a per-share dividend:",
    intro_over="The product is elementwise, so ``.over`` partitions identically and is shown only for consistency:",
    intro_missing="A ``null`` then a ``NaN`` in ``quantity`` (both propagate through the product) make the "
    "missing-data handling visible:",
)
