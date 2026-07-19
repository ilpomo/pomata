"""Declaration for ``pomata.pnl.cost_funding`` — the signed funding charge, an elementwise triple product, degree-1."""

import math

from pomata.pnl import cost_funding
from tests.pnl.enums import BehaviorNan, BehaviorNull, ConventionSign, SpaceCost
from tests.pnl.harness import suite_pnl
from tests.pnl.oracles import reference_cost_funding
from tests.support.declaration import Example, Golden, Pin, ScaleAxis

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
    reference='Shiller, R. J. (1993). "Measuring Asset Values for Cash Settlement in Derivative '
    'Markets: Hedonic Repeated Measures Indices and Perpetual Futures." *The Journal of '
    "Finance*, 48(3), 911-931.",
    doi="https://doi.org/10.1111/j.1540-6261.1993.tb04024.x",
    wikipedia="https://en.wikipedia.org/wiki/Perpetual_futures",
    see_also=(
        ("cost_borrow", "The short-borrow holding cost on the equity side."),
        ("cost_notional", "The maker/taker fee on each perpetual-swap trade."),
        ("pnl_net", "Subtracts the composed cost from the gross PnL."),
    ),
    notes=(
        (
            "Sign convention",
            "The cost follows ``sign(quantity) * sign(funding_rate)``: a long pays a positive rate "
            "and is rebated by a negative one; a short is the mirror image.",
        ),
        ("Off-funding bars", "Pass ``funding_rate = 0`` on bars with no funding event; the cost is then ``0`` there."),
    ),
    bullets=(
        ("Null", "a ``null`` quantity makes that row ``null`` (``null`` takes precedence over ``NaN``)."),
        ("NaN", "a ``NaN`` quantity yields ``NaN`` for that row."),
        (
            "Non-finite input",
            "an ``inf`` quantity follows IEEE-754 through the arithmetic, the signed triple product "
            "``quantity * price * funding_rate``; a flat bar at an infinite ``price`` is ``0 * inf``, "
            "i.e. ``NaN`` (the sign included).",
        ),
        (
            "Partitioning",
            "already correct on a multi-series panel: ``.over(...)`` partitions identically and is "
            "therefore optional here.",
        ),
    ),
    returns_body="The per-bar funding cost for each row, the same length as the inputs -- positive where "
    "the holder pays and negative (a rebate) where the holder receives.",
    args_prose={
        "quantity": "Signed position size in units / shares / contracts held over the bar (e.g. ``100``, ``-2``).",
        "price": 'Instrument price series (e.g. ``pl.col("close")``); must share a length and alignment '
        "with ``quantity``.",
        "funding_rate": "Per-bar funding rate as a signed fraction of notional, supplied as a series so it can be "
        "``0`` on the bars between funding events (e.g. ``0.0001`` = 1 bp); a positive rate "
        "charges longs and rebates shorts.",
    },
    examples=(
        Example(
            inputs={
                "quantity": (10.0, 10.0, -5.0, -5.0, 20.0),
                "price": (100.0, 102.0, 101.0, 104.0, 103.0),
                "funding_rate": (0.0001, 0.0001, 0.0001, -0.0001, 0.0001),
            },
            round_to=6,
        ),
        Example(
            inputs={
                "quantity": (10.0, 10.0, -5.0, 2.0, 2.0, -3.0),
                "price": (100.0, 102.0, 101.0, 50.0, 51.0, 49.0),
                "funding_rate": (0.0001, 0.0001, 0.0001, 0.0001, -0.0001, 0.0001),
            },
            intro="On a multi-ticker panel, partition with ``.over`` — for this elementwise holding cost it "
            "is optional (the result is identical without it) and shown here only for consistency:",
            partition=("A", "A", "A", "B", "B", "B"),
            round_to=6,
        ),
        Example(
            inputs={
                "quantity": (10.0, None, -5.0, float("nan"), 20.0),
                "price": (100.0, 102.0, 101.0, 104.0, 103.0),
                "funding_rate": (0.0001, 0.0001, 0.0001, 0.0001, 0.0001),
            },
            intro="A ``null`` (which propagates) and a ``NaN`` make the missing-data handling visible:",
            round_to=6,
        ),
        Example(
            inputs={
                "quantity": (float("inf"), -2.0, float("-inf")),
                "price": (10.0, float("inf"), 20.0),
                "funding_rate": (0.01, 0.02, -0.01),
            },
            intro="**Non-finite input** — the signed triple product keeps its sign at infinite magnitude, "
            "so the funding cost is ``inf`` or ``-inf`` following the sign of ``quantity * price * "
            "funding_rate``:",
        ),
    ),
)
