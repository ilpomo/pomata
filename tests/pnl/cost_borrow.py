"""Declaration for ``pomata.pnl.cost_borrow`` — the borrow charge on the short leg, elementwise, propagating."""

import math

from pomata.pnl import cost_borrow
from tests.pnl.enums import BehaviorNan, BehaviorNull, ConventionSign, SpaceCost
from tests.pnl.harness import suite_pnl
from tests.pnl.oracles import reference_cost_borrow
from tests.support.declaration import Example, Golden, Pin, ScaleAxis

COST_BORROW = suite_pnl(
    factory=cost_borrow,
    inputs=("quantity", "price"),
    params={"rate": 0.0001},
    null=BehaviorNull.PROPAGATES,
    nan=BehaviorNan.PROPAGATES,
    space=SpaceCost.CASH,
    sign=ConventionSign.SHORT_ONLY,
    oracle=reference_cost_borrow,
    # Degree-1 homogeneous in the short notional; only the quantity axis is exercised.
    scaling=(ScaleAxis(roles=("quantity",), degree=1),),
    raises=(
        ({"rate": -0.0001}, r"rate must be a finite number >= 0"),
        ({"rate": math.nan}, r"rate must be a finite number >= 0"),
        ({"rate": math.inf}, r"rate must be a finite number >= 0"),
        ({"rate": -math.inf}, r"rate must be a finite number >= 0"),
    ),
    golden=Golden(
        inputs={
            "quantity": (100.0, -50.0, -50.0, -20.0, -20.0),
            "price": (10.0, 11.0, 12.0, 13.0, 14.0),
        },
        output=(0.0, 0.055, 0.06, 0.026, 0.028),
        round_to=6,
    ),
    pins=(
        Pin(
            label="single_row",
            inputs={"quantity": (-50.0,), "price": (10.0,)},
            expected=(0.05,),
            reason="a one-row short series resolves to max(50, 0) * 10 * 0.0001 = 0.05",
        ),
        Pin(
            label="null_takes_precedence_over_nan",
            inputs={"quantity": (None, -50.0), "price": (math.nan, 10.0)},
            expected=(None, 0.05),
            reason="a null in one input against a NaN in the other yields null (null wins over NaN); the flow rungs "
            "poison every input with the same kind, never one null with one NaN",
        ),
        Pin(
            label="long_or_flat_is_zero",
            inputs={"quantity": (100.0, 0.0, 50.0), "price": (10.0, 11.0, 12.0)},
            expected=(0.0, 0.0, 0.0),
            reason="a long or flat quantity has zero borrow cost regardless of price",
        ),
        Pin(
            label="matches_reference_mixed_long_short",
            inputs={
                "quantity": (100.0, -50.0, -50.0, -20.0, -20.0, 0.0, -80.0, 40.0),
                "price": (10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0),
            },
            expected=(0.0, 0.055, 0.06, 0.026, 0.028, 0.0, 0.128, 0.0),
            reason="a hand-checked mixed long/short/flat series: the short branch max(-quantity, 0) charges exactly "
            "the short bars while longs and flats stay at exactly zero",
        ),
        Pin(
            label="infinite_short_notional_charges_inf",
            inputs={"quantity": (math.inf, -2.0, -math.inf), "price": (10.0, math.inf, 20.0)},
            expected=(0.0, math.inf, math.inf),
            reason="an infinite long is free to hold (only the short branch pays borrow) while an infinite short "
            "notional charges an infinite fee; the property tiers set allow_infinity=False",
        ),
    ),
    reference='D\'Avolio, G. (2002). "The market for borrowing stock." *Journal of Financial Economics*, '
    "66(2-3), 271-306.",
    doi="https://doi.org/10.1016/S0304-405X(02)00206-4",
    wikipedia="https://en.wikipedia.org/wiki/Securities_lending",
    see_also=(
        ("dividend", "The equity holding cashflow on the income side."),
        ("cost_funding", "The perpetual-swap holding cost."),
        ("pnl_net", "Subtracts the composed cost from the gross PnL."),
    ),
    notes=(("Long / flat", "A non-negative quantity has zero borrow cost (only the short part is charged)."),),
    bullets=(
        ("Null", "a ``null`` quantity makes that row ``null`` (``null`` takes precedence over ``NaN``)."),
        ("NaN", "a ``NaN`` quantity yields ``NaN`` for that row."),
        (
            "Non-finite input",
            "an ``inf`` quantity follows IEEE-754 through the arithmetic, where the short-only clip "
            "frees an infinite long (``0``) and an infinite short notional charges an ``inf`` fee; a "
            "flat or long bar at an infinite ``price`` is ``0 * inf``, i.e. ``NaN`` (the sign "
            "included).",
        ),
        (
            "Partitioning",
            "already correct on a multi-series panel: ``.over(...)`` partitions identically and is "
            "therefore optional here.",
        ),
    ),
    returns_body="The per-bar borrow cost for each row, the same length as the inputs -- a non-negative "
    "cost on short bars (for a non-negative price; a negative price yields an economically "
    "meaningless negative value) and ``0`` on long or flat bars.",
    raises_prose="ValueError: If ``rate`` is not a finite number ``>= 0`` (i.e. ``< 0``, ``NaN``, or ``±inf``).",
    args_prose={
        "quantity": "Signed position size in units / shares / contracts held over the bar; only the short "
        "part (``q < 0``) is charged.",
        "price": 'Instrument price series (e.g. ``pl.col("close")``); must share a length and alignment '
        "with ``quantity``.",
        "rate": "Per-bar borrow rate, as a fraction of the short notional (e.g. an annual rate divided by "
        "the bars per year). Must be a finite number ``>= 0``.",
    },
    examples=(
        Example(
            inputs={"quantity": (100.0, -50.0, -50.0, -20.0, -20.0), "price": (10.0, 11.0, 12.0, 13.0, 14.0)},
            params={"rate": 0.0001},
            round_to=6,
        ),
        Example(
            inputs={
                "quantity": (100.0, -50.0, -50.0, -20.0, -20.0, 30.0),
                "price": (10.0, 11.0, 12.0, 13.0, 14.0, 15.0),
            },
            intro="On a multi-ticker panel, partition with ``.over`` — for this elementwise holding cost it "
            "is optional (the result is identical without it) and shown here only for consistency:",
            partition=("A", "A", "A", "B", "B", "B"),
            params={"rate": 0.0001},
            round_to=6,
        ),
        Example(
            inputs={
                "quantity": (-50.0, None, -50.0, float("nan"), -20.0),
                "price": (10.0, 11.0, float("nan"), 12.0, 13.0),
            },
            intro="A ``null`` (which propagates) and a ``NaN`` make the missing-data handling visible:",
            params={"rate": 0.0001},
            round_to=6,
        ),
        Example(
            inputs={"quantity": (float("inf"), -2.0, float("-inf")), "price": (10.0, float("inf"), 20.0)},
            intro="**Non-finite input** — an infinite long holds for free (only the short side pays borrow) "
            "while an infinite short notional charges an infinite fee, so the cost is ``0`` or "
            "``inf``:",
            params={"rate": 0.0001},
        ),
    ),
)
