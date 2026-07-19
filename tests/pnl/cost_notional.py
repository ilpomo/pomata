"""Declaration for ``pomata.pnl.cost_notional`` — the bps-of-traded-notional fee, turnover-scaled, propagating."""

import math

from pomata.pnl import cost_notional
from tests.pnl.enums import BehaviorNan, BehaviorNull, ConventionSign, SpaceCost
from tests.pnl.harness import suite_pnl
from tests.pnl.oracles import reference_cost_notional
from tests.support.declaration import Example, Golden, Pin, ScaleAxis

COST_NOTIONAL = suite_pnl(
    factory=cost_notional,
    inputs=("quantity", "price"),
    params={"rate": 0.001},
    null=BehaviorNull.PROPAGATES,
    nan=BehaviorNan.PROPAGATES,
    space=SpaceCost.CASH,
    sign=ConventionSign.LONG_SHORT,
    oracle=reference_cost_notional,
    # Degree-1 homogeneous in the position and in the price (each scales the traded notional linearly).
    scaling=(
        ScaleAxis(roles=("quantity",), degree=1),
        ScaleAxis(roles=("price",), degree=1),
    ),
    raises=(
        ({"rate": -0.001}, r"rate must be a finite number >= 0"),
        ({"rate": math.nan}, r"rate must be a finite number >= 0"),
        ({"rate": math.inf}, r"rate must be a finite number >= 0"),
        ({"rate": -math.inf}, r"rate must be a finite number >= 0"),
    ),
    golden=Golden(
        inputs={
            "quantity": (10.0, 10.0, -5.0, -5.0, 20.0),
            "price": (100.0, 102.0, 101.0, 104.0, 103.0),
        },
        output=(1.0, 0.0, 1.515, 0.0, 2.575),
    ),
    pins=(
        Pin(
            label="single_row",
            inputs={"quantity": (10.0,), "price": (100.0,)},
            expected=(1.0,),
            reason="the first row charges on the entry trade |quantity| * price * rate = 10 * 100 * 0.001 = 1.0",
        ),
        Pin(
            label="null_takes_precedence_over_nan",
            inputs={"quantity": (10.0, None), "price": (100.0, math.nan)},
            expected=(1.0, None),
            reason="a null in quantity against a NaN in price at the same row yields null (null wins)",
        ),
        Pin(
            label="consecutive_infinities_make_nan",
            inputs={"quantity": (math.inf, math.inf, 1.0, -math.inf), "price": (100.0, 100.0, 100.0, 100.0)},
            expected=(math.inf, math.nan, math.inf, math.inf),
            reason="two consecutive equal-sign infinite quantities make inf - inf = NaN turnover at the second bar; "
            "the property tiers set allow_infinity=False",
        ),
    ),
    wikipedia="https://en.wikipedia.org/wiki/Transaction_cost",
    see_also=(
        ("cost_per_share", "A per-unit-traded commission."),
        ("cost_fixed", "A flat charge per trade."),
        ("pnl_net", "Subtracts the composed cost from the gross PnL."),
    ),
    notes=(
        (
            "Flat start",
            "The pre-series quantity is taken as ``0`` (via :func:`turnover`), so the first row "
            "charges on the entry trade.",
        ),
    ),
    bullets=(
        ("Null", "a ``null`` quantity makes that row ``null`` (``null`` takes precedence over ``NaN``)."),
        ("NaN", "a ``NaN`` quantity yields ``NaN`` for that row."),
        (
            "Non-finite input",
            "an ``inf`` quantity follows IEEE-754 through the arithmetic of the turnover difference, "
            "an infinite move charging an ``inf`` cost; a held bar at an infinite ``price`` is ``0 * "
            "inf``, i.e. ``NaN`` (the sign, and any ``inf - inf = NaN``, included).",
        ),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="The per-bar notional cost for each row, the same length as the inputs. The first row "
    "charges on ``|quantity_0| * price_0`` (the entry trade from a flat start).",
    raises_prose="ValueError: If ``rate`` is not a finite number ``>= 0`` (i.e. ``< 0``, ``NaN``, or ``±inf``).",
    args_prose={
        "quantity": "Signed position size in units / shares / contracts held over the bar (e.g. ``100``, ``-2``).",
        "price": 'Instrument price series (e.g. ``pl.col("close")``); must share a length and alignment '
        "with ``quantity``.",
        "rate": "Proportional cost rate, the fee as a fraction of traded notional (e.g. ``0.001`` = 10 "
        "bps). Must be a finite number ``>= 0``.",
    },
    examples=(
        Example(
            inputs={"quantity": (10.0, 10.0, -5.0, -5.0, 20.0), "price": (100.0, 102.0, 101.0, 104.0, 103.0)},
            params={"rate": 0.001},
            round_to=4,
        ),
        Example(
            inputs={"quantity": (10.0, 10.0, -5.0, 2.0, 2.0, 2.0), "price": (100.0, 102.0, 101.0, 50.0, 51.0, 49.0)},
            intro="On a multi-ticker panel, wrap the call in ``.over`` so each ticker starts flat:",
            partition=("AAPL",) * 3 + ("NVDA",) * 3,
            params={"rate": 0.001},
            round_to=4,
        ),
        Example(
            inputs={
                "quantity": (10.0, None, -5.0, float("nan"), 20.0),
                "price": (100.0, 102.0, 101.0, 104.0, float("nan")),
            },
            intro="A ``null`` (which voids the rows that reference it) and a ``NaN`` make the missing-data "
            "handling visible:",
            params={"rate": 0.001},
            round_to=4,
        ),
        Example(
            inputs={
                "quantity": (float("inf"), float("inf"), 1.0, float("-inf")),
                "price": (100.0, 100.0, 100.0, 100.0),
            },
            intro="**Non-finite input** — two consecutive same-sign infinite quantities make the turnover "
            "``inf - inf`` at the second bar, so that bar's cost is ``NaN`` while the surrounding "
            "infinite trades still cost ``inf``:",
            params={"rate": 0.001},
        ),
    ),
)
