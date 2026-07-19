"""Declaration for ``pomata.pnl.cost_per_share`` — the per-unit commission, turnover-scaled, propagating, degree-1."""

import math

from pomata.pnl import cost_per_share
from tests.pnl.enums import BehaviorNan, BehaviorNull, ConventionSign, SpaceCost
from tests.pnl.harness import suite_pnl
from tests.pnl.oracles import reference_cost_per_share
from tests.support.declaration import Example, Golden, Pin, ScaleAxis

COST_PER_SHARE = suite_pnl(
    factory=cost_per_share,
    inputs=("quantity",),
    params={"fee": 0.01},
    null=BehaviorNull.PROPAGATES,
    nan=BehaviorNan.PROPAGATES,
    space=SpaceCost.CASH,
    sign=ConventionSign.LONG_SHORT,
    oracle=reference_cost_per_share,
    # Degree-1 homogeneous in quantity (it scales turnover by a fixed fee).
    scaling=(ScaleAxis(roles=("quantity",), degree=1),),
    raises=(
        ({"fee": -0.01}, r"fee must be a finite number >= 0"),
        ({"fee": math.nan}, r"fee must be a finite number >= 0"),
        ({"fee": math.inf}, r"fee must be a finite number >= 0"),
        ({"fee": -math.inf}, r"fee must be a finite number >= 0"),
    ),
    golden=Golden(
        inputs={"quantity": (10.0, 10.0, -5.0, -5.0, 20.0)},
        output=(0.1, 0.0, 0.15, 0.0, 0.25),
    ),
    pins=(
        Pin(
            label="single_row",
            inputs={"quantity": (10.0,)},
            expected=(0.1,),
            reason="a one-element series resolves to |quantity| * fee = 10 * 0.01 = 0.1 on the entry trade",
        ),
        Pin(
            label="null_takes_precedence_over_nan",
            inputs={"quantity": (10.0, None, math.nan, 20.0)},
            expected=(0.1, None, None, math.nan),
            reason="the traded row where a NaN quantity meets the previous row's null yields null (null wins), while "
            "the next trade off the NaN is NaN",
        ),
        Pin(
            label="consecutive_infinities",
            inputs={"quantity": (math.inf, math.inf, 1.0, -math.inf)},
            expected=(math.inf, math.nan, math.inf, math.inf),
            reason="two consecutive equal-sign infinities make inf - inf = NaN turnover at the second bar; the "
            "property tiers set allow_infinity=False, so only this pin reaches the branch",
        ),
    ),
    wikipedia="https://en.wikipedia.org/wiki/Transaction_cost",
    see_also=(
        ("cost_fixed", "A flat charge per trade."),
        ("cost_notional", "A proportional (bps-of-notional) commission."),
        ("pnl_net", "Subtracts the composed cost from the gross PnL."),
    ),
    notes=(
        (
            "Flat start",
            "The pre-series quantity is taken as ``0`` (via :func:`turnover`), so the first row "
            "charges on ``|quantity_0|`` (entering the initial position is a trade).",
        ),
    ),
    bullets=(
        ("Null", "a ``null`` quantity makes that row ``null`` (``null`` takes precedence over ``NaN``)."),
        ("NaN", "a ``NaN`` quantity yields ``NaN`` for that row."),
        (
            "Non-finite input",
            "an ``inf`` quantity follows IEEE-754 through the arithmetic of the turnover difference, "
            "an infinite move charging an ``inf`` cost (the sign, and any ``inf - inf = NaN``, "
            "included).",
        ),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="The per-bar per-share cost for each row, the same length as ``quantity``. The first row "
    "charges on ``|quantity_0|`` (the entry trade from a flat start).",
    raises_prose="ValueError: If ``fee`` is not a finite number ``>= 0`` (i.e. ``< 0``, ``NaN``, or ``±inf``).",
    args_prose={
        "quantity": "Signed position size in units / shares / contracts held over the bar (e.g. ``100``, ``-2``).",
        "fee": "Commission per unit traded, in the account currency (e.g. ``0.01`` = one cent per "
        "share). Must be a finite number ``>= 0``.",
    },
    examples=(
        Example(inputs={"quantity": (10.0, 10.0, -5.0, -5.0, 20.0)}, params={"fee": 0.01}, round_to=4),
        Example(
            inputs={"quantity": (10.0, 10.0, -5.0, 2.0, 2.0, 2.0)},
            intro="On a multi-ticker panel, wrap the call in ``.over`` so each ticker starts flat:",
            partition=("A", "A", "A", "B", "B", "B"),
            params={"fee": 0.01},
            round_to=4,
        ),
        Example(
            inputs={"quantity": (10.0, None, -5.0, float("nan"), 20.0)},
            intro="A ``null`` (which voids its own row and the next) and a ``NaN`` make the missing-data "
            "handling visible:",
            params={"fee": 0.01},
            round_to=4,
        ),
        Example(
            inputs={"quantity": (float("inf"), float("inf"), 1.0, float("-inf"))},
            intro="**Non-finite input** — two consecutive same-sign infinite quantities make the turnover "
            "``inf - inf`` at the second bar, so that bar's cost is ``NaN`` while the surrounding "
            "infinite trades still cost ``inf``:",
            params={"fee": 0.01},
        ),
    ),
)
