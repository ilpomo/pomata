"""Declaration for ``pomata.pnl.cost_fixed`` — the flat per-trade fee, elementwise, propagating, scale-invariant."""

import math

from pomata.pnl import cost_fixed
from tests.pnl.enums import BehaviorNan, BehaviorNull, ConventionSign, SpaceCost
from tests.pnl.harness import suite_pnl
from tests.pnl.oracles import reference_cost_fixed
from tests.support.declaration import Golden, Pin, ScaleAxis

COST_FIXED = suite_pnl(
    factory=cost_fixed,
    inputs=("quantity",),
    params={"fee": 1.0},
    null=BehaviorNull.PROPAGATES,
    nan=BehaviorNan.PROPAGATES,
    space=SpaceCost.CASH,
    sign=ConventionSign.LONG_SHORT,
    oracle=reference_cost_fixed,
    # Scaling the quantity by a positive constant leaves the trade-bar set unchanged, so the fee schedule is invariant,
    # degree 0.
    scaling=(ScaleAxis(roles=("quantity",), degree=0),),
    raises=(
        ({"fee": -1.0}, r"fee must be a finite number >= 0"),
        ({"fee": math.nan}, r"fee must be a finite number >= 0"),
        ({"fee": math.inf}, r"fee must be a finite number >= 0"),
        ({"fee": -math.inf}, r"fee must be a finite number >= 0"),
    ),
    golden=Golden(
        inputs={"quantity": (10.0, 10.0, -5.0, -5.0, 20.0)},
        output=(1.0, 0.0, 1.0, 0.0, 1.0),
    ),
    pins=(
        Pin(
            label="single_row",
            inputs={"quantity": (10.0,)},
            expected=(1.0,),
            reason="a one-element series charges the fee on the entry trade, not null",
        ),
        Pin(
            label="null_takes_precedence_over_nan",
            inputs={"quantity": (10.0, None, math.nan, 20.0)},
            expected=(1.0, None, None, math.nan),
            reason="the traded row where a NaN quantity meets the previous row's null yields null (null wins), while "
            "the next trade off the NaN is NaN",
        ),
        Pin(
            label="consecutive_infinities",
            inputs={"quantity": (math.inf, math.inf, 1.0, -math.inf)},
            expected=(1.0, math.nan, 1.0, 1.0),
            reason="two consecutive equal-sign infinities make the turnover inf - inf = NaN and the masked fee NaN; "
            "the property tiers set allow_infinity=False",
        ),
    ),
    wikipedia="https://en.wikipedia.org/wiki/Transaction_cost",
    see_also=(
        ("cost_per_share", "A per-unit-traded commission."),
        ("cost_notional", "A proportional (bps-of-notional) commission."),
        ("pnl_net", "Subtracts the composed cost from the gross PnL."),
    ),
    notes=(
        (
            "Flat start",
            "The pre-series quantity is taken as ``0`` (via :func:`turnover`), so the first row "
            "charges the ``fee`` (entering the initial position is a trade).",
        ),
    ),
    bullets=(
        ("Null", "a ``null`` quantity makes that row ``null`` (``null`` takes precedence over ``NaN``)."),
        ("NaN", "a ``NaN`` quantity yields ``NaN`` for that row."),
        (
            "Non-finite input",
            "an ``inf`` quantity follows IEEE-754 through the arithmetic of the turnover difference, "
            "whose ``inf`` marks a trade and charges the flat ``fee`` (the sign, and any ``inf - inf "
            "= NaN``, included).",
        ),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="The per-bar fixed cost for each row, the same length as ``quantity`` -- ``fee`` where "
    "the quantity changes (the first row counts as a trade from a flat start) and ``0`` where "
    "it is held.",
    raises_prose="ValueError: If ``fee`` is not a finite number ``>= 0`` (i.e. ``< 0``, ``NaN``, or ``±inf``).",
    args_prose={
        "quantity": "Signed position size in units / shares / contracts held over the bar (e.g. ``100``, ``-2``).",
        "fee": "Flat charge per trade, in the account currency. Must be a finite number ``>= 0``.",
    },
    intro_over="On a multi-ticker panel, wrap the call in ``.over`` so each ticker starts flat:",
    intro_missing="A ``null`` (which voids its own row and the next) and a ``NaN`` make the missing-data "
    "handling visible:",
)
