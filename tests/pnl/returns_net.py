"""Declaration for ``pomata.pnl.returns_net`` — the gross-return-minus-cost difference, jointly degree-1."""

import math

from pomata.pnl import returns_net
from tests.pnl.enums import BehaviorNan, BehaviorNull, ConventionSign, SpaceCost
from tests.pnl.harness import suite_pnl
from tests.pnl.oracles import reference_returns_net
from tests.support.declaration import Golden, Pin, ScaleAxis

RETURNS_NET = suite_pnl(
    factory=returns_net,
    inputs=("returns_gross", "cost"),
    params={},
    null=BehaviorNull.PROPAGATES,
    nan=BehaviorNan.PROPAGATES,
    space=SpaceCost.RETURNS,
    sign=ConventionSign.LONG_SHORT,
    oracle=reference_returns_net,
    # The difference is degree-1 homogeneous when both inputs are scaled together.
    scaling=(ScaleAxis(roles=("returns_gross", "cost"), degree=1),),
    golden=Golden(
        inputs={
            "returns_gross": (0.05, -0.02, 0.03, 0.01, 0.0),
            "cost": (0.0005, 0.0015, 0.0005, 0.0, 0.0005),
        },
        output=(0.0495, -0.0215, 0.0295, 0.01, -0.0005),
    ),
    pins=(
        Pin(
            label="single_row",
            inputs={"returns_gross": (0.05,), "cost": (0.0005,)},
            expected=(0.0495,),
            reason="a one-row series resolves to the single difference 0.05 - 0.0005 = 0.0495",
        ),
        Pin(
            label="null_takes_precedence_over_nan",
            inputs={"returns_gross": (None, 0.05), "cost": (math.nan, 0.0005)},
            expected=(None, 0.0495),
            reason="a null in one input against a NaN in the other at the same row yields null (null wins over NaN)",
        ),
        Pin(
            label="consecutive_infinities_make_nan",
            inputs={"returns_gross": (math.inf, 0.05), "cost": (math.inf, 0.01)},
            expected=(math.nan, 0.04),
            reason="a same-sign infinite gross return and cost cancel to inf - inf = NaN; the property tiers set "
            "allow_infinity=False",
        ),
    ),
    wikipedia="https://en.wikipedia.org/wiki/Rate_of_return",
    see_also=(
        ("returns_gross", "The gross return this nets costs from."),
        ("cost_proportional", "The usual source of ``cost`` (a proportional, bps-of-notional fee)."),
        ("equity_curve", "Compounds these net returns into a capital curve."),
    ),
    bullets=(
        ("Null", "a ``null`` gross return makes that row ``null`` (``null`` takes precedence over ``NaN``)."),
        ("NaN", "a ``NaN`` gross return yields ``NaN`` for that row."),
        (
            "Non-finite input",
            "an ``inf`` gross return follows IEEE-754 through the arithmetic (the sign, and any ``inf "
            "- inf = NaN``, included).",
        ),
        (
            "Partitioning",
            "already correct on a multi-series panel: ``.over(...)`` partitions identically and is "
            "therefore optional here.",
        ),
    ),
    returns_body="The net strategy return for each row, the same length as the inputs. There is no window "
    "and no warm-up of its own: every row is ``returns_gross`` minus ``cost`` at that row.",
    args_prose={
        "returns_gross": "Gross per-bar strategy returns, typically from :func:`returns_gross`.",
        "cost": "Per-bar transaction cost as a return drag, typically from :func:`cost_proportional` (sum "
        "several with ``+``).",
    },
    intro_basic="Basic usage on a gross return and a cost series:",
    intro_over="The subtraction is elementwise, so ``.over`` partitions identically and is shown only for consistency:",
    intro_missing="A ``null`` then a ``NaN`` in ``returns_gross`` (both propagate through the subtraction) "
    "make the missing-data handling visible:",
)
