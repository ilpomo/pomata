"""Declaration for ``pomata.pnl.cost_proportional`` — the bps-of-weight-turnover fee, propagating, degree-1."""

import math

from pomata.pnl import cost_proportional
from tests.pnl.enums import BehaviorNan, BehaviorNull, ConventionSign, SpaceCost
from tests.pnl.harness import suite_pnl
from tests.pnl.oracles import reference_cost_proportional
from tests.support.declaration import Example, Golden, Pin, ScaleAxis

COST_PROPORTIONAL = suite_pnl(
    factory=cost_proportional,
    inputs=("weight",),
    params={"rate": 0.001},
    null=BehaviorNull.PROPAGATES,
    nan=BehaviorNan.PROPAGATES,
    space=SpaceCost.RETURNS,
    sign=ConventionSign.LONG_SHORT,
    oracle=reference_cost_proportional,
    # Degree-1 homogeneous in the weight (it scales turnover by a fixed rate).
    scaling=(ScaleAxis(roles=("weight",), degree=1),),
    raises=(
        ({"rate": -0.001}, r"rate must be a finite number >= 0"),
        ({"rate": math.nan}, r"rate must be a finite number >= 0"),
        ({"rate": math.inf}, r"rate must be a finite number >= 0"),
        ({"rate": -math.inf}, r"rate must be a finite number >= 0"),
    ),
    golden=Golden(
        inputs={"weight": (0.5, 1.0, -0.5, -0.5, 0.0)},
        output=(0.0005, 0.0005, 0.0015, 0.0, 0.0005),
    ),
    pins=(
        Pin(
            label="single_row",
            inputs={"weight": (0.5,)},
            expected=(0.0005,),
            reason="a one-element series resolves to |weight| * rate = 0.5 * 0.001 = 0.0005 on the entry trade",
        ),
        Pin(
            label="null_takes_precedence_over_nan",
            inputs={"weight": (0.5, None, math.nan, 1.0)},
            expected=(0.0005, None, None, math.nan),
            reason="the turnover row where a NaN weight meets the previous row's null yields null (null wins), while "
            "the next turnover off the NaN is NaN",
        ),
        Pin(
            label="consecutive_infinities_make_nan",
            inputs={"weight": (math.inf, math.inf, 1.0, -math.inf)},
            expected=(math.inf, math.nan, math.inf, math.inf),
            reason="two consecutive equal-sign infinities make inf - inf = NaN turnover at the second bar; the "
            "property tiers set allow_infinity=False",
        ),
    ),
    reference='Magill, M. J. P. & Constantinides, G. M. (1976). "Portfolio selection with transactions '
    'costs." *Journal of Economic Theory*, 13(2), 245-263.',
    doi="https://doi.org/10.1016/0022-0531(76)90018-1",
    wikipedia="https://en.wikipedia.org/wiki/Transaction_cost",
    see_also=(
        ("cost_slippage", "The fixed half-spread cost, the complementary per-trade leg; sum the two for both."),
        ("turnover", "The traded fraction this scales."),
        ("returns_net", "Subtracts the composed cost from the gross return."),
    ),
    notes=(
        (
            "Flat start",
            "The weight before the series is taken as ``0`` (via :func:`turnover`), so the first row "
            "is ``|weight_0| * rate``: establishing the initial weight carries its cost.",
        ),
    ),
    bullets=(
        ("Null", "a ``null`` weight makes that row ``null`` (``null`` takes precedence over ``NaN``)."),
        ("NaN", "a ``NaN`` weight yields ``NaN`` for that row."),
        (
            "Non-finite input",
            "an ``inf`` weight follows IEEE-754 through the arithmetic of the turnover difference, an "
            "infinite move charging an ``inf`` cost (the sign, and any ``inf - inf = NaN``, "
            "included).",
        ),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="The per-bar proportional cost for each row, the same length as ``weight``. The first row "
    "is ``|weight_0| * rate`` (the cost of the entry trade from a flat start, per "
    ":func:`turnover`).",
    raises_prose="ValueError: If ``rate`` is not a finite number ``>= 0`` (i.e. ``< 0``, ``NaN``, or ``±inf``).",
    args_prose={
        "rate": "Proportional cost rate, the fee as a fraction of traded notional (e.g. ``0.001`` = 10 "
        "bps). Must be a finite number ``>= 0``.",
    },
    examples=(
        Example(inputs={"weight": (0.5, 1.0, -0.5, -0.5, 0.0)}, params={"rate": 0.001}, round_to=4),
        Example(
            inputs={"weight": (0.5, 1.0, -0.5, 1.0, 1.0, 0.0)},
            intro="On a multi-ticker panel, wrap the call in ``.over`` so each ticker starts flat and never "
            "reaches across the boundary:",
            partition=("A", "A", "A", "B", "B", "B"),
            params={"rate": 0.001},
            round_to=4,
        ),
        Example(
            inputs={"weight": (0.5, None, -0.5, float("nan"), 0.0)},
            intro="A ``null`` (which voids its own row and the next) and a ``NaN`` make the missing-data "
            "handling visible:",
            params={"rate": 0.001},
            round_to=4,
        ),
        Example(
            inputs={"weight": (float("inf"), float("inf"), 1.0, float("-inf"))},
            intro="**Non-finite input** — two consecutive same-sign infinite weights make the turnover "
            "``inf - inf`` at the second bar, so that bar's cost is ``NaN`` while the surrounding "
            "infinite trades still cost ``inf``:",
            params={"rate": 0.001},
        ),
    ),
)
