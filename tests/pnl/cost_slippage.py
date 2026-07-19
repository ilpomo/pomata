"""Declaration for ``pomata.pnl.cost_slippage`` — the half-spread crossing charge, turnover-scaled, propagating."""

import math

from pomata.pnl import cost_slippage
from tests.pnl.enums import BehaviorNan, BehaviorNull, ConventionSign, SpaceCost
from tests.pnl.harness import suite_pnl
from tests.pnl.oracles import reference_cost_slippage
from tests.support.declaration import Example, Golden, Pin, ScaleAxis

COST_SLIPPAGE = suite_pnl(
    factory=cost_slippage,
    inputs=("weight",),
    params={"half_spread": 0.002},
    null=BehaviorNull.PROPAGATES,
    nan=BehaviorNan.PROPAGATES,
    space=SpaceCost.RETURNS,
    sign=ConventionSign.LONG_SHORT,
    oracle=reference_cost_slippage,
    # Degree-1 homogeneous in the weight (it scales turnover by a fixed half-spread).
    scaling=(ScaleAxis(roles=("weight",), degree=1),),
    raises=(
        ({"half_spread": -0.002}, r"half_spread must be a finite number >= 0"),
        ({"half_spread": math.nan}, r"half_spread must be a finite number >= 0"),
        ({"half_spread": math.inf}, r"half_spread must be a finite number >= 0"),
        ({"half_spread": -math.inf}, r"half_spread must be a finite number >= 0"),
    ),
    golden=Golden(
        inputs={"weight": (0.5, 1.0, -0.5, -0.5, 0.0)},
        output=(0.001, 0.001, 0.003, 0.0, 0.001),
    ),
    pins=(
        Pin(
            label="single_row",
            inputs={"weight": (0.5,)},
            expected=(0.001,),
            reason="a one-element series resolves to |weight| * half_spread = 0.5 * 0.002 = 0.001 on the entry trade",
        ),
        Pin(
            label="null_takes_precedence_over_nan",
            inputs={"weight": (0.5, None, math.nan, 1.0)},
            expected=(0.001, None, None, math.nan),
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
    reference='Demsetz, H. (1968). "The Cost of Transacting." *The Quarterly Journal of Economics*, 82(1), 33-53.',
    doi="https://doi.org/10.2307/1882244",
    wikipedia="https://en.wikipedia.org/wiki/Slippage_%28finance%29",
    see_also=(
        ("cost_proportional", "The proportional broker fee, the complementary per-trade leg; sum the two for both."),
        ("turnover", "The traded fraction this scales."),
        ("returns_net", "Subtracts the composed cost from the gross return."),
    ),
    notes=(
        (
            "Flat start",
            "The weight before the series is taken as ``0`` (via :func:`turnover`), so the first row "
            "is ``|weight_0| * half_spread``: establishing the initial weight crosses the spread.",
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
    returns_body="The per-bar slippage cost for each row, the same length as ``weight``. The first row is "
    "``|weight_0| * half_spread`` (the cost of the entry trade from a flat start, per "
    ":func:`turnover`).",
    raises_prose="ValueError: If ``half_spread`` is not a finite number ``>= 0`` (i.e. ``< 0``, ``NaN``, or ``±inf``).",
    args_prose={
        "half_spread": "Fixed bid-ask half-spread crossed per trade, as a fraction (half the full spread; e.g. "
        "``0.002``). Must be a finite number ``>= 0``.",
    },
    examples=(
        Example(inputs={"weight": (0.5, 1.0, -0.5, -0.5, 0.0)}, params={"half_spread": 0.002}, round_to=4),
        Example(
            inputs={"weight": (0.5, 1.0, -0.5, 1.0, 1.0, 0.0)},
            intro="On a multi-ticker panel, wrap the call in ``.over`` so each ticker starts flat and never "
            "reaches across the boundary:",
            partition=("AAPL",) * 3 + ("NVDA",) * 3,
            params={"half_spread": 0.002},
            round_to=4,
        ),
        Example(
            inputs={"weight": (0.5, None, -0.5, float("nan"), 0.0)},
            intro="A ``null`` (which voids its own row and the next) and a ``NaN`` make the missing-data "
            "handling visible:",
            params={"half_spread": 0.002},
            round_to=4,
        ),
        Example(
            inputs={"weight": (float("inf"), float("inf"), 1.0, float("-inf"))},
            intro="**Non-finite input** — two consecutive same-sign infinite weights make the turnover "
            "``inf - inf`` at the second bar, so that bar's cost is ``NaN`` while the surrounding "
            "infinite trades still cost ``inf``:",
            params={"half_spread": 0.002},
        ),
    ),
)
