"""Declaration for ``pomata.pnl.turnover`` — the absolute one-bar weight change, propagating, degree-1."""

import math

from pomata.pnl import turnover
from tests.pnl.enums import BehaviorNan, BehaviorNull, ConventionSign, SpaceCost
from tests.pnl.harness import suite_pnl
from tests.pnl.oracles import reference_turnover
from tests.support.declaration import Example, Golden, Pin, ScaleAxis

TURNOVER = suite_pnl(
    factory=turnover,
    inputs=("weight",),
    params={},
    null=BehaviorNull.PROPAGATES,
    nan=BehaviorNan.PROPAGATES,
    space=SpaceCost.RETURNS,
    sign=ConventionSign.LONG_SHORT,
    oracle=reference_turnover,
    # The absolute weight change |w_t - w_{t-1}| is degree-1 homogeneous.
    scaling=(ScaleAxis(roles=("weight",), degree=1),),
    golden=Golden(
        inputs={"weight": (0.5, 1.0, -0.5, -0.5, 0.0)},
        output=(0.5, 0.5, 1.5, 0.0, 0.5),
    ),
    pins=(
        Pin(
            label="single_row",
            inputs={"weight": (0.7,)},
            expected=(0.7,),
            reason="a one-element series resolves to |weight_0| = 0.7 (the entry trade off a flat start), not null",
        ),
        Pin(
            label="null_takes_precedence_over_nan",
            inputs={"weight": (0.5, None, math.nan, 1.0)},
            expected=(0.5, None, None, math.nan),
            reason="the difference row where a NaN weight meets the previous row's null yields null (null wins), while "
            "the next difference off the NaN is NaN",
        ),
        Pin(
            label="consecutive_infinities_make_nan",
            inputs={"weight": (math.inf, math.inf, 1.0, -math.inf)},
            expected=(math.inf, math.nan, math.inf, math.inf),
            reason="a single inf carries |inf| forward, while two consecutive equal-sign infinities make inf - inf = "
            "NaN; the property tiers set allow_infinity=False",
        ),
    ),
    reference="Grinold, R. C. & Kahn, R. N. (2000). *Active Portfolio Management: A Quantitative "
    "Approach for Producing Superior Returns and Controlling Risk* (2nd ed.). McGraw-Hill.",
    see_also=(
        ("cost_proportional", "The proportional transaction cost this turnover scales."),
        ("cost_slippage", "A per-trade slippage cost also driven by the traded fraction."),
        ("returns_gross", "The gross return of the same ``weight``."),
    ),
    notes=(
        (
            "Flat start",
            "The weight before the series is taken as ``0``, so the first row is ``|weight_0|`` "
            "rather than ``null``; establishing the initial weight from cash is a real trade and "
            "carries its cost.",
        ),
    ),
    bullets=(
        (
            "Null",
            "a ``null`` weight makes that row ``null`` (``null`` takes precedence over ``NaN``) — "
            "and, via the one-bar difference, the next row too, then turnover resumes.",
        ),
        ("NaN", "a ``NaN`` weight yields ``NaN`` for that row — and the next, via the one-bar difference."),
        (
            "Non-finite input",
            "an ``inf`` weight follows IEEE-754 through the arithmetic, where a single infinite "
            "``weight`` carries ``|inf| = inf`` forward (the sign, and any ``inf - inf = NaN``, "
            "included).",
        ),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="The traded fraction for each row, the same length as ``weight``. The first row is "
    "``|weight_0|`` (the trade from a flat start), not ``null``.",
    intro_basic="Basic usage on a weight series:",
    examples=(
        Example(inputs={"weight": (0.5, 1.0, -0.5, -0.5, 0.0, 1.0, 1.0, -1.0)}, round_to=4),
        Example(
            inputs={"weight": (0.5, 1.0, -0.5, -0.5, 1.0, 1.0, 0.0, 0.5)},
            intro="On a multi-ticker panel, wrap the call in ``.over`` so each ticker starts flat and never "
            "differences across the boundary:",
            partition=("AAPL",) * 4 + ("NVDA",) * 4,
            round_to=4,
        ),
        Example(
            inputs={"weight": (0.5, None, -0.5, float("nan"), 0.0)},
            intro="A ``null`` (which voids its own row and the next, since the difference references the "
            "previous weight) then a ``NaN`` (likewise) make the missing-data handling visible:",
            round_to=4,
        ),
        Example(
            inputs={"weight": (float("inf"), float("inf"), 1.0, float("-inf"))},
            intro="**Non-finite input** — a single infinite weight carries ``|inf|`` forward, while two "
            "consecutive equal-sign infinite weights difference to ``inf - inf``, so that bar's "
            "turnover is ``NaN``:",
        ),
    ),
)
