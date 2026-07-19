"""Declaration for ``pomata.pnl.equity_curve`` — compounding cumulation, bridged nulls, latched NaNs, scale-exempt."""

import math

from pomata.pnl import equity_curve
from tests.pnl.enums import BehaviorNan, BehaviorNull, ConventionSign, SpaceCost
from tests.pnl.harness import suite_pnl
from tests.pnl.oracles import reference_equity_curve
from tests.support.declaration import Example, Golden, Pin, ScaleExempt

EQUITY_CURVE = suite_pnl(
    factory=equity_curve,
    inputs=("returns",),
    params={},
    null=BehaviorNull.BRIDGED,
    nan=BehaviorNan.LATCHES,
    space=SpaceCost.RETURNS,
    sign=ConventionSign.LONG_SHORT,
    oracle=reference_equity_curve,
    # A nonlinear compounding transform — neither scale-invariant nor homogeneous.
    scaling=ScaleExempt(reason="nonlinear compounding: neither scale-invariant nor homogeneous"),
    golden=Golden(
        inputs={"returns": (0.1, -0.05, 0.2, 0.1)},
        output=(1.1, 1.045, 1.254, 1.3794),
    ),
    pins=(
        Pin(
            label="single_row",
            inputs={"returns": (0.1,)},
            expected=(1.1,),
            reason="a one-element series resolves to 1 + return with no warm-up of its own",
        ),
        Pin(
            label="leading_null_passthrough",
            inputs={"returns": (None, 0.1, 0.2, -0.05)},
            expected=(None, 1.1, 1.32, 1.254),
            reason="a leading warm-up null stays null and the compounded curve begins at the first defined return; "
            "the function declares no warm-up window, so no generic warm-up rung exercises a leading null",
        ),
        Pin(
            label="infinite_return_flips_the_curve_sign",
            inputs={"returns": (math.inf, 0.1, -math.inf, 0.2)},
            expected=(math.inf, math.inf, -math.inf, -math.inf),
            reason="a +inf return inflates the compounded curve to +inf and a later -inf factor flips it to -inf, "
            "which then persists; the property tiers set allow_infinity=False",
        ),
    ),
    wikipedia="https://en.wikipedia.org/wiki/Rate_of_return",
    see_also=(
        ("cumulative_pnl", "The additive (fixed-notional) twin, a cumulative sum of returns."),
        ("returns_gross", "The per-bar strategy returns this typically compounds."),
        ("drawdown", "The metric that reads this equity curve, its decline from the running peak."),
    ),
    bullets=(
        (
            "Null",
            "a leading ``null`` run stays ``null`` until the first non-null seed; an interior "
            "``null`` yields ``null`` at that position while the recursion continues across the gap.",
        ),
        (
            "NaN",
            "a ``NaN`` contaminates the recursive state and yields ``NaN`` for every subsequent non-null position.",
        ),
        (
            "Non-finite input",
            "an ``inf`` return follows IEEE-754 through the arithmetic, where a later opposite-sign "
            "infinite factor flips the running product's sign (the sign included).",
        ),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="The compounded equity for each row, the same length as ``returns``, expressed as a "
    "growth factor relative to a starting capital of ``1`` (multiply by the starting capital "
    "for a currency curve).",
    args_prose={
        "returns": "Input per-bar returns to compound, typically the strategy's gross or net returns (e.g. "
        "from :func:`returns_gross`).",
    },
    intro_basic="Basic usage on a per-bar return series:",
    examples=(
        Example(inputs={"returns": (0.1, -0.05, 0.2, 0.1, -0.15, 0.05, 0.3, -0.1)}, round_to=4),
        Example(
            inputs={"returns": (0.1, 0.2, -0.05, 0.1, 0.0, 0.1, 0.1, -0.2)},
            intro="On a multi-ticker panel, wrap the call in ``.over`` so each ticker compounds independently:",
            partition=("A", "A", "A", "A", "B", "B", "B", "B"),
            round_to=4,
        ),
        Example(
            inputs={"returns": (None, 0.1, 0.2, float("nan"), 0.1)},
            intro="A leading ``null`` stays ``null`` (the curve begins at the first defined return) and a "
            "later ``NaN`` then contaminates every row after it:",
            round_to=4,
        ),
        Example(
            inputs={"returns": (float("inf"), 0.1, float("-inf"), 0.2)},
            intro="**Non-finite input** — a ``+inf`` return inflates the compounded curve to ``+inf``, and "
            "a later ``-inf`` factor flips its sign to ``-inf``, which then persists:",
        ),
    ),
)
