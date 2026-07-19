"""Declaration for ``pomata.pnl.cumulative_pnl`` — the additive running total, bridged nulls, latched NaNs, degree-1."""

import math

from pomata.pnl import cumulative_pnl
from tests.pnl.enums import BehaviorNan, BehaviorNull, ConventionSign, SpaceCost
from tests.pnl.harness import suite_pnl
from tests.pnl.oracles import reference_cumulative_pnl
from tests.support.declaration import Example, Golden, Pin, ScaleAxis

CUMULATIVE_PNL = suite_pnl(
    factory=cumulative_pnl,
    inputs=("returns",),
    params={},
    null=BehaviorNull.BRIDGED,
    nan=BehaviorNan.LATCHES,
    space=SpaceCost.CASH,
    sign=ConventionSign.LONG_SHORT,
    oracle=reference_cumulative_pnl,
    # The linear (additive) twin of equity_curve: degree-1 homogeneous, not scale-exempt.
    scaling=(ScaleAxis(roles=("returns",), degree=1),),
    golden=Golden(
        inputs={"returns": (0.1, -0.05, 0.2, 0.1)},
        output=(0.1, 0.05, 0.25, 0.35),
    ),
    pins=(
        Pin(
            label="single_row",
            inputs={"returns": (0.1,)},
            expected=(0.1,),
            reason="a one-element series resolves to that single return with no warm-up",
        ),
        Pin(
            label="warmup_leading_null",
            inputs={"returns": (None, 0.1, 0.2, -0.05)},
            expected=(None, 0.1, 0.3, 0.25),
            reason="a leading warm-up null stays null and the running total begins at the first defined return; the "
            "function declares no warm-up window, so no generic warm-up rung exercises a leading null",
        ),
        Pin(
            label="infinite_pnl_latches_nan_on_cancellation",
            inputs={"returns": (math.inf, 0.1, -math.inf, 0.2)},
            expected=(math.inf, math.inf, math.nan, math.nan),
            reason="the running total holds +inf until the opposite infinity cancels it to inf - inf = NaN, which "
            "then contaminates every later row; the property tiers set allow_infinity=False",
        ),
    ),
    wikipedia="https://en.wikipedia.org/wiki/Rate_of_return",
    see_also=(
        ("equity_curve", "The compounded (reinvested) return-flow cumulation, a product of one-plus-returns."),
        ("pnl_net", "The per-bar net P&L this typically cumulates in the cash flow."),
        ("returns_net", "The per-bar net return it cumulates for an additive, fixed-notional total."),
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
            "an ``inf`` return follows IEEE-754 through the arithmetic, and a canceling opposite "
            "infinity latches the running total to ``NaN`` (the sign, and any ``inf - inf = NaN``, "
            "included).",
        ),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="The running sum for each row, the same length as ``returns``.",
    args_prose={
        "returns": "Input per-bar values to cumulate — the strategy's net P&L (e.g. from :func:`pnl_net`) "
        "for a currency total, or a net-return series for an additive (fixed-notional) return "
        "total.",
    },
    intro_basic="Basic usage on a per-bar P&L series:",
    examples=(
        Example(inputs={"returns": (0.1, -0.05, 0.2, 0.1, -0.15, 0.05, 0.3, -0.1)}, round_to=4),
        Example(
            inputs={"returns": (0.1, 0.2, -0.05, 0.1, 0.0, 0.1, 0.1, -0.2)},
            intro="On a multi-ticker panel, wrap the call in ``.over`` so each ticker accumulates independently:",
            partition=("AAPL",) * 4 + ("NVDA",) * 4,
            round_to=4,
        ),
        Example(
            inputs={"returns": (0.1, None, 0.2, float("nan"), 0.1)},
            intro="A ``null`` (skipped, the running total carries across it) then a ``NaN`` (which "
            "contaminates every later row) in ``returns`` make the missing-data handling visible:",
            round_to=4,
        ),
        Example(
            inputs={"returns": (float("inf"), 0.1, float("-inf"), 0.2)},
            intro="**Non-finite input** — a ``+inf`` return latches the running total at ``+inf`` until an "
            "opposite-sign infinity cancels it to ``NaN``, which then persists for every later row:",
        ),
    ),
)
