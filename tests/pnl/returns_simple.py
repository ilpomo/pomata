"""Declaration for ``pomata.pnl.returns_simple`` — the one-bar arithmetic return, propagating, scale-invariant."""

import math

from pomata.pnl import returns_simple
from tests.pnl.enums import BehaviorNan, BehaviorNull, ConventionSign, SpaceCost, Warmup
from tests.pnl.harness import suite_pnl
from tests.pnl.oracles import reference_returns_simple
from tests.support.declaration import Example, Golden, Pin, ScaleAxis

RETURNS_SIMPLE = suite_pnl(
    factory=returns_simple,
    inputs=("price",),
    params={},
    null=BehaviorNull.PROPAGATES,
    nan=BehaviorNan.PROPAGATES,
    space=SpaceCost.RETURNS,
    sign=ConventionSign.LONG_SHORT,
    warmup=Warmup.ONE_ROW,
    oracle=reference_returns_simple,
    # A price ratio minus one: scaling the series leaves the return unchanged, degree 0.
    scaling=(ScaleAxis(roles=("price",), degree=0),),
    golden=Golden(
        inputs={"price": (100.0, 105.0, 102.0, 108.0, 110.0)},
        output=(None, 0.05, -0.0286, 0.0588, 0.0185),
    ),
    pins=(
        Pin(
            label="null_precedence",
            inputs={"price": (100.0, None, math.nan, 108.0)},
            expected=(None, None, None, math.nan),
            reason="the row where a NaN price meets the previous row's null is null (null wins), while the next return "
            "off the NaN is NaN",
        ),
        Pin(
            label="zero_previous_price",
            inputs={"price": (0.0, 10.0, 0.0, 0.0)},
            expected=(None, math.inf, -1.0, math.nan),
            reason="the IEEE division boundary: a non-zero change over a zero previous price is +/-inf, a zero change "
            "over zero is NaN; the fuzz draws strictly positive prices, so only a fixed case holds this boundary",
        ),
        Pin(
            label="negative_zero_positive_change",
            inputs={"price": (-0.0, 10.0)},
            expected=(None, -math.inf),
            reason="a -0.0 previous price flips the sign of the infinite return: a positive change over -0.0 is -inf",
        ),
        Pin(
            label="negative_zero_negative_change",
            inputs={"price": (-0.0, -10.0)},
            expected=(None, math.inf),
            reason="the counterpart: a negative change over -0.0 is +inf",
        ),
        Pin(
            label="consecutive_infinities",
            inputs={"price": (math.inf, math.inf, 1.0, -math.inf)},
            expected=(None, math.nan, -1.0, -math.inf),
            reason="two consecutive equal-sign infinite prices make the second return inf / inf - 1 = NaN; the "
            "property tiers set allow_infinity=False",
        ),
    ),
    reference='Meucci, A. (2010). "Quant Nugget 2: Linear vs. Compounded Returns." *GARP Risk '
    "Professional*, April 2010, 49-51.",
    wikipedia="https://en.wikipedia.org/wiki/Rate_of_return",
    see_also=(
        ("returns_log", "The logarithmic sibling, which aggregates across time rather than across assets."),
        ("equity_curve", "Compounds the simple returns into the growth path of one unit of capital."),
        ("cumulative_pnl", "The additive running total of a per-bar P&L or return series."),
    ),
    bullets=(
        (
            "Null",
            "a ``null`` price makes that row ``null`` (``null`` takes precedence over ``NaN``) — "
            "reading two endpoints, a ``null`` at the current or the previous row voids the output "
            "that references it.",
        ),
        (
            "NaN",
            "a ``NaN`` price yields ``NaN`` for that row — a fixed-lag transform of two endpoints, "
            "not a recurrence, so a ``NaN`` (like a ``null``) contaminates only the rows that "
            "reference it and never latches onto the rest of the series.",
        ),
        (
            "Degenerate denominator",
            "the previous price is ``0``, so a zero change is a ``0 / 0``, i.e. ``NaN`` — while a "
            "non-zero change over it is ``+/-inf`` (the sign tracks the change), reported, not "
            "clipped, and a negative-zero ``-0.0`` previous price flips that sign but does not arise "
            "from real price data.",
        ),
        (
            "Non-finite input",
            "an ``inf`` price follows IEEE-754 through the ratio and the minus one, where two "
            "consecutive same-sign infinite prices divide to ``inf / inf = NaN`` (the sign, and that "
            "indeterminate ``inf / inf``, included).",
        ),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="The simple return for each row, the same length as ``expr``. The first value is ``null`` "
    "(warm-up) -- the lagged term ``expr.shift(1)`` is undefined for the first row, so no "
    "return can be measured there.",
    example_columns={"price": "close"},
    examples=(
        Example(inputs={"price": (100.0, 102.0, 101.0, 105.0, 104.0, 107.0, 110.0, 108.0, 112.0)}, round_to=4),
        Example(
            inputs={"price": (100.0, 105.0, 102.0, 108.0, 50.0, 52.0, 51.0, 55.0)},
            intro="On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up independently:",
            partition=("AAPL",) * 4 + ("NVDA",) * 4,
            round_to=4,
        ),
        Example(
            inputs={"price": (100.0, 105.0, None, 108.0, 110.0, float("nan"), 113.0, 115.0)},
            intro="A ``null`` (whose lag voids the next bar too) and a ``NaN`` (which propagates) touch "
            "only the positions that reference them before the series recovers, making the "
            "missing-data handling visible:",
            round_to=4,
        ),
        Example(
            inputs={"price": (0.0, 10.0, 0.0, 0.0)},
            intro="**Degenerate denominator** — a nonzero price change over a zero previous price is "
            "``+inf`` or ``-inf`` (sign-tracking), while a zero change over a zero previous price is "
            "``NaN``:",
        ),
        Example(
            inputs={"price": (float("inf"), float("inf"), 1.0, float("-inf"))},
            intro="**Non-finite input** — two consecutive same-sign infinite prices divide to an "
            "indeterminate ``inf / inf``, so the second return is ``NaN``:",
        ),
    ),
)
