"""Declaration for ``pomata.pnl.returns_log`` — the one-bar log return, propagating, scale-invariant, additive."""

import math

from pomata.pnl import returns_log
from tests.pnl.enums import BehaviorNan, BehaviorNull, ConventionSign, SpaceCost, Warmup
from tests.pnl.harness import suite_pnl
from tests.pnl.oracles import reference_returns_log
from tests.support.declaration import Golden, Pin, ScaleAxis

RETURNS_LOG = suite_pnl(
    factory=returns_log,
    inputs=("price",),
    params={},
    null=BehaviorNull.PROPAGATES,
    nan=BehaviorNan.PROPAGATES,
    space=SpaceCost.RETURNS,
    sign=ConventionSign.LONG_SHORT,
    warmup=Warmup.ONE_ROW,
    oracle=reference_returns_log,
    # The log of a price ratio: scaling the series cancels, leaving the return unchanged, degree 0.
    scaling=(ScaleAxis(roles=("price",), degree=0),),
    golden=Golden(
        inputs={"price": (100.0, 105.0, 102.0, 108.0, 110.0)},
        output=(None, 0.0488, -0.029, 0.0572, 0.0183),
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
            label="domain_boundaries",
            inputs={"price": (10.0, -5.0, 0.0, 5.0)},
            expected=(None, math.nan, -math.inf, math.inf),
            reason="the IEEE logarithm boundaries: a negative relative is NaN, a zero relative is -inf, a zero "
            "previous price makes the relative +inf",
        ),
        Pin(
            label="negative_zero_positive_change",
            inputs={"price": (-0.0, 5.0)},
            expected=(None, math.nan),
            reason="over a -0.0 previous price the relative flips sign, so a positive price gives a negative relative "
            "and NaN after the log",
        ),
        Pin(
            label="negative_zero_negative_change",
            inputs={"price": (-0.0, -5.0)},
            expected=(None, math.inf),
            reason="the counterpart: over a -0.0 previous price a negative price gives a positive relative, so +inf",
        ),
        Pin(
            label="consecutive_infinities",
            inputs={"price": (math.inf, math.inf, 1.0, -math.inf)},
            expected=(None, math.nan, -math.inf, math.nan),
            reason="two consecutive equal-sign infinite prices make the second log return log(inf / inf) = NaN; the "
            "property tiers set allow_infinity=False",
        ),
    ),
    reference='Meucci, A. (2010). "Quant Nugget 2: Linear vs. Compounded Returns." *GARP Risk '
    "Professional*, April 2010, 49-51.",
    wikipedia="https://en.wikipedia.org/wiki/Rate_of_return#Logarithmic_or_continuously_compounded_return",
    see_also=(
        ("returns_simple", "The arithmetic sibling, which aggregates across assets rather than across time."),
        ("cumulative_pnl", "The additive running total; log returns sum to the total log return over a horizon."),
        ("equity_curve", "Compounds the gross returns into the growth path of one unit of capital."),
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
            "Domain",
            "a negative price relative (the prices straddle zero) is outside the logarithm's "
            "strictly-positive domain, so the result is a loud ``NaN`` — except with both prices "
            "negative, where the ratio is positive and the log is silently finite, an economically "
            "meaningless number the caller must screen for.",
        ),
        (
            "Degenerate denominator",
            "both the price and the previous price zero give a ``0 / 0``, i.e. ``NaN`` (the logarithm "
            "then carries the ``NaN`` through); a zero price relative logs to ``-inf`` and a positive "
            "price over a zero previous price logs to ``+inf`` — reported, not clipped (a "
            "negative-zero ``-0.0`` previous price swaps which zero case applies but does not arise "
            "from real price data).",
        ),
        (
            "Non-finite input",
            "an ``inf`` price follows IEEE-754 through the ratio and its logarithm, where two "
            "consecutive same-sign infinite prices divide to ``inf / inf = NaN`` (the sign, and that "
            "indeterminate ``inf / inf``, included).",
        ),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="The log return for each row, the same length as ``expr``. The first value is ``null`` "
    "(warm-up) -- the lagged term ``expr.shift(1)`` is undefined for the first row, so no "
    "return can be measured there.",
    intro_over="On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up independently:",
    intro_missing="A ``null`` (whose lag voids the next bar too) and a ``NaN`` (which propagates) touch "
    "only the positions that reference them before the series recovers, making the "
    "missing-data handling visible:",
)
