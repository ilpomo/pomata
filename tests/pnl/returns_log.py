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
)
