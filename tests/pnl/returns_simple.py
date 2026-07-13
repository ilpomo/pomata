"""Spec for ``pomata.pnl.returns_simple`` — the one-bar arithmetic return, propagating, scale-invariant."""

import math

from tests.pnl.oracles import returns_simple_reference
from tests.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.pnl import returns_simple

RETURNS_SIMPLE = Spec(
    factory=returns_simple,
    inputs=("price",),
    params={},
    shape=Shape.SERIES,
    warmup=1,
    oracle=returns_simple_reference,
    # A price ratio minus one: scaling the series leaves the return unchanged, degree 0.
    scale=(ScaleAxis(roles=("price",), degree=0),),
    golden_input={"price": (100.0, 105.0, 102.0, 108.0, 110.0)},
    golden_output=(None, 0.05, -0.0286, 0.0588, 0.0185),
    pins=(
        SpecPin(
            label="null_precedence",
            inputs={"price": (100.0, None, math.nan, 108.0)},
            expected=(None, None, None, math.nan),
            reason="the row where a NaN price meets the previous row's null is null (null wins), while the next return "
            "off the NaN is NaN",
        ),
        SpecPin(
            label="zero_previous_price",
            inputs={"price": (0.0, 10.0, 0.0, 0.0)},
            expected=(None, math.inf, -1.0, math.nan),
            reason="the IEEE division boundary: a non-zero change over a zero previous price is +/-inf, a zero change "
            "over zero is NaN; the fuzz draws strictly positive prices, so only a fixed case holds this boundary",
        ),
        SpecPin(
            label="negative_zero_positive_change",
            inputs={"price": (-0.0, 10.0)},
            expected=(None, -math.inf),
            reason="a -0.0 previous price flips the sign of the infinite return: a positive change over -0.0 is -inf",
            signed=True,
        ),
        SpecPin(
            label="negative_zero_negative_change",
            inputs={"price": (-0.0, -10.0)},
            expected=(None, math.inf),
            reason="the counterpart: a negative change over -0.0 is +inf",
            signed=True,
        ),
        SpecPin(
            label="consecutive_infinities",
            inputs={"price": (math.inf, math.inf, 1.0, -math.inf)},
            expected=(None, math.nan, -1.0, -math.inf),
            reason="two consecutive equal-sign infinite prices make the second return inf / inf - 1 = NaN; the "
            "property tiers set allow_infinity=False",
        ),
    ),
)
