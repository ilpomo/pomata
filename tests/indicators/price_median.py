"""Declaration for ``pomata.indicators.price_median`` — the high-low midpoint, elementwise, propagating, degree-1."""

import math

from pomata.indicators import price_median
from tests.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests.indicators.harness import suite_indicators
from tests.indicators.oracles import reference_price_median
from tests.support.declaration import Golden, Pin, ScaleAxis, Shape

PRICE_MEDIAN = suite_indicators(
    factory=price_median,
    inputs=("high", "low"),
    params={},
    null=BehaviorNull.PROPAGATES,
    nan=BehaviorNan.PROPAGATES,
    shape=Shape.SERIES,
    warmup=Warmup.NONE,
    oracle=reference_price_median,
    scaling=(ScaleAxis(roles=("high", "low"), degree=1),),
    talib=RelationTalib.MATCHES,
    golden=Golden(
        inputs={"high": (11.0, 12.0, 13.0, 12.5, 14.0), "low": (9.0, 10.0, 11.0, 11.0, 12.0)},
        output=(10.0, 11.0, 12.0, 11.75, 13.0),
    ),
    pins=(
        Pin(
            label="null_precedence_null_high_nan_low",
            inputs={"high": (11.0, None), "low": (9.0, math.nan)},
            expected=(10.0, None),
            reason="a null in high against a NaN in low on the same row yields null — null wins over NaN ",
        ),
    ),
)
