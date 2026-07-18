"""
Declaration for ``pomata.indicators.price_typical`` — the HLC mean, elementwise, propagating, degree-1 homogeneous.
"""

import math

from pomata.indicators import price_typical
from tests_new.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests_new.indicators.harness import suite_indicators
from tests_new.indicators.oracles import reference_price_typical
from tests_new.support.declaration import Golden, Pin, ScaleAxis, Shape

PRICE_TYPICAL = suite_indicators(
    factory=price_typical,
    inputs=("high", "low", "close"),
    params={},
    null=BehaviorNull.PROPAGATES,
    nan=BehaviorNan.PROPAGATES,
    shape=Shape.SERIES,
    warmup=Warmup.NONE,
    oracle=reference_price_typical,
    scaling=(ScaleAxis(roles=("high", "low", "close"), degree=1),),
    talib=RelationTalib.MATCHES,
    golden=Golden(
        inputs={
            "high": (11.0, 12.0, 13.0, 12.5, 14.0),
            "low": (9.0, 10.0, 11.0, 11.0, 12.0),
            "close": (10.0, 11.5, 12.5, 11.5, 13.5),
        },
        output=(10.0, 11.1667, 12.1667, 11.6667, 13.1667),
    ),
    pins=(
        Pin(
            label="null_precedence_null_high_nan_low",
            inputs={"high": (11.0, None), "low": (9.0, math.nan), "close": (10.0, 11.5)},
            expected=(10.0, None),
            reason="a null in high combined with a NaN in low on the same row yields null — null wins over NaN",
        ),
    ),
)
