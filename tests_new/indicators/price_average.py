"""
Declaration for ``pomata.indicators.price_average`` — the OHLC mean, elementwise, propagating, degree-1 homogeneous.
"""

import math

from pomata.indicators import price_average
from tests_new.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests_new.indicators.harness import suite_indicators
from tests_new.indicators.oracles import reference_price_average
from tests_new.support.declaration import Golden, Pin, ScaleAxis, Shape

PRICE_AVERAGE = suite_indicators(
    factory=price_average,
    inputs=("open", "high", "low", "close"),
    params={},
    null=BehaviorNull.PROPAGATES,
    nan=BehaviorNan.PROPAGATES,
    shape=Shape.SERIES,
    warmup=Warmup.NONE,
    oracle=reference_price_average,
    scaling=(ScaleAxis(roles=("open", "high", "low", "close"), degree=1),),
    talib=RelationTalib.MATCHES,
    golden=Golden(
        inputs={
            "open": (10.0, 11.0, 12.0, 11.5, 13.0),
            "high": (11.0, 12.0, 13.0, 12.5, 14.0),
            "low": (9.0, 10.0, 11.0, 11.0, 12.0),
            "close": (10.0, 11.5, 12.5, 11.5, 13.5),
        },
        output=(10.0, 11.125, 12.125, 11.625, 13.125),
    ),
    pins=(
        Pin(
            label="null_takes_precedence_over_nan",
            inputs={"open": (10.0, None), "high": (11.0, math.nan), "low": (9.0, 10.0), "close": (10.0, 11.5)},
            expected=(10.0, None),
            reason="a null in open and a NaN in high on the same row yields null — null wins over NaN",
        ),
    ),
)
