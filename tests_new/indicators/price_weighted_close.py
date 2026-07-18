"""Declaration for ``pomata.indicators.price_weighted_close`` — the close-weighted HLC mean, elementwise, degree-1."""

import math

from pomata.indicators import price_weighted_close
from tests_new.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests_new.indicators.harness import suite_indicators
from tests_new.indicators.oracles import reference_price_weighted_close
from tests_new.support.declaration import Golden, Pin, ScaleAxis, Shape

PRICE_WEIGHTED_CLOSE = suite_indicators(
    factory=price_weighted_close,
    inputs=("high", "low", "close"),
    params={},
    null=BehaviorNull.PROPAGATES,
    nan=BehaviorNan.PROPAGATES,
    shape=Shape.SERIES,
    warmup=Warmup.NONE,
    oracle=reference_price_weighted_close,
    scaling=(ScaleAxis(roles=("high", "low", "close"), degree=1),),
    talib=RelationTalib.MATCHES,
    golden=Golden(
        inputs={
            "high": (11.0, 12.0, 13.0, 12.5, 14.0),
            "low": (9.0, 10.0, 11.0, 11.0, 12.0),
            "close": (10.0, 11.5, 12.5, 11.5, 13.5),
        },
        output=(10.0, 11.25, 12.25, 11.625, 13.25),
    ),
    pins=(
        Pin(
            label="null_propagates",
            inputs={"high": (11.0, None, 13.0), "low": (9.0, 10.0, 11.0), "close": (10.0, 11.5, 12.5)},
            expected=(10.0, None, 12.25),
            reason="a null in exactly one input role nulls that row only ",
        ),
        Pin(
            label="nan_propagates",
            inputs={"high": (11.0, math.nan, 13.0), "low": (9.0, 10.0, 11.0), "close": (10.0, 11.5, 12.5)},
            expected=(10.0, math.nan, 12.25),
            reason="a NaN in exactly one input role nans that row only ",
        ),
        Pin(
            label="null_takes_precedence_over_nan",
            inputs={"high": (11.0, None), "low": (9.0, math.nan), "close": (10.0, 11.5)},
            expected=(10.0, None),
            reason="a row carrying both a null (high) and a NaN (low) yields null — null wins over NaN ",
        ),
    ),
)
