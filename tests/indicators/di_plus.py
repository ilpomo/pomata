"""
Declaration for ``pomata.indicators.di_plus`` — Wilder's positive directional indicator, gap-bridging, scale-
invariant.
"""

import math

from pomata.indicators import di_plus
from tests.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests.indicators.harness import suite_indicators
from tests.indicators.oracles import reference_di_plus
from tests.support.declaration import Golden, Pin, ScaleAxis, Shape

DI_PLUS = suite_indicators(
    factory=di_plus,
    inputs=("high", "low", "close"),
    params={"window": 14},
    null=BehaviorNull.BRIDGED,
    nan=BehaviorNan.LATCHES,
    shape=Shape.SERIES,
    warmup=Warmup.WINDOW_MINUS_ONE,
    oracle=reference_di_plus,
    scaling=(ScaleAxis(roles=("high", "low", "close"), degree=0),),
    talib=RelationTalib.MATCHES,
    raises=(({"window": 0}, r"window must be >= 1"),),
    golden=Golden(
        inputs={
            "high": (10.0, 11.0, 12.0, 11.5, 13.0, 12.5, 14.0),
            "low": (9.0, 10.0, 11.0, 10.5, 12.0, 11.5, 13.0),
            "close": (9.5, 10.5, 11.5, 11.0, 12.5, 12.0, 13.5),
        },
        output=(None, 40.0, 54.5455, 31.5789, 58.8235, 36.1446, 59.7156),
        params={"window": 2},
    ),
    pins=(
        Pin(
            label="flat_window_is_nan",
            inputs={"high": (10.0,) * 8, "low": (10.0,) * 8, "close": (10.0,) * 8},
            params_override={"window": 3},
            expected=(None, None, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan),
            reason="a fully flat window makes the average true range zero, so the smoothed movement over it is the "
            "indeterminate 0/0",
        ),
    ),
)
