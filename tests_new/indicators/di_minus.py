"""
Declaration for ``pomata.indicators.di_minus`` — Wilder's negative directional indicator, gap-bridging, scale-
invariant.
"""

import math

from pomata.indicators import di_minus
from tests_new.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests_new.indicators.harness import suite_indicators
from tests_new.indicators.oracles import reference_di_minus
from tests_new.support.declaration import Golden, Pin, ScaleAxis, Shape

DI_MINUS = suite_indicators(
    factory=di_minus,
    inputs=("high", "low", "close"),
    params={"window": 14},
    null=BehaviorNull.BRIDGED,
    nan=BehaviorNan.LATCHES,
    shape=Shape.SERIES,
    warmup=Warmup.WINDOW_MINUS_ONE,
    oracle=reference_di_minus,
    scaling=(ScaleAxis(roles=("high", "low", "close"), degree=0),),
    talib=RelationTalib.MATCHES,
    raises=(({"window": 0}, r"window must be >= 1"),),
    golden=Golden(
        inputs={
            "high": (10.0, 11.0, 12.0, 11.5, 13.0, 12.5, 14.0),
            "low": (9.0, 10.0, 11.0, 10.5, 12.0, 11.5, 13.0),
            "close": (9.5, 10.5, 11.5, 11.0, 12.5, 12.0, 13.5),
        },
        output=(None, 0.0, 0.0, 21.0526, 7.8431, 24.0964, 9.4787),
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
