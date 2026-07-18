"""Declaration for ``pomata.indicators.dx`` — Wilder's directional movement index, gap-bridging, scale-invariant."""

import math

from pomata.indicators import dx
from tests_new.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests_new.indicators.harness import suite_indicators
from tests_new.indicators.oracles import reference_dx
from tests_new.support.declaration import Golden, Pin, ScaleAxis, Shape

DX = suite_indicators(
    factory=dx,
    inputs=("high", "low", "close"),
    params={"window": 14},
    null=BehaviorNull.BRIDGED,
    nan=BehaviorNan.LATCHES,
    shape=Shape.SERIES,
    warmup=Warmup.WINDOW_MINUS_ONE,
    oracle=reference_dx,
    scaling=(ScaleAxis(roles=("high", "low", "close"), degree=0),),
    talib=RelationTalib.MATCHES,
    raises=(({"window": 0}, r"window must be >= 1"),),
    golden=Golden(
        inputs={
            "high": (10.0, 11.0, 12.0, 11.5, 13.0, 12.5, 14.0),
            "low": (9.0, 10.0, 11.0, 10.5, 12.0, 11.5, 13.0),
            "close": (9.5, 10.5, 11.5, 11.0, 12.5, 12.0, 13.5),
        },
        output=(None, 100.0, 100.0, 20.0, 76.4706, 20.0, 72.6027),
        params={"window": 2},
    ),
    pins=(
        Pin(
            label="flat_window_is_nan",
            inputs={"high": (10.0,) * 8, "low": (10.0,) * 8, "close": (10.0,) * 8},
            params_override={"window": 3},
            expected=(None, None, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan),
            reason="a fully flat window has no movement either way, so both directional indicators are NaN and the "
            "indeterminate 0/0 spread propagates",
        ),
    ),
)
