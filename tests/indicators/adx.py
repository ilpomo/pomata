"""Declaration for ``pomata.indicators.adx`` — Wilder's average directional index, gap-bridging, scale-invariant."""

import math

from pomata.indicators import adx
from tests.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests.indicators.harness import suite_indicators
from tests.indicators.oracles import reference_adx
from tests.support.declaration import Golden, Pin, ScaleAxis, Shape

ADX = suite_indicators(
    factory=adx,
    inputs=("high", "low", "close"),
    params={"window": 14},
    null=BehaviorNull.BRIDGED,
    nan=BehaviorNan.LATCHES,
    shape=Shape.SERIES,
    warmup=Warmup.EXPR,
    warmup_value=26,
    oracle=reference_adx,
    scaling=(ScaleAxis(roles=("high", "low", "close"), degree=0),),
    talib=RelationTalib.MATCHES,
    raises=(({"window": 0}, r"window must be >= 1"),),
    golden=Golden(
        inputs={
            "high": (10.0, 11.0, 12.0, 11.5, 13.0, 12.5, 14.0, 13.5, 15.0, 14.5),
            "low": (9.0, 10.0, 11.0, 10.5, 12.0, 11.5, 13.0, 12.5, 14.0, 13.5),
            "close": (9.5, 10.5, 11.5, 11.0, 12.5, 12.0, 13.5, 13.0, 14.5, 14.0),
        },
        output=(None, None, 100.0, 60.0, 68.2353, 44.1176, 58.3602, 39.1801, 55.4486, 37.7243),
        params={"window": 2},
    ),
    pins=(
        Pin(
            label="flat_window_is_nan",
            inputs={"high": (10.0,) * 8, "low": (10.0,) * 8, "close": (10.0,) * 8},
            params_override={"window": 3},
            expected=(None, None, None, None, math.nan, math.nan, math.nan, math.nan),
            reason="a fully flat window makes the underlying dx the indeterminate 0/0 (both directional indicators are "
            "zero), which then poisons the Wilder smoothing recursion",
        ),
    ),
)
