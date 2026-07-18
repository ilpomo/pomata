"""
Declaration for ``pomata.indicators.adxr`` — Wilder's average directional index rating, gap-bridging, scale-invariant.
"""

import math

from pomata.indicators import adxr
from tests.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests.indicators.harness import suite_indicators
from tests.indicators.oracles import reference_adxr
from tests.support.declaration import Golden, Pin, ScaleAxis, Shape

ADXR = suite_indicators(
    factory=adxr,
    inputs=("high", "low", "close"),
    params={"window": 14},
    null=BehaviorNull.BRIDGED,
    nan=BehaviorNan.LATCHES,
    shape=Shape.SERIES,
    warmup=Warmup.EXPR,
    warmup_value=40,
    oracle=reference_adxr,
    scaling=(ScaleAxis(roles=("high", "low", "close"), degree=0),),
    talib=RelationTalib.DOCUMENTED_DIVERGENCE,
    talib_reason="ADXR averages ADX with its lagged self; pomata lags by `window`, TA-Lib by `window - 1`.",
    raises=(({"window": 0}, r"window must be >= 1"),),
    golden=Golden(
        inputs={
            "high": (10.0, 11.0, 12.0, 11.5, 13.0, 12.5, 14.0, 13.5, 15.0, 14.5),
            "low": (9.0, 10.0, 11.0, 10.5, 12.0, 11.5, 13.0, 12.5, 14.0, 13.5),
            "close": (9.5, 10.5, 11.5, 11.0, 12.5, 12.0, 13.5, 13.0, 14.5, 14.0),
        },
        output=(None, None, None, None, 84.1176, 52.0588, 63.2977, 41.6489, 56.9044, 38.4522),
        params={"window": 2},
    ),
    pins=(
        Pin(
            label="flat_window_is_nan",
            inputs={"high": (10.0,) * 11, "low": (10.0,) * 11, "close": (10.0,) * 11},
            params_override={"window": 3},
            expected=(
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                math.nan,
                math.nan,
                math.nan,
                math.nan,
            ),
            reason="a fully flat window makes the underlying dx the indeterminate 0/0, which poisons the Wilder "
            "smoothing recursion and the averaging of the current and one-window-back adx",
        ),
    ),
)
