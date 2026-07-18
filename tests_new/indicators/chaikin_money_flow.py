"""
Declaration for ``pomata.indicators.chaikin_money_flow`` — the windowed money-flow ratio, window-nulling, invariant.
"""

import math

from pomata.indicators import chaikin_money_flow
from tests_new.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests_new.indicators.harness import suite_indicators
from tests_new.indicators.oracles import reference_chaikin_money_flow
from tests_new.support.declaration import Golden, Pin, ScaleAxis, Shape

CHAIKIN_MONEY_FLOW = suite_indicators(
    factory=chaikin_money_flow,
    inputs=("high", "low", "close", "volume"),
    params={"window": 20},
    null=BehaviorNull.IN_WINDOW_IS_NULL,
    nan=BehaviorNan.PROPAGATES,
    shape=Shape.SERIES,
    warmup=Warmup.WINDOW_MINUS_ONE,
    oracle=reference_chaikin_money_flow,
    scaling=(
        ScaleAxis(roles=("high", "low", "close"), degree=0),
        ScaleAxis(roles=("volume",), degree=0),
    ),
    talib=RelationTalib.NO_EQUIVALENT,
    talib_reason="TA-Lib has the A/D line and Chaikin oscillator, but not the volume-normalized CMF.",
    raises=(({"window": 0}, r"window must be >= 1"),),
    golden=Golden(
        inputs={
            "high": (10.0, 12.0, 11.0, 13.0, 14.0),
            "low": (8.0, 9.0, 9.0, 10.0, 11.0),
            "close": (9.0, 11.0, 10.0, 12.0, 13.0),
            "volume": (100.0, 200.0, 150.0, 300.0, 250.0),
        },
        output=(None, None, 0.1481, 0.2564, 0.2619),
        params={"window": 3},
    ),
    pins=(
        Pin(
            label="zero_total_volume_is_nan",
            inputs={
                "high": (10.0, 12.0, 11.0),
                "low": (8.0, 9.0, 9.0),
                "close": (9.0, 11.0, 10.0),
                "volume": (0.0, 0.0, 0.0),
            },
            params_override={"window": 2},
            expected=(None, math.nan, math.nan),
            reason="a window whose total volume is zero divides by zero, the IEEE-754 0/0 == NaN",
        ),
        Pin(
            label="zero_volume_after_large_volume_is_nan",
            inputs={
                "high": (12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0),
                "low": (10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0),
                "close": (11.0, 12.5, 13.5, 14.0, 15.5, 16.0, 17.5),
                "volume": (1e16, 0.1, 0.2, 0.3, 0.0, 0.0, 0.0),
            },
            params_override={"window": 3},
            expected=(None, None, 0.0, 0.25, 0.2, 0.0, math.nan),
            reason="an all-zero-volume window still yields NaN after a large 1e16 volume has slid out: the rolling sum "
            "retains a sub-ULP residual on exit, but the exact all-zero detection pins the final window to NaN",
        ),
    ),
)
