"""Declaration for ``pomata.indicators.keltner_channels`` — the EMA-and-ATR band struct, gap-bridging, degree-1."""

import math

from pomata.indicators import keltner_channels
from tests_new.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests_new.indicators.harness import suite_indicators
from tests_new.indicators.oracles import reference_keltner_channels
from tests_new.support.declaration import Golden, Pin, ScaleAxis, Shape

KELTNER_CHANNELS = suite_indicators(
    factory=keltner_channels,
    inputs=("high", "low", "close"),
    params={"window": 20, "window_atr": 10, "multiplier": 2.0},
    null=BehaviorNull.BRIDGED,
    nan=BehaviorNan.LATCHES,
    shape=Shape.STRUCT,
    fields=("lower", "middle", "upper"),
    warmup=Warmup.PER_FIELD,
    warmup_value={"lower": 19, "middle": 19, "upper": 19},
    oracle=reference_keltner_channels,
    scaling=(ScaleAxis(roles=("high", "low", "close"), degree={"lower": 1, "middle": 1, "upper": 1}),),
    talib=RelationTalib.NO_EQUIVALENT,
    talib_reason="TA-Lib has no Keltner channels.",
    raises=(
        ({"window": 0}, r"window must be >= 1"),
        ({"window_atr": 0}, r"window_atr must be >= 1"),
        ({"multiplier": 0.0}, r"multiplier must be a finite number > 0"),
        ({"multiplier": -1.0}, r"multiplier must be a finite number > 0"),
        ({"multiplier": math.nan}, r"multiplier must be a finite number > 0"),
        ({"multiplier": math.inf}, r"multiplier must be a finite number > 0"),
        ({"multiplier": -math.inf}, r"multiplier must be a finite number > 0"),
    ),
    golden=Golden(
        inputs={
            "high": (10.0, 12.0, 11.0, 13.0, 15.0),
            "low": (8.0, 9.0, 9.5, 10.0, 12.0),
            "close": (9.0, 11.0, 10.0, 12.0, 14.0),
        },
        output={
            "lower": (None, None, 5.6667, 6.1111, 7.2407),
            "middle": (None, None, 10.0, 11.0, 12.5),
            "upper": (None, None, 14.3333, 15.8889, 17.7593),
        },
        params={"window": 3, "window_atr": 3},
    ),
    pins=(
        Pin(
            label="flat_range_zero_atr_collapses_to_ema",
            inputs={"high": (4.0, 4.0, 4.0, 4.0), "low": (4.0, 4.0, 4.0, 4.0), "close": (4.0, 4.0, 4.0, 4.0)},
            params_override={"window": 2, "window_atr": 2},
            expected={
                "lower": (None, 4.0, 4.0, 4.0),
                "middle": (None, 4.0, 4.0, 4.0),
                "upper": (None, 4.0, 4.0, 4.0),
            },
            reason="a flat series has zero ATR, so all three bands collapse onto the EMA of the close ",
        ),
    ),
)
