"""
Declaration for ``pomata.indicators.donchian_channels`` — the rolling high/low channel struct, window-nulling,
degree-1.
"""

from pomata.indicators import donchian_channels
from tests_new.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests_new.indicators.harness import suite_indicators
from tests_new.indicators.oracles import reference_donchian_channels
from tests_new.support.declaration import Golden, Pin, ScaleAxis, Shape

DONCHIAN_CHANNELS = suite_indicators(
    factory=donchian_channels,
    inputs=("high", "low"),
    params={"window": 20},
    null=BehaviorNull.IN_WINDOW_IS_NULL,
    nan=BehaviorNan.PROPAGATES,
    shape=Shape.STRUCT,
    fields=("lower", "middle", "upper"),
    warmup=Warmup.PER_FIELD,
    warmup_value={"lower": 19, "middle": 19, "upper": 19},
    oracle=reference_donchian_channels,
    scaling=(ScaleAxis(roles=("high", "low"), degree={"lower": 1, "middle": 1, "upper": 1}),),
    talib=RelationTalib.NO_EQUIVALENT,
    talib_reason="TA-Lib has no Donchian channels.",
    raises=(({"window": 0}, r"window must be >= 1"),),
    golden=Golden(
        inputs={
            "high": (11.0, 12.0, 13.0, 12.5, 14.0),
            "low": (9.0, 10.0, 11.0, 11.0, 12.0),
        },
        output={
            "lower": (None, None, 9.0, 10.0, 11.0),
            "middle": (None, None, 11.0, 11.5, 12.5),
            "upper": (None, None, 13.0, 13.0, 14.0),
        },
        params={"window": 3},
    ),
    pins=(
        Pin(
            label="window_one_tracks_each_bar",
            inputs={"high": (11.0, 12.0, 13.0), "low": (9.0, 10.0, 11.0)},
            params_override={"window": 1},
            expected={
                "lower": (9.0, 10.0, 11.0),
                "middle": (10.0, 11.0, 12.0),
                "upper": (11.0, 12.0, 13.0),
            },
            reason="window=1 makes the upper/lower channel the bar's own high/low and the middle their mean, with no "
            "warm-up",
        ),
    ),
)
