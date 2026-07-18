"""
Declaration for ``pomata.indicators.ichimoku`` — a struct of four rolling midpoints, per-field warm-ups, three
windows.
"""

from pomata.indicators import ichimoku
from tests_new.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests_new.indicators.harness import suite_indicators
from tests_new.indicators.oracles import reference_ichimoku
from tests_new.support.declaration import Golden, ScaleAxis, Shape

ICHIMOKU = suite_indicators(
    factory=ichimoku,
    inputs=("high", "low"),
    params={"window_tenkan": 9, "window_kijun": 26, "window_senkou": 52},
    null=BehaviorNull.IN_WINDOW_IS_NULL,
    nan=BehaviorNan.PROPAGATES,
    shape=Shape.STRUCT,
    fields=("tenkan", "kijun", "senkou_a", "senkou_b"),
    warmup=Warmup.PER_FIELD,
    warmup_value={"tenkan": 8, "kijun": 25, "senkou_a": 25, "senkou_b": 51},
    oracle=reference_ichimoku,
    scaling=(ScaleAxis(roles=("high", "low"), degree={"tenkan": 1, "kijun": 1, "senkou_a": 1, "senkou_b": 1}),),
    talib=RelationTalib.NO_EQUIVALENT,
    talib_reason="TA-Lib has no Ichimoku Kinko Hyo.",
    raises=(
        ({"window_tenkan": 0}, r"window_tenkan must be >= 1"),
        ({"window_kijun": 0, "window_tenkan": 1}, r"window_kijun must be >= 1"),
        ({"window_senkou": 0, "window_tenkan": 1, "window_kijun": 1}, r"window_senkou must be >= 1"),
        ({"window_kijun": 5}, r"windows must be ordered window_tenkan <= window_kijun <= window_senkou"),
        ({"window_senkou": 10}, r"windows must be ordered window_tenkan <= window_kijun <= window_senkou"),
    ),
    golden=Golden(
        inputs={
            "high": (10.0, 12.0, 11.0, 13.0, 14.0, 12.0, 15.0, 13.0),
            "low": (8.0, 9.0, 10.0, 11.0, 12.0, 10.0, 12.0, 11.0),
        },
        output={
            "tenkan": (None, 10.0, 10.5, 11.5, 12.5, 12.0, 12.5, 13.0),
            "kijun": (None, None, 10.0, 11.0, 12.0, 12.0, 12.5, 12.5),
            "senkou_a": (None, None, 10.25, 11.25, 12.25, 12.0, 12.5, 12.75),
            "senkou_b": (None, None, None, 10.5, 11.5, 12.0, 12.5, 12.5),
        },
        params={"window_tenkan": 2, "window_kijun": 3, "window_senkou": 4},
    ),
)
