"""
Declaration for ``pomata.indicators.atr_normalized`` — the ATR as a percentage of close, gap-bridging, scale-
invariant.
"""

from pomata.indicators import atr_normalized
from tests_new.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests_new.indicators.harness import suite_indicators
from tests_new.indicators.oracles import reference_atr_normalized
from tests_new.support.declaration import Golden, ScaleAxis, Shape

ATR_NORMALIZED = suite_indicators(
    factory=atr_normalized,
    inputs=("high", "low", "close"),
    params={"window": 14},
    null=BehaviorNull.BRIDGED,
    nan=BehaviorNan.LATCHES,
    shape=Shape.SERIES,
    warmup=Warmup.WINDOW_MINUS_ONE,
    oracle=reference_atr_normalized,
    scaling=(ScaleAxis(roles=("high", "low", "close"), degree=0),),
    talib=RelationTalib.MATCHES,
    raises=(({"window": 0}, r"window must be >= 1"),),
    golden=Golden(
        inputs={
            "high": (10.2, 10.5, 10.7, 10.3, 10.8),
            "low": (9.8, 10.0, 10.2, 9.9, 10.3),
            "close": (10.0, 10.3, 10.5, 10.1, 10.6),
        },
        output=(None, 4.3689, 4.5238, 5.3218, 5.8373),
        params={"window": 2},
    ),
)
