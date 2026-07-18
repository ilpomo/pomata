"""Declaration for ``pomata.indicators.atr`` — Wilder's Average True Range, gap-bridging, NaN-latching, degree-1."""

from pomata.indicators import atr
from tests.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests.indicators.harness import suite_indicators
from tests.indicators.oracles import reference_atr
from tests.support.declaration import Golden, Pin, ScaleAxis, Shape

ATR = suite_indicators(
    factory=atr,
    inputs=("high", "low", "close"),
    params={"window": 14},
    null=BehaviorNull.BRIDGED,
    nan=BehaviorNan.LATCHES,
    shape=Shape.SERIES,
    warmup=Warmup.WINDOW_MINUS_ONE,
    oracle=reference_atr,
    scaling=(ScaleAxis(roles=("high", "low", "close"), degree=1),),
    talib=RelationTalib.MATCHES,
    raises=(({"window": 0}, r"window must be >= 1"),),
    golden=Golden(
        inputs={
            "high": (10.0, 12.0, 11.0, 13.0, 15.0, 14.0, 16.0, 18.0),
            "low": (8.0, 9.0, 9.5, 10.0, 12.0, 11.0, 13.0, 15.0),
            "close": (9.0, 11.0, 10.0, 12.0, 14.0, 13.0, 15.0, 17.0),
        },
        output=(None, None, 2.1667, 2.4444, 2.6296, 2.7531, 2.8354, 2.8903),
        params={"window": 3},
    ),
    pins=(
        Pin(
            label="window_one_is_true_range",
            inputs={"high": (10.0, 12.0, 11.0, 13.0), "low": (8.0, 9.0, 9.5, 10.0), "close": (9.0, 11.0, 10.0, 12.0)},
            params_override={"window": 1},
            expected=(2.0, 3.0, 1.5, 3.0),
            reason="window=1 makes the Wilder smoothing the identity, so the ATR reproduces the true range",
        ),
    ),
)
