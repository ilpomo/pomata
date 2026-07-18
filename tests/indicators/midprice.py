"""
Declaration for ``pomata.indicators.midprice`` — the rolling high/low midprice of a bar series, window-nulling,
degree-1.
"""

from pomata.indicators import midprice
from tests.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests.indicators.harness import suite_indicators
from tests.indicators.oracles import reference_midprice
from tests.support.declaration import Golden, Pin, ScaleAxis, Shape

MIDPRICE = suite_indicators(
    factory=midprice,
    inputs=("high", "low"),
    params={"window": 14},
    null=BehaviorNull.IN_WINDOW_IS_NULL,
    nan=BehaviorNan.PROPAGATES,
    shape=Shape.SERIES,
    warmup=Warmup.WINDOW_MINUS_ONE,
    oracle=reference_midprice,
    scaling=(ScaleAxis(roles=("high", "low"), degree=1),),
    talib=RelationTalib.MATCHES,
    raises=(({"window": 0}, r"window must be >= 1"),),
    golden=Golden(
        inputs={
            "high": (11.0, 12.0, 13.0, 12.5, 14.0),
            "low": (9.0, 10.0, 11.0, 11.0, 12.0),
        },
        output=(None, None, 11.0, 11.5, 12.5),
        params={"window": 3},
    ),
    pins=(
        Pin(
            label="window_one_is_price_median",
            inputs={"high": (11.0, 12.0, 13.0), "low": (9.0, 10.0, 11.0)},
            params_override={"window": 1},
            expected=(10.0, 11.0, 12.0),
            reason="window=1 makes the extremes the bar's own high and low, so the midprice reduces to the per-bar "
            "price_median with no warm-up — the documented degenerate branch, mirroring midpoint's window=1 pin",
        ),
    ),
)
