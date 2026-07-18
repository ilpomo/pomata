"""Declaration for ``pomata.indicators.obv`` — On-Balance Volume, the signed-volume running total, gap-bridging."""

from pomata.indicators import obv
from tests.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests.indicators.harness import suite_indicators
from tests.indicators.oracles import reference_obv
from tests.support.declaration import Golden, Pin, ScaleAxis, Shape

OBV = suite_indicators(
    factory=obv,
    inputs=("price", "volume"),
    params={},
    null=BehaviorNull.BRIDGED,
    nan=BehaviorNan.LATCHES,
    shape=Shape.SERIES,
    warmup=Warmup.NONE,
    oracle=reference_obv,
    scaling=(
        ScaleAxis(roles=("volume",), degree=1),
        ScaleAxis(roles=("price",), degree=0),
    ),
    talib=RelationTalib.DOCUMENTED_DIVERGENCE,
    talib_reason="OBV is a cumulative sum with an arbitrary origin; pomata seeds OBV[0] = 0, TA-Lib uses volume[0].",
    golden=Golden(
        inputs={
            "price": (10.0, 12.0, 11.0, 11.0, 13.0, 9.0, 9.0, 14.0),
            "volume": (100.0, 200.0, 150.0, 80.0, 300.0, 250.0, 90.0, 400.0),
        },
        output=(0.0, 200.0, 50.0, 50.0, 350.0, 100.0, 100.0, 500.0),
    ),
    pins=(
        Pin(
            label="flat_price_never_moves_the_total",
            inputs={"price": (5.0, 5.0, 5.0, 5.0), "volume": (10.0, 20.0, 30.0, 40.0)},
            expected=(0.0, 0.0, 0.0, 0.0),
            reason="an unchanged price contributes no signed volume, so the running total stays at the seed 0 ",
        ),
    ),
)
