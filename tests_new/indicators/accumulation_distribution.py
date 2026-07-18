"""
Declaration for ``pomata.indicators.accumulation_distribution`` — the running money-flow-volume total, gap-bridging.
"""

from pomata.indicators import accumulation_distribution
from tests_new.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests_new.indicators.harness import suite_indicators
from tests_new.indicators.oracles import reference_accumulation_distribution
from tests_new.support.declaration import Golden, ScaleAxis, Shape

ACCUMULATION_DISTRIBUTION = suite_indicators(
    factory=accumulation_distribution,
    inputs=("high", "low", "close", "volume"),
    params={},
    null=BehaviorNull.BRIDGED,
    nan=BehaviorNan.LATCHES,
    shape=Shape.SERIES,
    warmup=Warmup.NONE,
    oracle=reference_accumulation_distribution,
    scaling=(
        ScaleAxis(roles=("volume",), degree=1),
        ScaleAxis(roles=("high", "low", "close"), degree=0),
    ),
    talib=RelationTalib.MATCHES,
    golden=Golden(
        inputs={
            "high": (10.0, 11.0, 12.0, 13.0, 14.0),
            "low": (8.0, 9.0, 10.0, 11.0, 12.0),
            "close": (9.0, 10.5, 10.0, 13.0, 12.5),
            "volume": (100.0, 200.0, 300.0, 400.0, 500.0),
        },
        output=(0.0, 100.0, -200.0, 200.0, -50.0),
    ),
)
