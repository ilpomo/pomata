"""
Declaration for ``pomata.indicators.aroon`` — the time-since-extreme struct (up, down), window-nulling, scale-
invariant.
"""

from pomata.indicators import aroon
from tests.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests.indicators.harness import suite_indicators
from tests.indicators.oracles import reference_aroon
from tests.support.declaration import Golden, Pin, ScaleAxis, Shape

AROON = suite_indicators(
    factory=aroon,
    inputs=("high", "low"),
    params={"window": 25},
    null=BehaviorNull.IN_WINDOW_IS_NULL,
    nan=BehaviorNan.PROPAGATES,
    shape=Shape.STRUCT,
    fields=("up", "down"),
    warmup=Warmup.PER_FIELD,
    warmup_value={"up": 25, "down": 25},
    oracle=reference_aroon,
    scaling=(ScaleAxis(roles=("high", "low"), degree={"up": 0, "down": 0}),),
    talib=RelationTalib.MATCHES,
    raises=(({"window": 0}, r"window must be >= 1"),),
    golden=Golden(
        inputs={
            "high": (10.0, 11.0, 12.0, 11.0, 13.0, 12.0, 14.0, 13.0),
            "low": (9.0, 10.0, 11.0, 10.0, 12.0, 11.0, 13.0, 12.0),
        },
        output={
            "up": (None, None, None, 66.6667, 100.0, 66.6667, 100.0, 66.6667),
            "down": (None, None, None, 0.0, 66.6667, 33.3333, 0.0, 33.3333),
        },
        params={"window": 3},
    ),
    pins=(
        Pin(
            label="current_extreme_reads_100",
            inputs={"high": (1.0, 2.0, 3.0), "low": (3.0, 2.0, 1.0)},
            params_override={"window": 2},
            expected={"up": (None, None, 100.0), "down": (None, None, 100.0)},
            reason="when the current bar holds the look-back high (low) the up (down) line reads 100 ",
        ),
        Pin(
            label="ties_use_most_recent_extreme",
            inputs={"high": (5.0, 5.0, 3.0), "low": (1.0, 2.0, 3.0)},
            params_override={"window": 2},
            expected={"up": (None, None, 50.0), "down": (None, None, 0.0)},
            reason="a repeated high resolves to the most recent occurrence (one bar back, up=50) ",
        ),
    ),
)
