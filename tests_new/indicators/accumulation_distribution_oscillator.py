"""
Declaration for ``pomata.indicators.accumulation_distribution_oscillator`` — the Chaikin A/D oscillator, gap-bridging.
"""

from pomata.indicators import accumulation_distribution_oscillator
from tests_new.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests_new.indicators.harness import suite_indicators
from tests_new.indicators.oracles import reference_accumulation_distribution_oscillator
from tests_new.support.declaration import Golden, ScaleAxis, Shape

ACCUMULATION_DISTRIBUTION_OSCILLATOR = suite_indicators(
    factory=accumulation_distribution_oscillator,
    inputs=("high", "low", "close", "volume"),
    params={"window_fast": 3, "window_slow": 10},
    null=BehaviorNull.BRIDGED,
    nan=BehaviorNan.LATCHES,
    shape=Shape.SERIES,
    warmup=Warmup.EXPR,
    warmup_value=9,
    oracle=reference_accumulation_distribution_oscillator,
    scaling=(
        ScaleAxis(roles=("volume",), degree=1),
        ScaleAxis(roles=("high", "low", "close"), degree=0),
    ),
    talib=RelationTalib.MATCHES,
    raises=(
        ({"window_fast": 0}, r"window_fast must be >= 1"),
        ({"window_slow": 0}, r"window_slow must be >= 1"),
        ({"window_fast": 10, "window_slow": 3}, r"windows must be ordered window_fast <= window_slow"),
    ),
    golden=Golden(
        inputs={
            "high": (10.2, 10.5, 10.7, 10.3, 10.8),
            "low": (9.8, 10.0, 10.2, 9.9, 10.3),
            "close": (10.0, 10.3, 10.5, 10.1, 10.6),
            "volume": (100.0, 150.0, 120.0, 200.0, 180.0),
        },
        output=(None, None, 13.0, 8.6667, 11.0556),
        params={"window_fast": 2, "window_slow": 3},
    ),
)
