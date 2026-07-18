"""Declaration for ``pomata.indicators.vwap`` — the cumulative volume-weighted average price, gap-bridging, degree-1."""

import math

from pomata.indicators import vwap
from tests_new.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests_new.indicators.harness import suite_indicators
from tests_new.indicators.oracles import reference_vwap
from tests_new.support.declaration import Golden, Pin, ScaleAxis, Shape

VWAP = suite_indicators(
    factory=vwap,
    inputs=("high", "low", "close", "volume"),
    params={},
    null=BehaviorNull.BRIDGED,
    nan=BehaviorNan.LATCHES,
    shape=Shape.SERIES,
    warmup=Warmup.NONE,
    oracle=reference_vwap,
    scaling=(
        ScaleAxis(roles=("high", "low", "close"), degree=1),
        ScaleAxis(roles=("volume",), degree=0),
    ),
    talib=RelationTalib.NO_EQUIVALENT,
    talib_reason="TA-Lib has no VWAP.",
    golden=Golden(
        inputs={
            "high": (2.0, 4.0, 6.0),
            "low": (0.0, 2.0, 4.0),
            "close": (1.0, 3.0, 5.0),
            "volume": (10.0, 20.0, 30.0),
        },
        output=(1.0, 2.3333, 3.6667),
    ),
    pins=(
        Pin(
            label="zero_leading_volume_is_nan_then_recovers",
            inputs={
                "high": (10.0, 11.0, 12.0),
                "low": (8.0, 9.0, 10.0),
                "close": (9.0, 10.0, 11.0),
                "volume": (0.0, 100.0, 100.0),
            },
            expected=(math.nan, 10.0, 10.5),
            reason="a zero cumulative volume at the first bar is the 0/0 degenerate (NaN); once volume accrues the "
            "running average recovers",
        ),
    ),
)
