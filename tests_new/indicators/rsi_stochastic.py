"""Declaration for ``pomata.indicators.rsi_stochastic`` — the stochastic of RSI struct (k, d), gap-bridging."""

import math

from pomata.indicators import rsi_stochastic
from tests_new.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests_new.indicators.harness import suite_indicators
from tests_new.indicators.oracles import reference_rsi_stochastic
from tests_new.support.declaration import Golden, Pin, ScaleAxis, Shape

RSI_STOCHASTIC = suite_indicators(
    factory=rsi_stochastic,
    inputs=("wave",),
    params={"window_rsi": 14, "window_k": 14, "window_d": 3},
    null=BehaviorNull.BRIDGED,
    nan=BehaviorNan.LATCHES,
    shape=Shape.STRUCT,
    fields=("k", "d"),
    warmup=Warmup.PER_FIELD,
    warmup_value={"k": 27, "d": 29},
    oracle=reference_rsi_stochastic,
    scaling=(ScaleAxis(roles=("wave",), degree={"k": 0, "d": 0}),),
    talib=RelationTalib.MATCHES,
    raises=(
        ({"window_rsi": 0}, r"window_rsi must be >= 1"),
        ({"window_k": 0}, r"window_k must be >= 1"),
        ({"window_d": 0}, r"window_d must be >= 1"),
    ),
    golden=Golden(
        inputs={"wave": (50.0, 51.0, 50.5, 52.0, 51.5, 53.0, 52.0, 54.0, 53.5, 55.0)},
        output={
            "k": (None, None, None, None, None, 94.7368, 0.0, 81.5861, 44.2237, 100.0),
            "d": (None, None, None, None, None, None, 47.3684, 40.793, 62.9049, 72.1118),
        },
        params={"window_rsi": 3, "window_k": 3, "window_d": 2},
    ),
    pins=(
        Pin(
            label="flat_rsi_window_is_nan",
            inputs={"wave": (10.0, 11.0, 12.0, 13.0, 14.0)},
            params_override={"window_rsi": 2, "window_k": 2, "window_d": 1},
            expected={
                "k": (None, None, None, math.nan, math.nan),
                "d": (None, None, None, math.nan, math.nan),
            },
            reason="a monotone run gives an exactly-flat RSI, so the %K channel normalization is the 0/0 degenerate "
            "NaN",
        ),
    ),
)
