"""Declaration for ``pomata.indicators.stochastic_fast`` — the fast stochastic struct (k, d), window-nulling."""

import math

from pomata.indicators import stochastic_fast
from tests.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests.indicators.harness import suite_indicators
from tests.indicators.oracles import reference_stochastic_fast
from tests.support.declaration import Golden, Pin, ScaleAxis, Shape

STOCHASTIC_FAST = suite_indicators(
    factory=stochastic_fast,
    inputs=("high", "low", "close"),
    params={"window_k": 14, "window_d": 3},
    null=BehaviorNull.IN_WINDOW_IS_NULL,
    nan=BehaviorNan.PROPAGATES,
    shape=Shape.STRUCT,
    fields=("k", "d"),
    warmup=Warmup.PER_FIELD,
    warmup_value={"k": 13, "d": 15},
    oracle=reference_stochastic_fast,
    scaling=(ScaleAxis(roles=("high", "low", "close"), degree={"k": 0, "d": 0}),),
    talib=RelationTalib.MATCHES,
    raises=(
        ({"window_k": 0}, r"window_k must be >= 1"),
        ({"window_d": 0}, r"window_d must be >= 1"),
    ),
    golden=Golden(
        inputs={
            "high": (10.0, 11.0, 12.0, 11.5, 13.0, 12.5, 14.0, 13.5, 15.0, 14.5),
            "low": (9.0, 10.0, 11.0, 10.5, 12.0, 11.5, 13.0, 12.5, 14.0, 13.5),
            "close": (9.5, 10.5, 11.5, 11.0, 12.5, 12.0, 13.5, 13.0, 14.5, 14.0),
        },
        output={
            "k": (None, None, None, None, 87.5, 66.6667, 85.7143, 71.4286, 85.7143, 71.4286),
            "d": (None, None, None, None, None, None, 79.9603, 74.6032, 80.9524, 76.1905),
        },
        params={"window_k": 5, "window_d": 3},
    ),
    pins=(
        Pin(
            label="flat_window_is_nan",
            inputs={"high": (10.0, 10.0, 10.0), "low": (10.0, 10.0, 10.0), "close": (10.0, 10.0, 10.0)},
            params_override={"window_k": 2, "window_d": 1},
            expected={"k": (None, math.nan, math.nan), "d": (None, math.nan, math.nan)},
            reason="a flat window makes the raw %K's 0/0 division NaN, which the %D pass carries through",
        ),
        Pin(
            label="flat_window_close_off_is_inf",
            inputs={"high": (10.0, 10.0, 10.0), "low": (10.0, 10.0, 10.0), "close": (20.0, 20.0, 20.0)},
            params_override={"window_k": 2, "window_d": 1},
            expected={"k": (None, math.inf, math.inf), "d": (None, math.inf, math.inf)},
            reason="a malformed bar whose close sits above a flat high==low look-back makes the raw %K a non-zero "
            "over zero, so %K is +inf, which the %D pass carries through — the infinity beside the 0/0 NaN pin",
        ),
    ),
)
