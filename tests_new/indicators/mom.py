"""Declaration for ``pomata.indicators.mom`` — the fixed-lag momentum difference, propagating, degree-1 homogeneous."""

import math

from pomata.indicators import mom
from tests_new.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests_new.indicators.harness import suite_indicators
from tests_new.indicators.oracles import reference_mom
from tests_new.support.declaration import Golden, Pin, ScaleAxis, Shape

MOM = suite_indicators(
    factory=mom,
    inputs=("expr",),
    params={"window": 3},
    null=BehaviorNull.PROPAGATES,
    nan=BehaviorNan.PROPAGATES,
    shape=Shape.SERIES,
    warmup=Warmup.WINDOW,
    oracle=reference_mom,
    scaling=(ScaleAxis(roles=("expr",), degree=1),),
    talib=RelationTalib.MATCHES,
    raises=(({"window": 0}, r"window must be >= 1"),),
    golden=Golden(
        inputs={"expr": (3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0)}, output=(None, None, None, -2.0, 4.0, 5.0, 1.0, 1.0)
    ),
    pins=(
        Pin(
            label="window_one_is_first_difference",
            inputs={"expr": (2.0, 4.0, 6.0, 8.0)},
            expected=(None, 2.0, 2.0, 2.0),
            params_override={"window": 1},
            reason="window=1 is the first difference with a single leading null",
        ),
        Pin(
            label="all_nan_series",
            inputs={"expr": (math.nan, math.nan, math.nan, math.nan)},
            expected=(None, math.nan, math.nan, math.nan),
            params_override={"window": 1},
            reason="an all-NaN series yields null during warm-up and NaN thereafter",
        ),
        Pin(
            label="constant_series_is_zero",
            inputs={"expr": (5.0, 5.0, 5.0, 5.0, 5.0, 5.0)},
            expected=(None, None, None, 0.0, 0.0, 0.0),
            reason="the momentum of a constant series is exactly zero once warmed up",
        ),
    ),
)
