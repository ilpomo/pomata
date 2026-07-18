"""
Declaration for ``pomata.indicators.time_series_forecast`` — the one-step least-squares forecast, degree-1
homogeneous.
"""

from pomata.indicators import time_series_forecast
from tests_new.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests_new.indicators.harness import suite_indicators
from tests_new.indicators.oracles import reference_time_series_forecast
from tests_new.support.declaration import Golden, ScaleAxis, Shape

TIME_SERIES_FORECAST = suite_indicators(
    factory=time_series_forecast,
    inputs=("expr",),
    params={"window": 14},
    null=BehaviorNull.IN_WINDOW_IS_NULL,
    nan=BehaviorNan.PROPAGATES,
    shape=Shape.SERIES,
    warmup=Warmup.WINDOW_MINUS_ONE,
    oracle=reference_time_series_forecast,
    scaling=(ScaleAxis(roles=("expr",), degree=1),),
    talib=RelationTalib.MATCHES,
    raises=(({"window": 1}, r"window must be >= 2"),),
    golden=Golden(
        inputs={"expr": (10.0, 11.0, 13.0, 12.0, 14.0, 13.0, 15.0)},
        output=(None, None, 14.3333, 13.0, 14.0, 14.0, 15.0),
        params={"window": 3},
    ),
)
