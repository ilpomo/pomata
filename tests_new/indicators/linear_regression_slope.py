"""
Declaration for ``pomata.indicators.linear_regression_slope`` — the rolling least-squares slope, degree-1 homogeneous.
"""

from pomata.indicators import linear_regression_slope
from tests_new.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests_new.indicators.harness import suite_indicators
from tests_new.indicators.oracles import reference_linear_regression_slope
from tests_new.support.declaration import Golden, ScaleAxis, Shape

LINEAR_REGRESSION_SLOPE = suite_indicators(
    factory=linear_regression_slope,
    inputs=("expr",),
    params={"window": 14},
    null=BehaviorNull.IN_WINDOW_IS_NULL,
    nan=BehaviorNan.PROPAGATES,
    shape=Shape.SERIES,
    warmup=Warmup.WINDOW_MINUS_ONE,
    oracle=reference_linear_regression_slope,
    scaling=(ScaleAxis(roles=("expr",), degree=1),),
    talib=RelationTalib.MATCHES,
    raises=(({"window": 1}, r"window must be >= 2"),),
    golden=Golden(
        inputs={"expr": (10.0, 11.0, 13.0, 12.0, 14.0, 13.0, 15.0)},
        output=(None, None, 1.5, 0.5, 0.5, 0.5, 0.5),
        params={"window": 3},
    ),
)
