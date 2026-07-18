"""
Declaration for ``pomata.indicators.linear_regression`` — the rolling least-squares endpoint, window-nulling,
degree-1.
"""

from pomata.indicators import linear_regression
from tests.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests.indicators.harness import suite_indicators
from tests.indicators.oracles import reference_linear_regression
from tests.support.declaration import Golden, ScaleAxis, Shape

LINEAR_REGRESSION = suite_indicators(
    factory=linear_regression,
    inputs=("expr",),
    params={"window": 14},
    null=BehaviorNull.IN_WINDOW_IS_NULL,
    nan=BehaviorNan.PROPAGATES,
    shape=Shape.SERIES,
    warmup=Warmup.WINDOW_MINUS_ONE,
    oracle=reference_linear_regression,
    scaling=(ScaleAxis(roles=("expr",), degree=1),),
    talib=RelationTalib.MATCHES,
    raises=(({"window": 1}, r"window must be >= 2"),),
    golden=Golden(
        inputs={"expr": (10.0, 11.0, 13.0, 12.0, 14.0, 13.0, 15.0)},
        output=(None, None, 12.8333, 12.5, 13.5, 13.5, 14.5),
        params={"window": 3},
    ),
)
