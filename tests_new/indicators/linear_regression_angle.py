"""
Declaration for ``pomata.indicators.linear_regression_angle`` — the least-squares slope as a degree angle, scale-
exempt.
"""

from pomata.indicators import linear_regression_angle
from tests_new.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests_new.indicators.harness import suite_indicators
from tests_new.indicators.oracles import reference_linear_regression_angle
from tests_new.support.declaration import Golden, ScaleExempt, Shape

LINEAR_REGRESSION_ANGLE = suite_indicators(
    factory=linear_regression_angle,
    inputs=("expr",),
    params={"window": 14},
    null=BehaviorNull.IN_WINDOW_IS_NULL,
    nan=BehaviorNan.PROPAGATES,
    shape=Shape.SERIES,
    warmup=Warmup.WINDOW_MINUS_ONE,
    oracle=reference_linear_regression_angle,
    scaling=ScaleExempt(
        reason="atan(slope) in degrees: a rescaling scales the slope inside the arctangent, so the angle is neither "
        "invariant nor degree-1 homogeneous; it is a bounded O(1) value in (-90, 90)"
    ),
    talib=RelationTalib.MATCHES,
    raises=(({"window": 1}, r"window must be >= 2"),),
    golden=Golden(
        inputs={"expr": (10.0, 11.0, 13.0, 12.0, 14.0, 13.0, 15.0)},
        output=(None, None, 56.3099, 26.5651, 26.5651, 26.5651, 26.5651),
        params={"window": 3},
    ),
)
