"""Spec for ``pomata.indicators.linear_regression_angle`` — the least-squares slope as a degree angle, scale-exempt."""

from tests_new.indicators.oracles import linear_regression_angle_reference
from tests_new.support.spec import ScaleExempt, Shape, Spec

from pomata.indicators import linear_regression_angle

LINEAR_REGRESSION_ANGLE = Spec(
    factory=linear_regression_angle,
    inputs=("expr",),
    params={"window": 14},
    shape=Shape.SERIES,
    warmup=13,
    raises=(({"window": 1}, r"window must be >= 2"),),
    oracle=linear_regression_angle_reference,
    # The arctangent of the slope in degrees: neither scale-invariant nor degree-1 homogeneous (a rescaling changes
    # the slope inside the atan), bounded in (-90, 90) (tests/indicators/test_linear_regression_angle.py sizing note).
    scale=ScaleExempt(
        reason="atan(slope) in degrees: a rescaling scales the slope inside the arctangent, so the angle is neither "
        "invariant nor degree-1 homogeneous; it is a bounded O(1) value in (-90, 90)"
    ),
    golden_params={"window": 3},
    golden_input={"expr": (10.0, 11.0, 13.0, 12.0, 14.0, 13.0, 15.0)},
    golden_output=(None, None, 56.3099, 26.5651, 26.5651, 26.5651, 26.5651),
)
