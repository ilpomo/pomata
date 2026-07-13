"""Spec for ``pomata.indicators.linear_regression_slope`` — the rolling least-squares slope, degree-1 homogeneous."""

from tests.indicators.oracles import linear_regression_slope_reference
from tests_new.support.spec import ScaleAxis, Shape, Spec

from pomata.indicators import linear_regression_slope

LINEAR_REGRESSION_SLOPE = Spec(
    factory=linear_regression_slope,
    inputs=("expr",),
    params={"window": 14},
    shape=Shape.SERIES,
    warmup=13,
    raises=(({"window": 1}, r"window must be >= 2"),),
    oracle=linear_regression_slope_reference,
    # The slope of the window's least-squares line (per-bar rise), homogeneous of degree 1 (tests/indicators/
    # test_linear_regression_slope.py::TestLinearRegressionSlopeProperties::test_scale_homogeneity).
    scale=(ScaleAxis(roles=("expr",), degree=1),),
    golden_params={"window": 3},
    golden_input={"expr": (10.0, 11.0, 13.0, 12.0, 14.0, 13.0, 15.0)},
    golden_output=(None, None, 1.5, 0.5, 0.5, 0.5, 0.5),
)
